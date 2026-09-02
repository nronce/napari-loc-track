"""The rendering engine: geometry, flux, grouping, backends and saving.

A reconstruction is a measurement, not a picture: if the splat loses or invents
signal, or lands half a super-resolved pixel off, everything downstream reads a
distorted structure. These tests pin the parts that would be invisible by eye -
that the total is the localization count, that a spot sits where it was located,
and that the three backends agree - alongside the mechanics of progress,
cancellation and metadata.
"""
import json
import math

import numpy as np
import pytest

from conftest import load_render

render = load_render()


# The renderer covers [-0.5, W-0.5) in camera pixels, exactly the extent of a
# W-pixel-wide image. Test data is generated inside it so nothing is clipped.
FIELD = (32, 32)
LOW, HIGH = -0.5, 31.5


@pytest.fixture
def spots():
    rng = np.random.default_rng(0)
    n = 5000
    return {
        "x": rng.uniform(LOW, HIGH, n),
        "y": rng.uniform(LOW, HIGH, n),
        "sigma": rng.uniform(0.2, 0.6, n),
        "photons": rng.uniform(100, 5000, n),
        "frames": rng.integers(0, 40, n),
        "n": n,
    }


# --- geometry ---------------------------------------------------------------


def test_output_shape_scales_with_oversampling():
    assert render.output_shape((10, 20), 4) == (40, 80)
    assert render.estimate_bytes((10, 20), 4, n_frames=3) == 40 * 80 * 3 * 4


@pytest.mark.parametrize("oversampling", [1, 2, 5, 10])
def test_a_localization_lands_where_it_was_located(oversampling):
    """The bin a spot falls in must map back to its own coordinate in napari.

    This is the half-pixel that separates a correct overlay from a render that
    looks like the sample drifted.
    """
    # deliberately not on a bin edge for any of these oversamplings
    position = np.array([7.31])
    image = render.render_frame(position, position, shape=(16, 16),
                                oversampling=oversampling, mode="histogram")
    row, col = np.argwhere(image > 0)[0]
    scale, translate = render.layer_transform(oversampling)
    assert abs(translate[1] + col * scale[1] - 7.31) <= 0.5 / oversampling
    assert abs(translate[0] + row * scale[0] - 7.31) <= 0.5 / oversampling


@pytest.mark.parametrize("oversampling", [1, 3, 10])
def test_the_render_grid_covers_the_image_edge_to_edge(oversampling):
    """The outer edges of the render must land on the edges of the image.

    Checked against the whole grid rather than one spot: an error here is a
    uniform sub-pixel offset, which looks exactly like sample drift in the
    finished reconstruction rather than like a bug.
    """
    field = (32, 48)
    rows, cols = render.output_shape(field, oversampling)
    scale, translate = render.layer_transform(oversampling)
    # napari places pixel i centred at translate + i * scale
    first_edge = translate[1] - scale[1] / 2
    last_edge = translate[1] + (cols - 1) * scale[1] + scale[1] / 2
    assert first_edge == pytest.approx(-0.5)
    assert last_edge == pytest.approx(field[1] - 0.5)
    first_edge = translate[0] - scale[0] / 2
    last_edge = translate[0] + (rows - 1) * scale[0] + scale[0] / 2
    assert first_edge == pytest.approx(-0.5)
    assert last_edge == pytest.approx(field[0] - 0.5)


def test_transform_composes_with_the_source_layer():
    scale, translate = render.layer_transform(
        4, origin=(-0.5, -0.5), source_scale=(0.1, 0.1), source_translate=(2.0, 3.0)
    )
    assert scale == pytest.approx((0.025, 0.025))
    assert translate == pytest.approx((2.0 - 0.1 * 0.375, 3.0 - 0.1 * 0.375))


# --- what the pixel values mean ---------------------------------------------


def test_histogram_counts_every_localization(spots):
    image = render.render_frame(spots["x"], spots["y"], shape=FIELD,
                                oversampling=8, mode="histogram")
    assert image.dtype == np.float32
    assert image.sum() == spots["n"]


def test_histogram_matches_numpy(spots):
    image = render.render_frame(spots["x"], spots["y"], shape=FIELD,
                                oversampling=8, mode="histogram")
    reference, _, _ = np.histogram2d(
        (spots["y"] + 0.5) * 8, (spots["x"] + 0.5) * 8,
        bins=[256, 256], range=[[0, 256], [0, 256]],
    )
    assert np.array_equal(image, reference.astype(np.float32))


def test_scatter_is_one_dot_per_occupied_bin(spots):
    counts = render.render_frame(spots["x"], spots["y"], shape=FIELD,
                                 oversampling=8, mode="histogram")
    dots = render.render_frame(spots["x"], spots["y"], shape=FIELD,
                               oversampling=8, mode="scatter")
    assert set(np.unique(dots)) <= {0.0, 1.0}
    assert dots.sum() == (counts > 0).sum()


def test_photon_weighting_sums_photons(spots):
    image = render.render_frame(spots["x"], spots["y"], shape=FIELD, oversampling=8,
                                mode="histogram", weights=spots["photons"])
    assert image.sum() == pytest.approx(spots["photons"].sum(), rel=1e-4)


@pytest.mark.parametrize("mode,extra", [
    ("gaussian_local", {"sigma_px": np.full(1, 0.5)}),
    ("gaussian_global", {"global_sigma_px": 0.5}),
])
def test_a_gaussian_carries_exactly_one_localization(mode, extra):
    """Blurring must move signal around, never create or destroy it."""
    centre = np.array([16.0])
    image = render.render_frame(centre, centre, shape=FIELD, oversampling=8,
                                mode=mode, **extra)
    assert image.sum() == pytest.approx(1.0, rel=1e-3)


def test_gaussian_width_is_the_requested_sigma():
    centre = np.array([16.0])
    image = render.render_frame(centre, centre, shape=FIELD, oversampling=8,
                                mode="gaussian_local", sigma_px=np.array([0.5]))
    rows = np.arange(image.shape[0])
    weight = image.sum(axis=1)
    mean = (weight * rows).sum() / weight.sum()
    variance = (weight * (rows - mean) ** 2).sum() / weight.sum()
    assert math.sqrt(variance) == pytest.approx(0.5 * 8, abs=0.15)


def test_gaussians_stay_sub_pixel():
    """A shift smaller than one camera pixel has to survive into the render."""
    columns = np.arange(FIELD[1] * 4)
    centres = []
    for x in (16.0, 16.1):
        image = render.render_frame(np.array([x]), np.array([16.0]), shape=FIELD,
                                    oversampling=4, mode="gaussian_local",
                                    sigma_px=np.array([0.5]))
        weight = image.sum(axis=0)
        centres.append((weight * columns).sum() / weight.sum())
    assert centres[1] - centres[0] == pytest.approx(0.1 * 4, abs=0.02)


def test_localizations_without_a_precision_are_still_drawn():
    """A NaN width must not silently delete the molecule from the picture."""
    positions = np.full(10, 16.0)
    sigma = np.full(10, np.nan)
    image = render.render_frame(positions, positions, shape=FIELD, oversampling=4,
                                mode="gaussian_local", sigma_px=sigma)
    assert image.sum() == pytest.approx(10.0, rel=1e-3)


def test_an_absurd_precision_cannot_paint_the_whole_image():
    huge = np.array([1e6])
    image = render.render_frame(np.array([16.0]), np.array([16.0]), shape=FIELD,
                                oversampling=4, mode="gaussian_local", sigma_px=huge)
    lit = np.argwhere(image > 0)
    span = lit.max(axis=0) - lit.min(axis=0)
    assert (span <= 2 * render.MAX_RADIUS).all()


# --- edges ------------------------------------------------------------------


def test_the_field_of_view_is_exactly_the_image():
    image = render.render_frame(
        np.array([LOW, 31.49, HIGH, 100.0]), np.array([LOW, 31.49, HIGH, 100.0]),
        shape=FIELD, oversampling=4, mode="histogram",
    )
    assert image.sum() == 2  # both corners in, the edge and the stray one out
    assert image[0, 0] == 1
    assert image[-1, -1] == 1


def test_a_spot_on_the_border_keeps_only_the_part_that_is_visible():
    image = render.render_frame(np.array([-0.4]), np.array([16.0]), shape=FIELD,
                                oversampling=4, mode="gaussian_local",
                                sigma_px=np.array([0.5]))
    assert 0.3 < image.sum() < 0.75


def test_empty_and_non_finite_input(spots):
    empty = render.render_frame(np.array([]), np.array([]), shape=(8, 8),
                                oversampling=2, mode="histogram")
    assert empty.shape == (16, 16) and empty.sum() == 0
    mixed = render.render_frame(np.array([1.0, np.nan]), np.array([1.0, 2.0]),
                                shape=(8, 8), oversampling=2, mode="histogram")
    assert mixed.sum() == 1


def test_an_unknown_mode_is_refused(spots):
    with pytest.raises(ValueError):
        render.render_frame(spots["x"], spots["y"], shape=FIELD,
                            oversampling=2, mode="rainbow")


# --- backends ---------------------------------------------------------------


@pytest.mark.parametrize("mode,extra", [
    ("histogram", {}),
    ("scatter", {}),
    ("gaussian_local", {"sigma_px": None}),
    ("gaussian_global", {"global_sigma_px": 0.4}),
])
def test_the_vectorized_backend_agrees_with_the_numba_one(spots, monkeypatch, mode, extra):
    """The vectorized splat is what the GPU runs, so it must match the CPU kernel.

    CuPy is rarely installed on a test machine; running the same code path
    through numpy is the closest check available that the GPU render would not
    quietly differ from the CPU one.
    """
    if not render.is_numba_available():
        pytest.skip("numba not installed, so there is only one backend")
    if "sigma_px" in extra:
        extra = {"sigma_px": spots["sigma"]}
    kwargs = dict(shape=FIELD, oversampling=4, mode=mode, **extra)

    with_numba = render.render_frame(spots["x"], spots["y"], **kwargs)
    monkeypatch.setattr(render, "numba", None)
    vectorized = render.render_frame(spots["x"], spots["y"], **kwargs)

    assert vectorized == pytest.approx(with_numba, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("mode,extra", [
    ("histogram", {}),
    ("scatter", {}),
    ("gaussian_local", {"sigma_px": None}),
    ("gaussian_global", {"global_sigma_px": 0.4}),
])
def test_the_gpu_and_the_cpu_produce_the_same_render(spots, mode, extra):
    """Which backend ran must never be visible in the result."""
    if not render.is_render_gpu_available():
        pytest.skip("no CUDA device / CuPy for this run")
    if "sigma_px" in extra:
        extra = {"sigma_px": spots["sigma"]}
    kwargs = dict(shape=FIELD, oversampling=4, mode=mode, **extra)

    on_gpu = render.render_frame(spots["x"], spots["y"], gpu=True, **kwargs)
    on_cpu = render.render_frame(spots["x"], spots["y"], gpu=False, **kwargs)

    assert on_gpu.dtype == on_cpu.dtype == np.float32
    # float32 accumulation in a different order, so not bit-identical
    assert on_gpu == pytest.approx(on_cpu, rel=1e-4, abs=1e-5)
    assert on_gpu.sum() == pytest.approx(on_cpu.sum(), rel=1e-5)


def test_choose_backend_explains_itself():
    use_gpu, why = render.choose_backend(FIELD, 4, prefer_gpu=False)
    assert use_gpu is False and "not requested" in why
    use_gpu, why = render.choose_backend(FIELD, 4, prefer_gpu=True)
    assert use_gpu is render.is_render_gpu_available() and why


# --- grouping raw frames into movie frames ----------------------------------


def test_block_grouping():
    assert render.group_bounds(0, 9, 4, "blocks") == [(0, 4), (4, 8), (8, 10)]


def test_cumulative_grouping_grows_from_the_first_frame():
    assert render.group_bounds(0, 9, 4, "cumulative") == [(0, 4), (0, 8), (0, 10)]


def test_sliding_grouping_overlaps_by_the_step():
    assert render.group_bounds(0, 9, 4, "sliding", 2) == [
        (0, 4), (2, 6), (4, 8), (6, 10), (8, 10)]
    # no step given: half a window, the usual smooth-movie default
    assert render.group_bounds(0, 9, 4, "sliding") == render.group_bounds(0, 9, 4, "sliding", 2)


def test_grouping_handles_a_single_frame_and_a_bad_name():
    assert render.group_bounds(5, 5, 10, "blocks") == [(5, 6)]
    with pytest.raises(ValueError):
        render.group_bounds(0, 9, 4, "spiral")
    with pytest.raises(ValueError):
        render.group_count(0, 9, 4, "spiral")


@pytest.mark.parametrize("grouping", list(render.GROUPINGS))
@pytest.mark.parametrize("first,last,size,step", [
    (0, 9, 4, 2), (0, 0, 1, 1), (3, 3, 10, 5), (0, 999, 1, 1),
    (100, 250, 7, 3), (0, 9, 100, 100), (5, 44, 10, 10),
])
def test_the_frame_count_always_matches_the_frames(grouping, first, last, size, step):
    """The size guard budgets from the count and the render allocates from the
    bounds; if they ever disagree, the guard is protecting the wrong number."""
    assert render.group_count(first, last, size, grouping, step) == len(
        render.group_bounds(first, last, size, grouping, step))


# --- movies -----------------------------------------------------------------


def test_a_block_movie_partitions_the_localizations(spots):
    movie = render.render_movie(spots["x"], spots["y"], spots["frames"], shape=FIELD,
                                oversampling=4, frames_per_group=10, grouping="blocks",
                                mode="histogram")
    assert movie.shape == (4, 128, 128)
    assert movie.sum() == spots["n"]
    expected = [int(((spots["frames"] >= lo) & (spots["frames"] < hi)).sum())
                for lo, hi in render.group_bounds(0, 39, 10)]
    assert [int(frame.sum()) for frame in movie] == expected


def test_a_movie_frame_equals_rendering_that_group_alone(spots):
    movie = render.render_movie(spots["x"], spots["y"], spots["frames"], shape=FIELD,
                                oversampling=4, frames_per_group=10, mode="histogram")
    inside = (spots["frames"] >= 10) & (spots["frames"] < 20)
    direct = render.render_frame(spots["x"][inside], spots["y"][inside], shape=FIELD,
                                 oversampling=4, mode="histogram")
    assert np.array_equal(movie[1], direct)


def test_a_cumulative_movie_builds_up_to_the_full_render(spots):
    blocks = render.render_movie(spots["x"], spots["y"], spots["frames"], shape=FIELD,
                                 oversampling=4, frames_per_group=10, grouping="blocks",
                                 mode="histogram")
    cumulative = render.render_movie(spots["x"], spots["y"], spots["frames"], shape=FIELD,
                                     oversampling=4, frames_per_group=10,
                                     grouping="cumulative", mode="histogram")
    assert cumulative == pytest.approx(np.cumsum(blocks, axis=0))
    whole = render.render_frame(spots["x"], spots["y"], shape=FIELD, oversampling=4,
                                mode="histogram")
    assert cumulative[-1] == pytest.approx(whole)


def test_a_cumulative_scatter_movie_only_ever_lights_pixels_up(spots):
    movie = render.render_movie(spots["x"], spots["y"], spots["frames"], shape=FIELD,
                                oversampling=4, frames_per_group=10,
                                grouping="cumulative", mode="scatter")
    assert set(np.unique(movie)) <= {0.0, 1.0}
    assert (np.diff(movie, axis=0) >= 0).all()


def test_a_sliding_window_is_the_sum_of_the_blocks_it_covers(spots):
    blocks = render.render_movie(spots["x"], spots["y"], spots["frames"], shape=FIELD,
                                 oversampling=4, frames_per_group=10, mode="histogram")
    sliding = render.render_movie(spots["x"], spots["y"], spots["frames"], shape=FIELD,
                                  oversampling=4, frames_per_group=20, grouping="sliding",
                                  step=10, mode="histogram")
    assert sliding.shape[0] == len(render.group_bounds(0, 39, 20, "sliding", 10))
    assert sliding[0] == pytest.approx(blocks[0] + blocks[1])


def test_a_movie_with_no_localizations_is_empty_not_broken():
    movie = render.render_movie(np.array([]), np.array([]), np.array([]), shape=(8, 8),
                                oversampling=2, frames_per_group=5, mode="histogram")
    assert movie.shape == (0, 16, 16)


# --- progress and cancellation ----------------------------------------------


def test_progress_is_monotonic_and_reaches_one(spots):
    seen = []
    generator = render.render_movie_iter(
        spots["x"], spots["y"], spots["frames"], shape=FIELD, oversampling=4,
        frames_per_group=10, mode="histogram")
    while True:
        try:
            seen.append(next(generator))
        except StopIteration:
            break
    assert seen == sorted(seen)
    assert all(0.0 <= value <= 1.0 for value in seen)
    assert seen[-1] == pytest.approx(1.0)


def test_a_render_can_be_abandoned_part_way(spots):
    generator = render.render_frame_iter(spots["x"], spots["y"], shape=FIELD,
                                         oversampling=4, mode="histogram")
    next(generator)
    generator.close()  # what the worker does on cancel; must not raise


# --- saving -----------------------------------------------------------------


def test_saving_writes_the_render_its_metadata_and_a_preview(tmp_path, spots):
    tifffile = pytest.importorskip("tifffile")
    image = render.render_frame(spots["x"], spots["y"], shape=FIELD, oversampling=4,
                                mode="histogram")
    settings = {"mode": "histogram", "oversampling": 4, "n_localizations": spots["n"]}

    written = render.save_render(tmp_path / "recon.tif", image, settings,
                                 super_pixel_size_nm=25.0, png=True)

    assert [path.name for path in written] == [
        "recon.tif", "recon_metadata.json", "recon.png"]
    # the values are the render's own, not a display stretch
    assert np.array_equal(tifffile.imread(tmp_path / "recon.tif"), image)
    sidecar = json.loads((tmp_path / "recon_metadata.json").read_text(encoding="utf-8"))
    assert sidecar["mode"] == "histogram"
    assert sidecar["super_resolved_pixel_size_nm"] == 25.0
    assert sidecar["image_shape"] == list(image.shape)
    assert (tmp_path / "recon.png").stat().st_size > 0


def test_the_tiff_carries_the_pixel_size_and_the_settings(tmp_path, spots):
    tifffile = pytest.importorskip("tifffile")
    image = render.render_frame(spots["x"], spots["y"], shape=FIELD, oversampling=4,
                                mode="histogram")
    render.save_render(tmp_path / "recon.tif", image, {"mode": "histogram"},
                       super_pixel_size_nm=25.0)

    with tifffile.TiffFile(tmp_path / "recon.tif") as handle:
        assert json.loads(handle.imagej_metadata["Info"])["mode"] == "histogram"
        assert handle.imagej_metadata["unit"] == "um"
        # 25 nm = 0.025 um, so 40 pixels per micron
        numerator, denominator = handle.pages[0].tags["XResolution"].value
        assert numerator / denominator == pytest.approx(40.0)


def test_a_saved_movie_records_its_time_axis(tmp_path, spots):
    tifffile = pytest.importorskip("tifffile")
    movie = render.render_movie(spots["x"], spots["y"], spots["frames"], shape=FIELD,
                                oversampling=4, frames_per_group=10, mode="histogram")
    render.save_render(tmp_path / "movie.tif", movie, {"grouping": "blocks"},
                       super_pixel_size_nm=25.0, frame_interval_s=0.5)

    with tifffile.TiffFile(tmp_path / "movie.tif") as handle:
        assert handle.imagej_metadata["finterval"] == 0.5
        assert json.loads(handle.imagej_metadata["Info"])["axes"] == "TYX"
    assert np.array_equal(tifffile.imread(tmp_path / "movie.tif"), movie)


# --- display, colour and compositing ----------------------------------------


def test_one_stretch_is_shared_by_the_whole_stack():
    """The stretch must not be recomputed per frame, or a movie pulses."""
    stack = np.zeros((3, 4, 4), np.float32)
    stack[0, 0, 0] = 1.0
    stack[1, 0, 0] = 5.0
    stack[2, 0, 0] = 10.0
    eight_bit = render.to_uint8(stack, limits=(0.0, 10.0))
    assert eight_bit.dtype == np.uint8
    assert [int(frame.max()) for frame in eight_bit] == [25, 127, 255]  # truncated
    # a per-frame stretch would have put 255 in every frame
    assert eight_bit[0].max() < eight_bit[-1].max()


def test_an_empty_render_stretches_to_black_not_to_noise():
    assert render.to_uint8(np.zeros((4, 4), np.float32)).max() == 0


def test_colorize_puts_a_colour_only_in_its_own_channels(spots):
    image = render.render_frame(spots["x"], spots["y"], shape=FIELD, oversampling=4,
                                mode="histogram")
    yellow = render.colorize(image, color="yellow")
    assert yellow.shape == image.shape + (3,)
    assert yellow.dtype == np.uint8
    assert yellow[..., 0].max() > 0 and yellow[..., 1].max() > 0
    assert yellow[..., 2].max() == 0
    # brightness still follows the render
    assert np.argmax(yellow[..., 0]) == np.argmax(render.to_uint8(image))


def test_colorize_accepts_a_colormap(spots):
    image = render.render_frame(spots["x"], spots["y"], shape=FIELD, oversampling=4,
                                mode="histogram")
    coloured = render.colorize(image, colormap="magma")
    assert coloured.shape == image.shape + (3,)
    assert coloured[image == 0].max() <= 5  # magma bottoms out at near-black


def test_blending_is_additive_and_clips():
    red = np.zeros((2, 2, 3), np.uint8)
    red[..., 0] = 200
    green = np.zeros((2, 2, 3), np.uint8)
    green[..., 1] = 100
    more_red = np.zeros((2, 2, 3), np.uint8)
    more_red[..., 0] = 200

    blended = render.blend_additive([red, green])
    assert blended[0, 0].tolist() == [200, 100, 0]
    assert render.blend_additive([red, more_red])[0, 0, 0] == 255  # clipped, not wrapped
    with pytest.raises(ValueError):
        render.blend_additive([])


# --- trajectories as lines ---------------------------------------------------


def test_a_trajectory_becomes_points_along_its_path():
    x = np.array([0.0, 10.0])
    y = np.array([0.0, 0.0])
    sample_x, sample_y, sample_frame = render.trajectory_samples(
        x, y, np.array([0, 1]), np.array([7, 7]), spacing_px=1.0)
    assert sample_x.size == 10
    assert sample_y == pytest.approx(np.zeros(10))
    assert sample_x.min() == pytest.approx(0.0)
    assert sample_x.max() == pytest.approx(9.0)
    # every sample belongs to the frame the segment starts on
    assert set(sample_frame.tolist()) == {0}


def test_separate_trajectories_are_never_joined():
    """A line from the end of one track to the start of the next would be pure
    fiction, and is exactly what a naive polyline would draw."""
    x = np.array([0.0, 1.0, 50.0, 51.0])
    y = np.zeros(4)
    frames = np.array([0, 1, 0, 1])
    particles = np.array([0, 0, 1, 1])
    sample_x, _sample_y, _frames = render.trajectory_samples(
        x, y, frames, particles, spacing_px=0.25)
    # nothing sampled in the gap between the two trajectories
    assert not ((sample_x > 2.0) & (sample_x < 49.0)).any()


def test_trajectory_samples_survive_degenerate_input():
    empty = np.empty(0)
    assert render.trajectory_samples(empty, empty, empty, empty, 0.5)[0].size == 0
    # a single point is not a segment
    lone = render.trajectory_samples(
        np.array([1.0]), np.array([1.0]), np.array([0]), np.array([0]), 0.5)
    assert lone[0].size == 0
    # a track that never moves still gets drawn as a dot
    still = render.trajectory_samples(
        np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([0, 1]), np.array([0, 0]), 0.5)
    assert still[0].size == 1


def test_a_movie_can_be_told_which_acquisition_it_covers():
    """An overlay spanning fewer frames must still get the base's movie frames."""
    x = np.full(4, 16.0)
    y = np.full(4, 16.0)
    frames = np.array([0, 1, 2, 3])  # only the very start of a long acquisition
    movie = render.render_movie(x, y, frames, shape=FIELD, oversampling=2,
                                frames_per_group=10, mode="histogram",
                                frame_range=(0, 39))
    assert movie.shape[0] == 4               # not 1, as its own range would give
    assert movie[0].sum() == 4               # all of it lands in the first group
    assert movie[1:].sum() == 0


def test_an_overlay_with_no_points_still_matches_the_movie_length():
    empty = np.empty(0)
    movie = render.render_movie(empty, empty, empty, shape=FIELD, oversampling=2,
                                frames_per_group=10, mode="histogram",
                                frame_range=(0, 39))
    assert movie.shape == (4,) + render.output_shape(FIELD, 2)
    assert movie.sum() == 0


# --- time stamps -------------------------------------------------------------


@pytest.fixture(scope="module")
def atlas():
    return render.glyph_atlas(24)


def test_the_glyphs_come_out_at_the_requested_height():
    for height in (12, 24, 60):
        small = render.glyph_atlas(height)
        assert abs(small["0"].shape[0] - height) <= 2
        assert small["0"].max() > 0.5  # actually has ink in it


def test_text_is_laid_out_left_to_right(atlas):
    one = render.compose_text(atlas, "1")
    twelve = render.compose_text(atlas, "12")
    assert twelve.shape[1] > one.shape[1]
    assert twelve.shape[0] == pytest.approx(one.shape[0], abs=4)
    assert render.compose_text(atlas, "").size <= 1


def test_time_is_formatted_for_the_length_of_the_movie():
    assert render.format_time(12.34, longest=50) == "12.3 s"
    assert render.format_time(83.0, longest=200) == "01:23"
    assert render.format_time(3723.0, longest=7200) == "01:02:03"
    # the whole movie shares one format, so labels don't change shape part way
    assert render.format_time(5.0, longest=200) == "00:05"


def test_burning_a_label_writes_into_the_pixels(atlas):
    frame = np.zeros((200, 300), np.uint8)
    render.burn_text(frame, render.compose_text(atlas, "12.5 s"), position="top left")
    assert frame.max() > 200
    assert frame[:100, :200].sum() > 0      # ink in the top-left
    assert frame[150:, 200:].sum() == 0     # and nowhere else


@pytest.mark.parametrize("position", ["top left", "top right", "bottom left", "bottom right"])
def test_the_label_goes_where_it_is_asked(atlas, position):
    frame = np.zeros((200, 300), np.uint8)
    render.burn_text(frame, render.compose_text(atlas, "9 s"), position=position)
    rows, cols = np.nonzero(frame)
    in_top = rows.mean() < 100
    in_left = cols.mean() < 150
    assert in_top == ("top" in position)
    assert in_left == ("left" in position)


def test_a_label_on_a_composite_keeps_its_colour(atlas):
    frame = np.zeros((200, 300, 3), np.uint8)
    render.burn_text(frame, render.compose_text(atlas, "1 s"), color="yellow")
    assert frame[..., 0].max() > 200 and frame[..., 1].max() > 200
    assert frame[..., 2].max() == 0


def test_a_label_bigger_than_the_frame_is_skipped_not_crashed():
    frame = np.zeros((10, 10), np.uint8)
    render.burn_text(frame, render.glyph_atlas(200)["0"], position="top left")
    assert frame.max() == 0  # nothing drawn, nothing raised


def test_burning_a_label_never_wraps_a_bright_pixel(atlas):
    frame = np.full((200, 300), 250, np.uint8)
    render.burn_text(frame, render.compose_text(atlas, "8"), position="top left")
    assert frame.max() == 255  # clipped, not wrapped around to black


# --- scale bar ---------------------------------------------------------------


@pytest.mark.parametrize("span_nm", [500, 2_000, 6_400, 25_000, 120_000, 1_500_000])
def test_the_default_bar_is_a_readable_fraction_of_the_view(span_nm):
    length = render.nice_scale_length(span_nm)
    assert 0.05 * span_nm <= length <= 0.4 * span_nm
    # and a number nobody has to decode: 1, 2 or 5 times a power of ten
    mantissa = length / 10.0 ** math.floor(math.log10(length))
    assert min(abs(mantissa - step) for step in (1.0, 2.0, 5.0)) < 1e-9


def test_the_default_bar_grows_with_the_view():
    lengths = [render.nice_scale_length(span) for span in (5_000, 50_000, 500_000)]
    assert lengths == sorted(lengths)
    assert lengths[0] < lengths[-1]


def test_a_bar_is_labelled_in_the_unit_a_reader_expects():
    assert render.format_length(500) == "500 nm"
    assert render.format_length(1000) == "1 µm"
    assert render.format_length(2500) == "2.5 µm"
    assert render.format_length(20000) == "20 µm"


def test_the_bar_mask_is_the_bar_and_its_label_together(atlas):
    label = render.compose_text(atlas, "1 µm")
    bar_only = render.scale_bar_mask(100, 6)
    assert bar_only.shape == (6, 100)
    assert bar_only.min() == 1.0  # solid

    with_label = render.scale_bar_mask(100, 6, label)
    assert with_label.shape[0] > bar_only.shape[0]
    assert with_label.shape[1] >= 100
    # the solid bar is along the bottom, the label above it
    assert with_label[-1].sum() >= 100
    assert with_label[0].sum() > 0
    assert with_label[0].sum() < with_label[-1].sum()


def test_the_micron_sign_actually_renders(atlas):
    assert "µ" in atlas
    assert atlas["µ"].max() > 0.5  # the font really has the glyph


def test_a_bar_is_drawn_where_it_is_asked():
    frame = np.zeros((300, 400), np.uint8)
    render.burn_text(frame, render.scale_bar_mask(120, 8), position="bottom right")
    rows, cols = np.nonzero(frame)
    assert rows.mean() > 150 and cols.mean() > 200
    assert frame.max() == 255
    # the bar is as long as it was asked to be
    assert (cols.max() - cols.min() + 1) == 120


# --- other layers on the grid, and cropping ----------------------------------


def test_a_camera_image_is_sampled_onto_the_render_grid():
    camera = np.arange(16, dtype=np.float32).reshape(4, 4)
    grid = render.resample_to_grid(camera, shape=(4, 4), origin=(-0.5, -0.5),
                                   oversampling=3)
    assert grid.shape == (12, 12)
    # each camera pixel becomes a 3x3 block of itself: nearest neighbour, so
    # no values that were never measured
    assert set(np.unique(grid)) == set(np.unique(camera))
    assert grid[0, 0] == camera[0, 0]
    assert grid[-1, -1] == camera[-1, -1]
    assert (grid[0:3, 0:3] == camera[0, 0]).all()


def test_resampling_honours_the_layer_transform():
    camera = np.arange(16, dtype=np.float32).reshape(4, 4)
    shifted = render.resample_to_grid(
        camera, shape=(4, 4), origin=(-0.5, -0.5), oversampling=1,
        source_scale=(1.0, 1.0), source_translate=(1.0, 0.0))
    # the layer sits one pixel lower, so the grid reads one row earlier
    assert (shifted[1:] == camera[:-1]).all()


def test_a_box_becomes_the_right_super_resolved_slices():
    rows, cols = render.box_to_slices((10.0, 20.0, 20.0, 40.0), shape=(64, 64),
                                      origin=(-0.5, -0.5), oversampling=4)
    assert rows == slice(42, 82)     # (10 + 0.5) * 4 .. (20 + 0.5) * 4
    assert cols == slice(82, 162)


def test_a_box_is_clamped_to_the_render():
    rows, cols = render.box_to_slices((-1000, -1000, 1000, 1000), shape=(8, 8),
                                      origin=(-0.5, -0.5), oversampling=2)
    assert rows == slice(0, 16) and cols == slice(0, 16)
    # a box entirely outside still yields something croppable, never empty
    rows, cols = render.box_to_slices((900, 900, 1000, 1000), shape=(8, 8),
                                      origin=(-0.5, -0.5), oversampling=2)
    assert rows.stop > rows.start and cols.stop > cols.start


def test_cropping_leaves_the_time_and_colour_axes_alone():
    rows, cols = slice(2, 6), slice(1, 4)
    grey_image = np.zeros((10, 10), np.float32)
    assert render.crop(grey_image, rows, cols).shape == (4, 3)
    movie = np.zeros((7, 10, 10), np.float32)
    assert render.crop(movie, rows, cols, is_movie=True).shape == (7, 4, 3)
    rgb = np.zeros((10, 10, 3), np.uint8)
    assert render.crop(rgb, rows, cols).shape == (4, 3, 3)
    rgb_movie = np.zeros((7, 10, 10, 3), np.uint8)
    assert render.crop(rgb_movie, rows, cols, is_movie=True).shape == (7, 4, 3, 3)


def test_a_crop_keeps_the_pixels_it_covers(spots):
    image = render.render_frame(spots["x"], spots["y"], shape=FIELD, oversampling=4,
                                mode="histogram")
    rows, cols = render.box_to_slices((8.0, 8.0, 16.0, 16.0), shape=FIELD,
                                      origin=(-0.5, -0.5), oversampling=4)
    cropped = render.crop(image, rows, cols)
    assert np.array_equal(cropped, image[rows, cols])
    assert cropped.sum() < image.sum()      # something really was cut away
    assert cropped.sum() > 0


# --- saving the display and composite formats -------------------------------


def test_an_eight_bit_movie_is_written_as_such(tmp_path, spots):
    tifffile = pytest.importorskip("tifffile")
    movie = render.render_movie(spots["x"], spots["y"], spots["frames"], shape=FIELD,
                                oversampling=2, frames_per_group=10, mode="histogram")
    light = render.to_uint8(movie)

    render.save_render(tmp_path / "light.tif", light, {"kind": "movie"},
                       super_pixel_size_nm=50.0, frame_interval_s=0.1)

    written = tifffile.imread(tmp_path / "light.tif")
    assert written.dtype == np.uint8
    assert np.array_equal(written, light)
    assert light.nbytes * 4 == movie.nbytes
    sidecar = json.loads(
        (tmp_path / "light_metadata.json").read_text(encoding="utf-8"))
    assert sidecar["dtype"] == "uint8"
    assert sidecar["axes"] == "TYX"
    assert "display levels" in sidecar["value_units"]


def test_an_rgb_composite_round_trips_with_its_channels(tmp_path, spots):
    tifffile = pytest.importorskip("tifffile")
    image = render.render_frame(spots["x"], spots["y"], shape=FIELD, oversampling=2,
                                mode="histogram")
    composite = render.blend_additive([
        render.colorize(image, colormap="magma"),
        render.colorize(image, color="cyan"),
    ])

    render.save_render(tmp_path / "composite.tif", composite, {"kind": "composite"},
                       super_pixel_size_nm=50.0, png=True)

    written = tifffile.imread(tmp_path / "composite.tif")
    assert written.shape == image.shape + (3,)
    assert np.array_equal(written, composite)
    sidecar = json.loads(
        (tmp_path / "composite_metadata.json").read_text(encoding="utf-8"))
    assert sidecar["axes"] == "YXS"
    assert (tmp_path / "composite.png").stat().st_size > 0


def test_a_composite_movie_keeps_its_time_and_colour_axes(tmp_path, spots):
    tifffile = pytest.importorskip("tifffile")
    movie = render.render_movie(spots["x"], spots["y"], spots["frames"], shape=FIELD,
                                oversampling=2, frames_per_group=10, mode="histogram")
    composite = render.colorize(movie, color="green")

    render.save_render(tmp_path / "cmovie.tif", composite, {}, super_pixel_size_nm=50.0,
                       frame_interval_s=0.25, png=True)

    with tifffile.TiffFile(tmp_path / "cmovie.tif") as handle:
        assert json.loads(handle.imagej_metadata["Info"])["axes"] == "TYXS"
        assert handle.imagej_metadata["finterval"] == 0.25
    assert np.array_equal(tifffile.imread(tmp_path / "cmovie.tif"), composite)


def test_contrast_limits_ignore_the_empty_background(spots):
    image = render.render_frame(spots["x"], spots["y"], shape=FIELD, oversampling=8,
                                mode="histogram")
    low, high = render.contrast_limits(image)
    assert low == 0.0
    assert 0 < high <= image.max()
    assert render.contrast_limits(np.zeros((4, 4), np.float32)) == (0.0, 1.0)
