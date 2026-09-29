"""The movie of a growth deformation: drawn from a measured record, and written."""
import importlib

import numpy as np

conftest = importlib.import_module("conftest")
deform = conftest.load_deform()
movie = conftest._load_standalone("napari_loc_track._deform_movie", "_deform_movie.py")
test_deform = importlib.import_module("test_deform")


def _measure():
    stack, _maps = test_deform._snapshots(K=3)
    region = (160, test_deform.SIZE - 160, 160, test_deform.SIZE - 160)
    record = deform.measure_deformation(stack, region=region, patch=128, step=64, lags=(1, 2),
                                        threads=2, backend="cpu")
    return record, stack


def test_the_movie_shows_every_stage_and_every_snapshot(tmp_path):
    record, stack = _measure()
    gen = movie.movie_frames_iter(record, stack, translation=None)
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            frames = stop.value
            break
    stages = len(record.steps["forward"]) + 1        # as recorded, then every map
    assert len(frames) == stages + len(record.t)
    rgb, seconds = frames[0]
    assert rgb.ndim == 3 and rgb.shape[2] == 3 and seconds > 0
    out = movie.write_movie(frames, tmp_path / "movie")
    assert out.is_file() and out.stat().st_size > 0


def test_a_record_without_steps_still_gets_its_time_lapse():
    record, stack = _measure()
    record.steps = {}
    gen = movie.movie_frames_iter(record, stack)
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            frames = stop.value
            break
    assert len(frames) == len(record.t)
