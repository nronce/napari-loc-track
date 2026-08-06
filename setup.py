from setuptools import setup, find_packages

setup(
    name="napari-loc-track",
    version="0.1.0",
    description="napari plugin for 2D single-molecule localization, tracking, and diffusion analysis",
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    package_data={"napari_loc_track": ["napari.yaml"]},
    entry_points={"napari.manifest": ["napari-loc-track = napari_loc_track:napari.yaml"]},
    install_requires=[
        "napari",
        "numpy",
        "pandas",
        "matplotlib",
        "trackpy",
        "qtpy",
        "tifffile",
        "numba",
        "scipy",
    ],
    python_requires=">=3.9",
)
