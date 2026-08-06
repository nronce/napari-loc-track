import napari

from .widget import LocalizationTrackingWidget


def make_widget(viewer: napari.Viewer):
    return LocalizationTrackingWidget(viewer)
