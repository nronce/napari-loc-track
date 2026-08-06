from .widget import LocalizationTrackingWidget


def napari_experimental_provide_dock_widget():
    return [LocalizationTrackingWidget]
