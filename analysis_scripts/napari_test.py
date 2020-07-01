from skimage import data
import napari
napari.gui_qt()
viewer = napari.view_image(data.astronaut(), rgb=True)