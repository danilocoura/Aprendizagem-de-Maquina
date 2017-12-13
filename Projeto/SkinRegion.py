import numpy as np
from scipy import ndimage

class SkinRegion(object):

    def __init__(self, image, skin_mask, label_number,rectangle_slices):
        """Creates a new skin region.

            image: The entire image in YCrCb mode.
            skin_mask: The entire image skin mask.
            labeled_image: A matrix of the size of the image with the region
                label in each position. See scipy.ndimage.measurements.label.
            label_number: The label number of this skin region.
            rectangle_slices: The slices to get the rectangle of the image in
                which the region fits as returned by
                scipy.ndimage.measurements.find_objects.
        """
               
        self.label_number = label_number
        self.y0 = rectangle_slices[0].start
        self.y1 = rectangle_slices[0].stop
        self.x0 = rectangle_slices[1].start 
        self.x1 = rectangle_slices[1].stop

        self.bounding_rectangle_size = (self.x1-self.x0)*(self.y1-self.y0)
        self.bounding_rectangle_skin_pixels = np.count_nonzero(skin_mask[rectangle_slices])
        self.bounding_ratio_skin_pixels = self.bounding_rectangle_skin_pixels / self.bounding_rectangle_size
        self.image_ratio_skin_pixels = self.bounding_rectangle_skin_pixels / (image.shape[0]*image.shape[1])
        self.bounding_rectangle_average_pixel_intensity = np.average(image[rectangle_slices].take([0], axis=2))
