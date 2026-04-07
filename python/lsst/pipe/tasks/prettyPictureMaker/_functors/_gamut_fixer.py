# This file is part of pipe_tasks.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

__all__ = ("GamutFixer",)

import skimage
import numpy as np
import logging


from lsst.cpputils import fixGamutOK
from lsst.pipe.tasks.prettyPictureMaker.types import LABImage, RGBImage
from lsst.pex.config.configurableActions import ConfigurableAction
from lsst.pex.config import ChoiceField, Field
from lsst.rubinoxide import rgb
from scipy.ndimage import label, find_objects, binary_dilation


def heal_gamut(
    lab_image: LABImage,
    mask: np.ndarray[tuple[int, int], np.bool],
    xyz_whitepoint: tuple[float, float],
    max_size: int = 500,
    dilation_iterations: int = 3,
) -> RGBImage:
    """Heal out-of-gamut regions in a Lab image using diffusion-based inpainting.

    Parameters
    ----------
    lab_image : `LABImage`
        A NxMx3 array in the Lab colorspace. Modified in-place by healing out-of-gamut regions.
        For each region, a copy is made before modification, then written back.
    mask : `numpy.ndarray` of `bool`
        A boolean mask indicating which pixels are out of gamut.
    xyz_whitepoint : `tuple` of `float`, `float`
        Sets the white point of the xyz colorspace in xy coordinates.
    max_size : `int`, optional
        Maximum size of regions to heal. Larger regions are skipped. Default is 500.
    dilation_iterations : `int`, optional
        Number of iterations for mask dilation. Default is 3.

    Returns
    -------
    result : `RGBImage`
        The healed image converted to RGB colorspace.

    Raises
    ------
    ValueError
        Raised if the shapes of lab_image and mask are incompatible.

    Notes
    -----
    The healing algorithm works by:

    1. Labeling connected regions in the mask
    2. For each region smaller than max_size:

       - Dilating the mask to create an annulus around the region
       - Computing average a,b color values from the annulus
       - Using rgb.inpaint_mask to interpolate the L, a, and b channels
       - Filling the masked region with the interpolated values

    3. Regions larger than max_size are skipped.
    """
    # Need to split all the regions of the mask
    labels = label(binary_dilation(mask, iterations=3))[0]
    places = find_objects(labels)
    # then grow the slices by 20% of the max size
    new_places = []
    for place in places:
        size = int(3 * np.min([sl.stop - sl.start for sl in place]))
        new_y = slice(np.max((0, place[0].start - size)), np.min((mask.shape[0], place[0].stop + size)), None)
        new_x = slice(np.max((0, place[1].start - size)), np.min((mask.shape[1], place[1].stop + size)), None)
        new_places.append((new_y, new_x))
    # for each slice, dilate the mask by n-pixels, and then diff the mask to make anulus
    # get the color data in the anulus
    # set the values in the mask for the a,b channels to the color data
    # use the rgb diffusion to fix the lum channel
    # assign the fixed image back into the
    for (place_y, place_x), (old_y, old_x) in zip(new_places, places):
        sub_labels = labels[old_y, old_x]
        if np.sum(sub_labels > 0) >= max_size:
            continue
        label_number = np.max(sub_labels)
        # copy to ensure contiguous array, this is faster than operating on view
        sub_mask = labels[place_y, place_x] == label_number
        sub_lab = np.copy(lab_image[place_y, place_x])
        new_lum = rgb.inpaint_mask(
            np.ascontiguousarray(sub_lab[..., 0]),
            sub_mask,
            iterations=62,
            radius=26,
            radius_center=5,
        )
        new_a = rgb.inpaint_mask(
            np.ascontiguousarray(sub_lab[..., 1]),
            sub_mask,
            iterations=62,
            radius=26,
            radius_center=5,
        )
        new_b = rgb.inpaint_mask(
            np.ascontiguousarray(sub_lab[..., 2]),
            sub_mask,
            iterations=62,
            radius=26,
            radius_center=5,
        )
        sub_lab[..., 0] = new_lum
        sub_lab[..., 1] = new_a
        sub_lab[..., 2] = new_b

        lab_image[place_y, place_x] = sub_lab

    return rgb.Oklab_to_RGB(np.ascontiguousarray(lab_image), xyz_whitepoint)


class GamutFixer(ConfigurableAction):
    """Fix out-of-gamut colors in images by remapping them back into the RGB gamut.

    This class provides multiple methods for handling colors that fall outside
    the standard RGB color gamut, including inpainting, mapping functions,
    and diffusion-based healing.
    """

    gamutMethod = ChoiceField[str](
        doc="If doRemapGamut is True this determines the method",
        default="inpaint",
        allowed={
            "mapping": "Use a mapping function",
            "inpaint": "Use surrounding pixels to determine likely value",
            "heal": "Heal regions with reverse diffusion",
            "none": "Don't fix out of gamut colors",
        },
    )
    max_size = Field[int](doc="Maximum number of contiguous pixels that will be fixed", default=10000)

    def __call__(self, Lab: LABImage, xyz_whitepoint: tuple[float, float]) -> RGBImage:
        """Remap colors that fall outside an RGB color gamut back into it.

        When gamutMethod is 'heal', regions larger than the configured max_size
        are skipped.

        Parameters
        ----------
        Lab : `LABImage`
            A NxMx3 array in the Lab colorspace containing the image data.
        xyz_whitepoint : `tuple` of `float`, `float`
            Sets the white point of the xyz colorspace in xy coordinates.

        Returns
        -------
        result : `RGBImage`
            The image with out-of-gamut colors remapped to RGB colorspace.

        Raises
        ------
        ValueError
            Raise if the gamut method is not one of the supported methods.
        """
        rgb_prime = rgb.Oklab_to_RGB(np.ascontiguousarray(Lab), xyz_whitepoint)

        if self.gamutMethod == "none":
            return rgb_prime

        # Determine if there are any out of bounds pixels
        outOfBounds = np.bitwise_or(
            np.bitwise_or(rgb_prime[:, :, 0] > 1, rgb_prime[:, :, 0] < 0),
            np.bitwise_or(rgb_prime[:, :, 1] > 1, rgb_prime[:, :, 1] < 0),
        )
        outOfBounds = np.bitwise_or(
            outOfBounds, np.bitwise_or(rgb_prime[:, :, 2] > 1, rgb_prime[:, :, 2] < 0)
        )

        # If all pixels are in bounds, return immediately.
        if not np.any(outOfBounds):
            logging.info("There are no out of gamut pixels.")
            return rgb_prime

        logging.info("There are out of gamut pixels, remapping colors")
        match self.gamutMethod:
            case "inpaint":
                logging.debug("Running inpaint")
                labels, num_features = label(outOfBounds)
                label_counts = np.bincount(labels.ravel())[0:]
                for index, amount in zip(range(num_features), label_counts):
                    if amount > self.max_size:
                        logging.debug(f"Eliminating {amount} pixels")
                        # ignore areas that are too large
                        outOfBounds[labels == index] = 0

                results = skimage.restoration.inpaint_biharmonic(rgb_prime, outOfBounds, channel_axis=-1)
            case "mapping":
                results = fixGamutOK(Lab[outOfBounds])
                Lab[outOfBounds] = results
                results = rgb.Oklab_to_RGB(np.ascontiguousarray(Lab), xyz_whitepoint)
            case "heal":
                results = heal_gamut(Lab, outOfBounds, xyz_whitepoint)
            case _:
                raise ValueError(f"gamut correction {self.gamutMethod} is not supported")

        logging.debug(f"The total number of remapped pixels is: {np.sum(outOfBounds)}")
        return results
