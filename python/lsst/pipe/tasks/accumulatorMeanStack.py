# This file is part of pipe_tasks.
#
# LSST Data Management System
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
# See COPYRIGHT file at the top of the source tree.
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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#
import numpy as np


__all__ = ['AccumulatorMeanStack', 'stats_ctrl_to_threshold_dict']


def stats_ctrl_to_threshold_dict(stats_ctrl):
    """Convert stats control to threshold dict"""
    threshold_dict = {}
    for bit in range(64):
        threshold_dict[bit] = stats_ctrl.getMaskPropagationThreshold(bit)

    return threshold_dict


class AccumulatorMeanStack(object):
    """Stack masked images.

    Parameters
    ----------
    shape : `tuple`
        Shape of the input and output images.
    bit_mask_value : `int`
        Bit mask to flag for "bad" inputs that should not be stacked.
    mask_threshold_dict : `dict` [`int`: `float`], optional
        Dictionary of mapping from bit number to threshold for flagging.
        Only bad bits (in bit_mask_value) which mask fractional weight
        greater than this threshold will be flagged in the output image.
    mask_map : `list` [`tuple`], optional
        Mapping from input image bits to aggregated coadd bits.
    no_good_pixels_mask : `int`, optional
        Bit mask to set when there are no good pixels in the stack.
        If not set then will set coadd masked image 'NO_DATA' bit.
    calc_error_from_input_variance : `bool`, optional
        Calculate the error from the input variance?
    compute_n_image : `bool`, optional
        Calculate the n_image map as well as stack?
    do_sigma_clip : `bool`, optional
        Perform sigma-clipped statistics
    sigma_clip : `float`, optional
        Sigma for outlier rejection.  Only used if do_clip is True.
    clipped_bit : `int`, optional
        Bit to set for clipped pixels.
    """
    def __init__(self, shape,
                 bit_mask_value, mask_threshold_dict={},
                 mask_map=[], no_good_pixels_mask=None,
                 calc_error_from_input_variance=True,
                 compute_n_image=False, do_sigma_clip=False,
                 sigma_clip=3.0, clipped_bit=None):
        self.shape = shape
        self.bit_mask_value = bit_mask_value
        self.mask_map = mask_map
        self.no_good_pixels_mask = no_good_pixels_mask
        self.calc_error_from_input_variance = calc_error_from_input_variance
        self.compute_n_image = compute_n_image
        self.do_sigma_clip = do_sigma_clip
        self.sigma_clip = sigma_clip
        self.clipped_bit = clipped_bit

        # Only track threshold bits that are in the bad bit_mask_value.
        self.mask_threshold_dict = {}
        for bit in mask_threshold_dict:
            if (self.bit_mask_value & 2**bit) > 0:
                self.mask_threshold_dict[bit] = mask_threshold_dict[bit]

        # sum(weight)
        self.sum_weight = np.zeros(shape, dtype=np.float64)
        # sum(weight*data)
        self.sum_wdata = np.zeros(shape, dtype=np.float64)

        if calc_error_from_input_variance:
            # sum(weight**2 * variance)
            self.sum_w2var = np.zeros(shape, dtype=np.float64)
        else:
            # sum(weight**2)
            self.sum_weight2 = np.zeros(shape, dtype=np.float64)
            # sum(weight**2 * data)
            self.sum_w2data = np.zeros(shape, dtype=np.float64)

        self.or_mask = np.zeros(shape, dtype=np.int64)
        self.rejected_weights_by_bit = {}
        for bit in self.mask_threshold_dict:
            self.rejected_weights_by_bit[bit] = np.zeros(shape, dtype=np.float64)

        if self.compute_n_image:
            self.n_image = np.zeros(shape, dtype=np.int32)

        if self.do_sigma_clip:
            self._sigma_clip_initial_iteration = True
            self._clip_mean_image = np.zeros(shape, dtype=np.float64)
            self._clip_sigma_image = np.zeros(shape, dtype=np.float64)
        else:
            self._sigma_clip_initial_iteration = False

    def add_masked_image(self, masked_image, weight=1.0):
        """Add a masked image to the stack.

        Parameters
        ----------
        masked_image : `lsst.afw.image.MaskedImage`
            Masked image to add to the stack.
        """
        if self.do_sigma_clip and not self._sigma_clip_initial_iteration:
            clip_delta = np.abs(masked_image.image.array - self._clip_mean_image)
            good_pixels = np.where(((masked_image.mask.array & self.bit_mask_value) == 0)
                                   & np.isfinite(masked_image.mask.array)
                                   & (clip_delta < self.sigma_clip*self._clip_sigma_image))
            clipped_pixels = np.where(clip_delta >= self.sigma_clip*self._clip_sigma_image)
            if self.clipped_bit is not None:
                self.or_mask[clipped_pixels] |= self.clipped_bit
        else:
            good_pixels = np.where(((masked_image.mask.array & self.bit_mask_value) == 0)
                                   & np.isfinite(masked_image.mask.array))

        self.sum_weight[good_pixels] += weight
        self.sum_wdata[good_pixels] += weight*masked_image.image.array[good_pixels]

        if self.compute_n_image:
            self.n_image[good_pixels] += 1

        if self.calc_error_from_input_variance:
            self.sum_w2var[good_pixels] += (weight**2.)*masked_image.variance.array[good_pixels]
        else:
            self.sum_weight2[good_pixels] += weight**2.
            self.sum_w2data[good_pixels] += (weight**2.)*masked_image.image.array[good_pixels]

        # Mask bits are propagated for good pixels
        self.or_mask[good_pixels] |= masked_image.mask.array[good_pixels]

        # Bad pixels are only tracked if they cross a threshold
        for bit in self.mask_threshold_dict:
            bad_pixels = np.where((masked_image.mask.array & 2**bit) > 0)
            self.rejected_weights_by_bit[bit][bad_pixels] += weight

    def fill_stacked_masked_image(self, stacked_masked_image):
        """Fill the stacked mask image after accumulation.

        Parameters
        ----------
        stacked_masked_image : `lsst.afw.image.MaskedImage`
            Total masked image.
        """
        with np.warnings.catch_warnings():
            # Let the NaNs through and flag bad pixels below
            np.warnings.simplefilter("ignore")

            # The image plane is sum(weight*data)/sum(weight)
            stacked_masked_image.image.array[:, :] = self.sum_wdata/self.sum_weight

            if self.calc_error_from_input_variance:
                mean_var = self.sum_w2var/(self.sum_weight**2.)
            else:
                # Compute the biased estimator
                variance = self.sum_w2data/self.sum_weight - stacked_masked_image.image.array[:, :]**2.
                # De-bias
                variance *= (self.sum_weight**2.)/(self.sum_weight**2. - self.sum_weight2)

                # Compute the mean variance
                mean_var = variance*self.sum_weight2/(self.sum_weight**2.)

            stacked_masked_image.variance.array[:, :] = mean_var

            # Propagate bits when they cross the threshold
            for bit in self.mask_threshold_dict:
                hypothetical_total_weight = self.sum_weight + self.rejected_weights_by_bit[bit]
                self.rejected_weights_by_bit[bit] /= hypothetical_total_weight
                propagate = np.where(self.rejected_weights_by_bit[bit] > self.mask_threshold_dict[bit])
                self.or_mask[propagate] |= 2**bit

            # Map the mask planes to new bits
            for mask_tuple in self.mask_map:
                self.or_mask[(self.or_mask & mask_tuple[0]) > 0] |= mask_tuple[1]

            stacked_masked_image.mask.array[:, :] = self.or_mask

            if self.do_sigma_clip:
                self._sigma_clip_initial_iteration = False
                self._clip_mean_image[:, :] = stacked_masked_image.image.array[:, :]
                self._clip_sigma_image[:, :] = np.sqrt(stacked_masked_image.variance.array[:, :])

        if self.no_good_pixels_mask is None:
            mask_dict = stacked_masked_image.maskedImage().getMask().getMaskPlaneDict()
            no_good_pixels_mask = 2**(mask_dict['NO_DATA'])
        else:
            no_good_pixels_mask = self.no_good_pixels_mask

        bad_pixels = (self.sum_weight <= 0.0)
        stacked_masked_image.mask.array[bad_pixels] |= no_good_pixels_mask
