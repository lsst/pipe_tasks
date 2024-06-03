from tqdm import tqdm
import numpy as np
from .gaussian_processes import (
    GaussianProcessTreegp,
    GaussianProcessHODLRSolver,
    GaussianProcessGPyTorch,
)
from lsst.meas.algorithms import CloughTocher2DInterpolatorUtils as ctUtils
from lsst.geom import Box2I, Point2I, Extent2I
from lsst.afw.geom import SpanSet
from scipy.stats import binned_statistic_2d
import copy


def timer(f):
    import functools

    @functools.wraps(f)
    def f2(*args, **kwargs):
        import time
        import inspect

        t0 = time.time()
        result = f(*args, **kwargs)
        t1 = time.time()
        fname = repr(f).split()[1]
        print("time for %s = %.4f" % (fname, t1 - t0))
        return result

    return f2


def meanify(params, coords, bin_spacing=10, stat_used="mean", x_min=None, x_max=None, y_min=None, y_max=None):
    """
    Compute the mean of the given parameters over a grid of coordinates.

    Parameters:
    - params (numpy.ndarray): Array of parameters to be averaged.
    - coords (numpy.ndarray): Array of coordinates corresponding to the parameters.
    - bin_spacing (float, optional): Spacing between bins in the units of the coordinates. Default is 10.
    - stat_used (str, optional): Statistic to be computed. Default is 'mean'.
    - x_min (float, optional): Minimum x-coordinate value. If not provided, the minimum value from the coordinates array is used.
    - x_max (float, optional): Maximum x-coordinate value. If not provided, the maximum value from the coordinates array is used.
    - y_min (float, optional): Minimum y-coordinate value. If not provided, the minimum value from the coordinates array is used.
    - y_max (float, optional): Maximum y-coordinate value. If not provided, the maximum value from the coordinates array is used.

    Returns:
    - coords0 (numpy.ndarray): Array of coordinates corresponding to the averaged parameters.
    - params0 (numpy.ndarray): Array of averaged parameters.

    Note: The bin spacing is in the units of the coordinates.
    """
    if x_min is None:
        x_min = np.min(coords[:, 0])
    if x_max is None:
        x_max = np.max(coords[:, 0])
    if y_min is None:
        y_min = np.min(coords[:, 1])
    if y_max is None:
        y_max = np.max(coords[:, 1])

    nbin_x = int((x_max - x_min) / bin_spacing)
    nbin_y = int((y_max - y_min) / bin_spacing)
    binning = [np.linspace(x_min, x_max, nbin_x), np.linspace(y_min, y_max, nbin_y)]
    nbinning = (len(binning[0]) - 1) * (len(binning[1]) - 1)
    Filter = np.array([True] * nbinning)

    average, u0, v0, _ = binned_statistic_2d(
        coords[:, 0], coords[:, 1], params, bins=binning, statistic=stat_used
    )

    # get center of each bin
    u0 = u0[:-1] + (u0[1] - u0[0]) / 2.0
    v0 = v0[:-1] + (v0[1] - v0[0]) / 2.0
    u0, v0 = np.meshgrid(u0, v0)

    average = average.T
    average = average.reshape(-1)
    Filter &= np.isfinite(average).reshape(-1)

    coords0 = np.array([u0.reshape(-1), v0.reshape(-1)]).T
    coords0 = coords0[Filter]
    params0 = average[Filter]

    return coords0, params0


class InterpolateOverDefectGaussianProcess:
    """
    Class for interpolating over defects in a masked image using Gaussian Processes.

    Args:
        maskedImage (MaskedImage): The masked image containing defects.
        defects (list, optional): List of defect names to interpolate over. Defaults to ["SAT"].
        fwhm (float, optional): FWHM from PSF and used as prior for correlation length. Defaults to 5.
        block_size (int, optional): Size of the block for block interpolation method. Defaults to 100.
        solver (str, optional): Solver to use for Gaussian Process interpolation. Options are "treegp", "george", and "gpytorch". Defaults to "treegp".
        method (str, optional): Interpolation method to use. Options are "block" and "spanset". Defaults to "block".
        use_binning (bool, optional): Whether to use binning for large areas. Defaults to False.
        bin_spacing (float, optional): Spacing for binning. Defaults to 10.
    """

    def __init__(
        self,
        maskedImage,
        defects=["SAT", "INTRP"],
        fwhm=10,
        block_size=100,
        solver="treegp",
        method="block",
        use_binning=False,
        bin_spacing=10,
    ):
        """
        Initializes the InterpolateOverDefectGaussianProcess class.

        Args:
            maskedImage (MaskedImage): The masked image containing defects.
            defects (list, optional): List of defect names to interpolate over. Defaults to ["SAT"].
            fwhm (float, optional): FWHM from PSF and used as prior for correlation length. Defaults to 5.
            block_size (int, optional): Size of the block for block interpolation method. Defaults to 100.
            solver (str, optional): Solver to use for Gaussian Process interpolation. Options are "treegp", "george", and "gpytorch". Defaults to "treegp".
            method (str, optional): Interpolation method to use. Options are "block" and "spanset". Defaults to "block".
            use_binning (bool, optional): Whether to use binning for large areas. Defaults to False.
            bin_spacing (float, optional): Spacing for binning. Defaults to 10.
        """

        if solver not in ["treegp", "george", "gpytorch"]:
            raise ValueError(
                "Only treegp, george, and gpytorch are supported for solver. Current value: %s"
                % (self.optimizer)
            )
        if solver == "treegp":
            self.solver = GaussianProcessTreegp
        elif solver == "george":
            self.solver = GaussianProcessHODLRSolver
        elif solver == "gpytorch":
            self.solver = GaussianProcessGPyTorch

        if method not in ["block", "spanset"]:
            raise ValueError(
                "Only block and spanset are supported for method. Current value: %s" % (self.method)
            )

        self.method = method
        self.block_size = block_size

        self.use_binning = use_binning
        self.bin_spacing = bin_spacing

        self.maskedImage = maskedImage
        self.defects = defects
        self.correlation_length = fwhm

    def _interpolate_over_defects_spanset(self):
        """
        Interpolates over defects using the spanset method.
        """

        mask = self.maskedImage.getMask()
        badPixelMask = mask.getPlaneBitMask(self.defects)
        badMaskSpanSet = SpanSet.fromMask(mask, badPixelMask).split()

        bbox = self.maskedImage.getBBox()
        glob_xmin, glob_xmax = bbox.minX, bbox.maxX
        glob_ymin, glob_ymax = bbox.minY, bbox.maxY

        condition = False
        for i in tqdm(range(len(badMaskSpanSet))):
            spanset = badMaskSpanSet[i]
            bbox = spanset.getBBox()
            # Dilate the bbox to make sure we have enough good pixels around the defect
            # For now, we dilate by 5 times the correlation length
            # For GP with isotropic kernel, points at 5 correlation lengths away have negligible
            # effect on the prediction.
            bbox = bbox.dilatedBy(self.correlation_length * 5)
            xmin, xmax = max([glob_xmin, bbox.minX]), min(glob_xmax, bbox.maxX)
            ymin, ymax = max([glob_ymin, bbox.minY]), min(glob_ymax, bbox.maxY)
            localBox = Box2I(Point2I(xmin, ymin), Extent2I(xmax - xmin, ymax - ymin))
            problem_size = (xmax - xmin) * (ymax - ymin)
            if problem_size > 10000 and not self.use_binning:
                # TO DO: need to implement a better way to interpolate over large areas
                # TO DO: One suggested idea might be to bin the area and average and interpolate using
                # TO DO: the average values.
                print("Problem size is too large to interpolate over. Skipping.")
                print("Problem size: ", problem_size)
                print("xmin, xmax, ymin, ymax: ", xmin, xmax, ymin, ymax)
                print("bbox: ", bbox)
                print("Use interpolate_over_defects_block instead for this spanset.")
                try:
                    sub_masked_image = self.maskedImage[localBox]
                except:
                    condition = True
                    break
                try:
                    sub_masked_image = self._interpolate_over_defects_block(maskedImage=sub_masked_image)
                except:
                    continue
                self.maskedImage[localBox] = sub_masked_image
            else:
                try:
                    sub_masked_image = self.maskedImage[localBox]
                except:
                    condition = True
                    break
                try:
                    sub_masked_image = self.interpolate_sub_masked_image(sub_masked_image)
                except:
                    continue
                self.maskedImage[localBox] = sub_masked_image
        if condition:
            breakpoint()

    def _interpolate_over_defects_block(self, maskedImage=None):
        """
        Interpolates over defects using the block method.

        Args:
            maskedImage (ndarray, optional): The masked image to interpolate over. If not provided, the method will use the
                `maskedImage` attribute of the class.

        Returns:
            ndarray: The interpolated masked image.
        """
        if maskedImage is None:
            maskedImage = self.maskedImage
            bbox = None
        else:
            bbox = maskedImage.getBBox()
            ox = bbox.beginX
            oy = bbox.beginY
            maskedImage.setXY0(0, 0)

        nx = maskedImage.getDimensions()[0]
        ny = maskedImage.getDimensions()[1]

        for x in tqdm(range(0, nx, self.block_size)):
            for y in range(0, ny, self.block_size):
                sub_nx = min(self.block_size, nx - x)
                sub_ny = min(self.block_size, ny - y)
                sub_masked_image = maskedImage[x : x + sub_nx, y : y + sub_ny]
                sub_masked_image = self.interpolate_sub_masked_image(sub_masked_image)
                maskedImage[x : x + sub_nx, y : y + sub_ny] = sub_masked_image

        if bbox is not None:
            maskedImage.setXY0(ox, oy)

        return maskedImage

    @timer
    def interpolate_over_defects(self):
        """
        Interpolates over defects using the specified method.
        """

        if self.method == "block":
            self.maskedImage = self._interpolate_over_defects_block()
        elif self.method == "spanset":
            self._interpolate_over_defects_spanset()

    def _good_pixel_binning(self, good_pixel):
        """
        Performs binning of good pixel data.

        Parameters:
        - good_pixel (numpy.ndarray): An array containing the good pixel data.

        Returns:
        - numpy.ndarray: An array containing the binned data.

        """
        coord, params = meanify(
            good_pixel[:, 2:].T, good_pixel[:, :2], bin_spacing=self.bin_spacing, stat_used="mean"
        )
        return np.array([coord[:, 0], coord[:, 1], params]).T

    def interpolate_sub_masked_image(self, sub_masked_image):
        """
        Interpolates over defects in a sub-masked image.

        Args:
            sub_masked_image (MaskedImage): The sub-masked image containing defects.

        Returns:
            MaskedImage: The sub-masked image with defects interpolated.
        """

        cut = self.correlation_length * 5
        bad_pixel, good_pixel = ctUtils.findGoodPixelsAroundBadPixels(
            sub_masked_image, self.defects, buffer=cut
        )
        # Do nothing if bad pixel is None.
        if np.shape(bad_pixel)[0] == 0:
            return sub_masked_image
        # Do GP interpolation if bad pixel found.
        else:
            # gp interpolation
            mean = np.mean(good_pixel[:, 2:])
            sub_image_array = sub_masked_image.getVariance().array
            white_noise = np.sqrt(np.mean(sub_image_array[np.isfinite(sub_image_array)]))
            kernel_amplitude = np.std(good_pixel[:, 2:])
            if self.use_binning:
                good_pixel = self._good_pixel_binning(copy.deepcopy(good_pixel))

            gp = self.solver(
                std=np.sqrt(kernel_amplitude),
                correlation_length=self.correlation_length,
                white_noise=white_noise,
                mean=mean,
            )
            if bad_pixel.size > 20000:
                raise ValueError("Too many pixels")
            gp.fit(good_pixel[:, :2], np.squeeze(good_pixel[:, 2:]))
            gp_predict = gp.predict(bad_pixel[:, :2])

            bad_pixel[:, 2:] = gp_predict.reshape(np.shape(bad_pixel[:, 2:]))

            # update_value
            ctUtils.updateImageFromArray(sub_masked_image.image, bad_pixel)
            return sub_masked_image
