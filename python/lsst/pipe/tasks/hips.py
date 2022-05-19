#
# LSST Data Management System
# Copyright 2022 AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
"""Tasks for making and manipulating HIPS images."""

__all__ = ["HighResolutionHipsTask", "HighResolutionHipsConfig", "HighResolutionHipsConnections"]

from collections import defaultdict
import numpy as np
import argparse
import sys

from lsst.sphgeom import RangeSet, HealpixPixelization
from lsst.utils.timer import timeMethod
from lsst.daf.butler import Butler, DatasetRef, Quantum
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.geom as geom


class HighResolutionHipsConnections(pipeBase.PipelineTaskConnections,
                                    dimensions=("healpix9", "band"),
                                    defaultTemplates={"coaddName": "deep"}):
    coadd_exposure_handles = pipeBase.connectionTypes.Input(
        doc="Coadded exposures to convert to HIPS format.",
        name="{coaddName}Coadd_calexp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "band"),
        multiple=True,
        deferLoad=True,
    )
    hips_exposures = pipeBase.connectionTypes.Output(
        doc="HIPS-compatible HPX image.",
        name="{coaddName}Coadd_hpx",
        storageClass="ExposureF",
        dimensions=("healpix11", "band"),
        multiple=True,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        quantum_order = None
        for dim in self.dimensions:
            if 'healpix' in dim:
                quantum_order = int(dim.split('healpix')[1])
        if quantum_order is None:
            raise ValueError("Must specify a healpix dimension in quantum dimensions.")

        if quantum_order > config.hips_order:
            raise ValueError("Quantum healpix dimension order must not be greater than hips_order")

        order = None
        for dim in self.hips_exposures.dimensions:
            if 'healpix' in dim:
                order = int(dim.split('healpix')[1])
        if order is None:
            raise ValueError("Must specify a healpix dimension in hips_exposure dimensions.")

        if order != config.hips_order:
            raise ValueError("healpix dimension order must match config.hips_order.")


class HighResolutionHipsConfig(pipeBase.PipelineTaskConfig,
                               pipelineConnections=HighResolutionHipsConnections):
    """Configuration parameters for HighResolutionHipsTask.

    Notes
    -----
    A HiPS image covers one HEALPix cell, with the HEALPix nside equal to
    2**hips_order. Each cell is 'shift_order' orders deeper than the HEALPix
    cell, with 2**shift_order x 2**shift_order sub-pixels on a side, which
    defines the target resolution of the HiPS image. The IVOA recommends
    shift_order=9, for 2**9=512 pixels on a side.

    Table 5 from
    https://www.ivoa.net/documents/HiPS/20170519/REC-HIPS-1.0-20170519.pdf
    shows the relationship between hips_order, number of tiles (full
    sky coverage), cell size, and sub-pixel size/image resolution (with
    the default shift_order=9):
    +------------+-----------------+--------------+------------------+
    | hips_order | Number of Tiles | Cell Size    | Image Resolution |
    +============+=================+==============+==================+
    | 0          | 12              | 58.63 deg    | 6.871 arcmin     |
    | 1          | 48              | 29.32 deg    | 3.435 arcmin     |
    | 2          | 192             | 14.66 deg    | 1.718 arcmin     |
    | 3          | 768             | 7.329 deg    | 51.53 arcsec     |
    | 4          | 3072            | 3.665 deg    | 25.77 arcsec     |
    | 5          | 12288           | 1.832 deg    | 12.88 arcsec     |
    | 6          | 49152           | 54.97 arcmin | 6.442 arcsec     |
    | 7          | 196608          | 27.48 arcmin | 3.221 arcsec     |
    | 8          | 786432          | 13.74 arcmin | 1.61 arcsec      |
    | 9          | 3145728         | 6.871 arcmin | 805.2mas         |
    | 10         | 12582912        | 3.435 arcmin | 402.6mas         |
    | 11         | 50331648        | 1.718 arcmin | 201.3mas         |
    | 12         | 201326592       | 51.53 arcsec | 100.6mas         |
    | 13         | 805306368       | 25.77 arcsec | 50.32mas         |
    +------------+-----------------+--------------+------------------+
    """
    hips_order = pexConfig.Field(
        doc="HIPS image order.",
        dtype=int,
        default=11,
    )
    shift_order = pexConfig.Field(
        doc="HIPS shift order (such that each tile is 2**shift_order pixels on a side)",
        dtype=int,
        default=9,
    )
    warp = pexConfig.ConfigField(
        dtype=afwMath.Warper.ConfigClass,
        doc="Warper configuration",
    )

    def setDefaults(self):
        self.warp.warpingKernelName = "lanczos5"


class HighResolutionHipsTask(pipeBase.PipelineTask):
    """Task for making high resolution HIPS images."""
    ConfigClass = HighResolutionHipsConfig
    # The name should include the quantum healpix order.
    _DefaultName = "highResolutionHips9"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.warper = afwMath.Warper.fromConfig(self.config.warp)

    @timeMethod
    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        healpix_dim = f"healpix{self.config.hips_order}"

        pixels = [hips_exposure.dataId[healpix_dim]
                  for hips_exposure in outputRefs.hips_exposures]

        outputs = self.run(pixels, inputs["coadd_exposure_handles"])

        hips_exposure_ref_dict = {hips_exposure_ref.dataId[healpix_dim]:
                                  hips_exposure_ref for hips_exposure_ref in outputRefs.hips_exposures}
        for pixel, hips_exposure in outputs.hips_exposures.items():
            butlerQC.put(hips_exposure, hips_exposure_ref_dict[pixel])

    def run(self, pixels, coadd_exposure_handles):
        """Run the HighResolutionHipsTask.

        Parameters
        ----------
        pixels : `Iterable` [ `int` ]
            Iterable of healpix pixels (nest ordering) to warp to.
        coadd_exposure_handles : `list` [`lsst.daf.butler.DeferredDatasetHandle`]
            Handles for the coadd exposures.

        Returns
        -------
        outputs : `lsst.pipe.base.Struct`
            ``hips_exposures`` is a dict with pixel (key) and hips_exposure (value)
        """
        self.log.info("Generating HIPS images for %d pixels at order %d", len(pixels), self.config.hips_order)

        npix = 2**self.config.shift_order
        bbox_hpx = geom.Box2I(corner=geom.Point2I(0, 0),
                              dimensions=geom.Extent2I(npix, npix))

        exp_hpx_dict = {}
        warp_dict = {}
        for pixel in pixels:
            wcs_hpx = afwGeom.makeHpxWcs(self.config.hips_order, pixel, shift_order=self.config.shift_order)
            exp_hpx = afwImage.ExposureF(bbox_hpx, wcs_hpx)
            exp_hpx_dict[pixel] = exp_hpx
            warp_dict[pixel] = []

        first_handle = True
        for handle in coadd_exposure_handles:
            coadd_exp = handle.get()

            for pixel in pixels:
                warped = self.warper.warpExposure(exp_hpx_dict[pixel].getWcs(), coadd_exp, maxBBox=bbox_hpx)

                exp = afwImage.ExposureF(exp_hpx_dict[pixel].getBBox(), exp_hpx_dict[pixel].getWcs())
                exp.maskedImage.set(np.nan, afwImage.Mask.getPlaneBitMask("NO_DATA"), np.nan)

                if first_handle:
                    exp_hpx_dict[pixel].mask.conformMaskPlanes(coadd_exp.mask.getMaskPlaneDict())
                    exp_hpx_dict[pixel].setFilterLabel(coadd_exp.getFilterLabel())
                    exp_hpx_dict[pixel].setPhotoCalib(coadd_exp.getPhotoCalib())

                if warped.getBBox().getArea() == 0 or not np.any(np.isfinite(warped.getImage().array)):
                    # There is no overlap, skip.
                    self.log.debug(
                        "No overlap between output HPX %d and input exposure %s",
                        pixel,
                        handle.dataId
                    )
                    continue

                exp.maskedImage.assign(warped.maskedImage, warped.getBBox())
                warp_dict[pixel].append(exp.maskedImage)

            first_handle = False

        stats_flags = afwMath.stringToStatisticsProperty('MEAN')
        stats_ctrl = afwMath.StatisticsControl()
        stats_ctrl.setNanSafe(True)
        stats_ctrl.setWeighted(True)
        stats_ctrl.setCalcErrorFromInputVariance(True)

        for pixel in pixels:
            exp_hpx_dict[pixel].maskedImage.set(np.nan, afwImage.Mask.getPlaneBitMask("NO_DATA"), np.nan)

            if not warp_dict[pixel]:
                # Nothing in this pixel
                self.log.debug("No data in HPX pixel %d", pixel)
                # Remove the pixel from the output
                exp_hpx_dict.pop(pixel)
                continue

            exp_hpx_dict[pixel].maskedImage = afwMath.statisticsStack(
                warp_dict[pixel],
                stats_flags,
                stats_ctrl,
                [1.0]*len(warp_dict[pixel]),
                clipped=0,
                maskMap=[]
            )

        return pipeBase.Struct(hips_exposures=exp_hpx_dict)
