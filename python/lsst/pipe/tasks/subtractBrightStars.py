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
#
"""Retrieve extended PSF model and subtract bright stars at calexp (ie
single visit) level.
"""

__all__ = ["SubtractBrightStarsTask"]

import numpy as np

from lsst import geom
from lsst.afw import math as afwMath
from lsst.afw import image as afwImage
import lsst.pex.config as pexConfig
from lsst.pipe import base as pipeBase
from lsst.meas.algorithms.loadIndexedReferenceObjects import LoadIndexedReferenceObjectsTask
from lsst.pex import exceptions as pexExceptions
from lsst.daf.persistence import butlerExceptions as bE


class SubtractBrightStarsConfig(pexConfig.Config):
    """Configuration parameters for SubtractBrightStarsTask
    """
    modelFilename = pexConfig.Field(
        dtype=str,
        doc="Absolute path to saved extended PSF model (temporary)",
    )
    doWriteSubtractor = pexConfig.Field(
        dtype=bool,
        doc="Should an exposure containing all bright star models be written to disk?",
        default=True
    )
    subtractorFilename = pexConfig.Field(
        dtype=str,
        doc="(TEMP) Absolute path to where the subtractor file should be written."
            "An ersatz of dataId template will be added.",
        default="subtractor"
    )
    doWriteSubtractedExposure = pexConfig.Field(
        dtype=bool,
        doc="Should an exposure with bright stars subtracted be written to disk?",
        default=True
    )
    subtractedExposureFilename = pexConfig.Field(
        dtype=str,
        doc="(TEMP) Absolute path to where the subtractor file should be written."
            "An ersatz of dataId template will be added.",
        default="subtractedExposure"
    )
    magLimit = pexConfig.Field(
        dtype=float,
        doc="Magnitude limit, in Gaia G; all stars brighter than this value will be subtracted",
        default=18
    )
    warpingKernelName = pexConfig.ChoiceField(
        dtype=str,
        doc="Warping kernel",
        default="lanczos5",
        allowed={
            "bilinear": "bilinear interpolation",
            "lanczos3": "Lanczos kernel of order 3",
            "lanczos4": "Lanczos kernel of order 4",
            "lanczos5": "Lanczos kernel of order 5",
        }
    )
    refCatLoader = pexConfig.ConfigurableField(
        target=LoadIndexedReferenceObjectsTask,
        doc="reference object loader for astrometric calibration",
    )

    def setDefaults(self):
        self.refCatLoader.ref_dataset_name = "gaia_dr2_20200414"


class SubtractBrightStarsTask(pipeBase.CmdLineTask):
    """TODO: write docstring
    """
    ConfigClass = SubtractBrightStarsConfig
    _DefaultName = "subtractBrightStars"

    def __init__(self, butler=None, initInputs=None, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        # configure Gaia refcat
        if butler is not None:
            self.makeSubtask('refCatLoader', butler=butler)

    def matchModel(self, inputModel, bss, subtractor):
        # make a copy of the input model
        model = inputModel.clone()
        modelStampSize = model.getDimensions()
        inv90Rots = 4 - bss.nb90Rots
        model = afwMath.rotateImageBy90(model, inv90Rots)
        warpCont = afwMath.WarpingControl(self.config.warpingKernelName)
        invImages = []
        for star in bss:
            if star.gaiaGMag < self.config.magLimit:
                # set origin
                model.setXY0(star.XY0)
                # create empty destination image
                invTransform = star.transform.inverted()
                invXY0 = geom.Point2I(invTransform.applyForward(geom.Point2D(star.XY0)))
                bbox = geom.Box2I(corner=invXY0, dimensions=modelStampSize)
                invImage = afwImage.MaskedImageF(bbox)
                # Apply inverse transform
                goodPix = afwMath.warpImage(invImage, model, invTransform, warpCont)
                if not goodPix:
                    self.log.debug(f"Warping of a model failed for star {star.gaiaId}:"
                                   "no good pixel in output")
                # Multiply by annularFlux
                invImage.image *= star.annularFlux
                # Replace nans before subtraction (note all nan pixels have
                # the NO_DATA flag)
                invImage.image.array[np.isnan(invImage.image.array)] = 0
                # Add matched model to subtractor exposure
                try:
                    subtractor[bbox] += invImage
                    invImages.append(invImage)
                except pexExceptions.LengthError:
                    # TODO: handle stars close to the edge (DM-25894)
                    self.log.debug(f"Star {star.gaiaId} not included in the subtractor"
                                   "because it was too close to the edge.")
        return subtractor, invImages

    @pipeBase.timeMethod
    def run(self, inputExposure, bss, refObjLoader=None, dataId=None):
        """TODO: write docstring

        Parameters
        ----------
        inputExposure : `afwImage.exposure.exposure.ExposureF`
            The image from which bright stars should be subtracted.
        bss : `meas.algorithms.brightStarStamps.BrightStarStamps`
            Set of bright star stamps extracted from this exposure.
        refCatLoader : `LoadIndexedReferenceObjectsTask`, optional
            Loader to find objects within a reference catalog.

        Returns
        -------
        result : TBW
        """
        self.log.info("Subtracting bright stars from exposure %s" % (dataId))
        # Read model
        model = afwImage.MaskedImageF(self.config.modelFilename)
        # Create an empty image the size of the exposure
        brightStarSubtractor = afwImage.MaskedImageF(bbox=inputExposure.getBBox())
        # Warp (and shift, and potentially rotate) model to fit each star
        subtractor, invImages = self.matchModel(model, bss, brightStarSubtractor)
        if self.config.doWriteSubtractor:
            if dataId is not None:
                subtractor.writeFits(self.config.subtractorFilename
                                     + f"-{dataId['visit']}-{dataId['ccd']}.fits")
        if self.config.doWriteSubtractedExposure:
            subtractedExposure = inputExposure.clone()
            subtractedExposure.image -= subtractor.image
            if dataId is not None:
                subtractedExposure.writeFits(self.config.subtractedExposureFilename
                                             + f"-{dataId['visit']}-{dataId['ccd']}.fits")
        return subtractor

    def runDataRef(self, dataRef):
        """ Read in selected calexp, corresponding bright star stamps
        and extended PSF model, perform subtraction and save resulting
        image to disk.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.butlerSubset.ButlerDataRef`
            Data reference to the calexp to subtract bright stars from.
        """
        calexp = dataRef.get("calexp")
        try:
            bss = dataRef.get("brightStarStamps")
        except bE.NoResults:
            self.log.info(f"No bright stars to subtract from exposure {dataRef.dataId}.")
            return
        output = self.run(calexp, bss, dataId=dataRef.dataId)
        return pipeBase.Struct(subtractor=output)
