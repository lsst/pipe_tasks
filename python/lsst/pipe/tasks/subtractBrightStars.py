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
import astropy.units as u

from lsst import geom
# from lsst.afw import math as afwMath
from lsst.afw import image as afwImage
from lsst.afw import cameraGeom as cg
from lsst.afw.geom import transformFactory as tFactory
import lsst.pex.config as pexConfig
from lsst.pipe import base as pipeBase
from lsst.meas.algorithms.loadIndexedReferenceObjects import LoadIndexedReferenceObjectsTask


class SubtractBrightStarsConfig(pexConfig.Config):
    """Configuration parameters for SubtractBrightStarsTask
    """
    magLimit = pexConfig.Field(
        dtype=float,
        doc="Magnitude limit, in Gaia G; all stars brighter than this value will be processed",
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
        },
    )
    refCatLoader = pexConfig.ConfigurableField(
        target=LoadIndexedReferenceObjectsTask,
        doc="reference object loader for astrometric calibration",
    )
    # TODO: Store this in brightStarStamps
    modelCenter = pexConfig.Field(
        dtype=int,
        doc="to be removed",
        default=(275, 275)
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

    def recoverBrightStarInfo(self, inputExposure, refCatLoader=None):
        """ TODO? fix docstring if this ends up still being needed

        Read position of bright stars within `inputExposure` from refCat
        amd extract them.

        Parameters
        ----------
        inputExposure : `afwImage.exposure.exposure.ExposureF`
            The image from which bright star stamps should be extracted.
        refCatLoader : `LoadIndexedReferenceObjectsTask`, optional
            Loader to find objects within a reference catalog.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Result struct with components:

            - ``starIms``: `list` of stamps
            - ``pixCenters``: `list` of corresponding coordinates to each
                star's center, in pixels.
            - ``GMags``: `list` of corresponding (Gaia) G magnitudes.
            - ``gaiaIds``: `np.ndarray` of corresponding unique Gaia
                identifiers.
        """
        if refCatLoader is None:
            refCatLoader = self.refCatLoader
        pixCenters = []
        wcs = inputExposure.getWcs()
        # select stars within input exposure from refcat
        withinCalexp = refCatLoader.loadPixelBox(inputExposure.getBBox(), wcs, filterName="phot_g_mean")
        refCat = withinCalexp.refCat
        # keep bright objects
        fluxLimit = ((self.config.magLimit*u.ABmag).to(u.nJy)).to_value()
        GFluxes = np.array(refCat['phot_g_mean_flux'])
        bright = GFluxes > fluxLimit
        ids = refCat.columns.extract("id", where=bright)["id"]
        selectedColumns = refCat.columns.extract('coord_ra', 'coord_dec', where=bright)
        for ra, dec in zip(selectedColumns["coord_ra"], selectedColumns["coord_dec"]):
            sp = geom.SpherePoint(ra, dec, geom.radians)
            cpix = wcs.skyToPixel(sp)
            # TODO: DM-25894 keep objects on or slightly beyond CCD edge
            if (cpix[0] >= self.config.stampSize[0]/2
                    and cpix[0] < inputExposure.getDimensions()[0] - self.config.stampSize[0]/2
                    and cpix[1] >= self.config.stampSize[1]/2
                    and cpix[1] < inputExposure.getDimensions()[1] - self.config.stampSize[1]/2):
                pixCenters += [cpix]
        return pipeBase.Struct(pixCenters=pixCenters,
                               gaiaIds=ids)

    def getTransform(self, stamps, pixCenters, model):
        """ TODO: hopefully this goes away too and we just read the
        transform from the bss
        """
        # warping control; only contains shiftingALg provided in config
        # warpCont = afwMath.WarpingControl(self.config.warpingKernelName)
        # Compare model to star stamp sizes
        bufferPix = (self.modelStampSize[0] - self.config.stampSize[0],
                     self.modelStampSize[1] - self.config.stampSize[1])
        # Initialize detector instance (note all stars were extracted from an
        # exposure from the same detector)
        det = stamps[0].getDetector()
        # Define correction for optical distortions
        pixToTan = det.getTransform(cg.PIXELS, cg.TAN_PIXELS)
        # Array of all possible rotations for detector orientation:
        possibleRots = np.array([k*np.pi/2 for k in range(4)])
        # determine how many, if any, rotations are required
        yaw = det.getOrientation().getYaw()
        nb90Rots = np.argmin(np.abs(possibleRots - float(yaw)))

        # apply transformation to each star
        # warpedModels = []
        for star, cent in zip(stamps, pixCenters):
            # (re)create empty destination image
            destImage = afwImage.MaskedImageF(*self.modelStampSize)
            # TODO: get this from bss
            bottomLeft = geom.Point2D(star.getImage().getXY0())
            newBottomLeft = pixToTan.applyForward(bottomLeft)
            newBottomLeft.setX(newBottomLeft.getX() - bufferPix[0]/2)
            newBottomLeft.setY(newBottomLeft.getY() - bufferPix[1]/2)
            # Convert to int
            newBottomLeft = geom.Point2I(newBottomLeft)
            # Set origin
            destImage.setXY0(newBottomLeft)

            # Define linear shifting to recenter stamps
            newCenter = pixToTan.applyForward(cent)  # center of warped star
            shift = self.modelCenter[0] + newBottomLeft[0] - newCenter[0],\
                self.modelCenter[1] + newBottomLeft[1] - newCenter[1]
            affineShift = geom.AffineTransform(shift)
            shiftTransform = tFactory.makeTransform(affineShift)

            # Define full transform (warp and shift)
            starWarper = pixToTan.then(shiftTransform)
            modelWarper = starWarper.inverted()

            # TODO: get this from bss
            return modelWarper, nb90Rots

    @pipeBase.timeMethod
    def run(self, inputExposure, refObjLoader=None, dataId=None):
        """TODO: write docstring

        Parameters
        ----------
        inputExposure : `afwImage.exposure.exposure.ExposureF`
            The image from which bright stars should be subtracted.
        refCatLoader : `LoadIndexedReferenceObjectsTask`, optional
            Loader to find objects within a reference catalog.

        Returns
        -------
        result :  `lsst.pipe.base.Struct`
            TBW
        """
        self.log.info("Subtracting bright stars from exposure %s" % (dataId))
        # Recover info. Essentially what ProcessBrightStarsTask.extractStamps
        # TODO: fix this.
        # brightStarInfo = self.recoverBrightStarInfo(inputExposure,
        # refCatLoader=refObjLoader)
        # Warp (and shift, and potentially rotate) model to fit stars
        # warpedStars = self.warpStamps(extractedStamps.starIms,
        # extractedStamps.pixCenters)
        # Multiply by annular flux to revert normalization

        # Subtract model
        return

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
        output = self.run(calexp, dataId=dataRef.dataId)
        # Save processed bright star stamps
        dataRef.put(output.brightStarStamps, "brightStarStamps")
        return pipeBase.Struct(brightStarStamps=output.brightStarStamps)
