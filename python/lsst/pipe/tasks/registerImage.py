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

"""
This module contains a Task to register (align) multiple images.
"""
__all__ = ["RegisterTask", "RegisterConfig"]

import math
import numpy

from lsst.pex.config import Config, Field, ConfigField
from lsst.pipe.base import Task, Struct
from lsst.meas.astrom.sip import makeCreateWcsWithSip
from lsst.afw.math import Warper

import lsst.geom as geom
import lsst.afw.table as afwTable


class RegisterConfig(Config):
    """Configuration for RegisterTask."""

    matchRadius = Field(dtype=float, default=1.0, doc="Matching radius (arcsec)", check=lambda x: x > 0)
    sipOrder = Field(dtype=int, default=4, doc="Order for SIP WCS", check=lambda x: x > 1)
    sipIter = Field(dtype=int, default=3, doc="Rejection iterations for SIP WCS", check=lambda x: x > 0)
    sipRej = Field(dtype=float, default=3.0, doc="Rejection threshold for SIP WCS", check=lambda x: x > 0)
    warper = ConfigField(dtype=Warper.ConfigClass, doc="Configuration for warping")


class RegisterTask(Task):
    """Task to register (align) multiple images.

    The 'run' method provides a revised Wcs from matches and fitting sources.
    Additional methods are provided as a convenience to warp an exposure
    ('warpExposure') and sources ('warpSources') with the new Wcs.
    """

    ConfigClass = RegisterConfig

    def run(self, inputSources, inputWcs, inputBBox, templateSources):
        """Register (align) an input exposure to the template
        The sources must have RA,Dec set, and accurate to within the
        'matchRadius' of the configuration in order to facilitate source
        matching.  We fit a new Wcs, but do NOT set it in the input exposure.

        Parameters
        ----------
        inputSources : `lsst.afw.table.SourceCatalog`
            Sources from input exposure.
        inputWcs : `lsst.afw.geom.SkyWcs`
            Wcs of input exposure.
        inputBBox : `lsst.geom.Box`
            Bounding box of input exposure.
        templateSources : `lsst.afw.table.SourceCatalog`
            Sources from template exposure.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``matches``
                Matches between sources (`list`).
            ``wcs``
                Wcs for input in frame of template (`lsst.afw.geom.SkyWcs`).
        """
        matches = self.matchSources(inputSources, templateSources)
        wcs = self.fitWcs(matches, inputWcs, inputBBox)
        return Struct(matches=matches, wcs=wcs)

    def matchSources(self, inputSources, templateSources):
        """Match sources between the input and template.

        The order of the input arguments matters (because the later Wcs
        fitting assumes a particular order).

        Parameters
        ----------
        inputSources : `lsst.afw.table.SourceCatalog`
            Source catalog of the input frame.
        templateSources : `lsst.afw.table.SourceCatalog`
            Source of the target frame.

        Returns
        -------
        matches: `list`
            Match list.
        """
        matches = afwTable.matchRaDec(templateSources, inputSources,
                                      self.config.matchRadius*geom.arcseconds)
        self.log.info("Matching within %.1f arcsec: %d matches", self.config.matchRadius, len(matches))
        self.metadata["MATCH_NUM"] = len(matches)
        if len(matches) == 0:
            raise RuntimeError("Unable to match source catalogs")
        return matches

    def fitWcs(self, matches, inputWcs, inputBBox):
        """Fit Wcs to matches.

        The fitting includes iterative sigma-clipping.

        Parameters
        ----------
        matches : `list`
            List of matches (first is target, second is input).
        inputWcs : `lsst.afw.geom.SkyWcs`
            Original input Wcs.
        inputBBox : `lsst.geom.Box`
            Bounding box of input exposure.

        Returns
        -------
        wcs: `lsst.afw.geom.SkyWcs`
            Wcs fitted to matches.
        """
        copyMatches = type(matches)(matches)
        refCoordKey = copyMatches[0].first.getTable().getCoordKey()
        inCentroidKey = copyMatches[0].second.getTable().getCentroidSlot().getMeasKey()
        for i in range(self.config.sipIter):
            sipFit = makeCreateWcsWithSip(copyMatches, inputWcs, self.config.sipOrder, inputBBox)
            self.log.debug("Registration WCS RMS iteration %d: %f pixels",
                           i, sipFit.getScatterInPixels())
            wcs = sipFit.getNewWcs()
            dr = [m.first.get(refCoordKey).separation(
                wcs.pixelToSky(m.second.get(inCentroidKey))).asArcseconds() for
                m in copyMatches]
            dr = numpy.array(dr)
            rms = math.sqrt((dr*dr).mean())  # RMS from zero
            rms = max(rms, 1.0e-9)  # Don't believe any RMS smaller than this
            self.log.debug("Registration iteration %d: rms=%f", i, rms)
            good = numpy.where(dr < self.config.sipRej*rms)[0]
            numBad = len(copyMatches) - len(good)
            self.log.debug("Registration iteration %d: rejected %d", i, numBad)
            if numBad == 0:
                break
            copyMatches = type(matches)(copyMatches[i] for i in good)

        sipFit = makeCreateWcsWithSip(copyMatches, inputWcs, self.config.sipOrder, inputBBox)
        self.log.info("Registration WCS: final WCS RMS=%f pixels from %d matches",
                      sipFit.getScatterInPixels(), len(copyMatches))
        self.metadata["SIP_RMS"] = sipFit.getScatterInPixels()
        self.metadata["SIP_GOOD"] = len(copyMatches)
        self.metadata["SIP_REJECTED"] = len(matches) - len(copyMatches)
        wcs = sipFit.getNewWcs()
        return wcs

    def warpExposure(self, inputExp, newWcs, templateWcs, templateBBox):
        """Warp input exposure to template frame.

        There are a variety of data attached to the exposure (e.g., PSF, PhotoCalib
        and other metadata), but we do not attempt to warp these to the template
        frame.

        Parameters
        ----------
        inputExp : `lsst.afw.image.Exposure`
            Input exposure, to be warped.
        newWcs : `lsst.afw.geom.SkyWcs`
            Revised Wcs for input exposure.
        templateWcs : `lsst.afw.geom.SkyWcs`
            Target Wcs.
        templateBBox : `lsst.geom.Box`
            Target bounding box.

        Returns
        -------
        alignedExp : `lsst.afw.image.Exposure`
            Warped exposure.
        """
        warper = Warper.fromConfig(self.config.warper)
        copyExp = inputExp.Factory(inputExp.getMaskedImage(), newWcs)
        alignedExp = warper.warpExposure(templateWcs, copyExp, destBBox=templateBBox)
        return alignedExp

    def warpSources(self, inputSources, newWcs, templateWcs, templateBBox):
        """Warp sources to the new frame.

        It would be difficult to transform all possible quantities of potential
        interest between the two frames.  We therefore update only the sky and
        pixel coordinates.

        Parameters
        ----------
        inputSources : `lsst.afw.table.SourceCatalog`
            Sources on input exposure, to be warped.
        newWcs : `lsst.afw.geom.SkyWcs`
            Revised Wcs for input exposure.
        templateWcs : `lsst.afw.geom.SkyWcs`
            Target Wcs.
        templateBBox : `lsst.geom.Box`
            Target bounding box.

        Returns
        -------
        alignedSources : `lsst.afw.table.SourceCatalog`
            Warped sources.
        """
        alignedSources = inputSources.copy(True)
        if not isinstance(templateBBox, geom.Box2D):
            # There is no method Box2I::contains(Point2D)
            templateBBox = geom.Box2D(templateBBox)
        table = alignedSources.getTable()
        coordKey = table.getCoordKey()
        centroidKey = table.getCentroidSlot().getMeasKey()
        deleteList = []
        for i, s in enumerate(alignedSources):
            oldCentroid = s.get(centroidKey)
            newCoord = newWcs.pixelToSky(oldCentroid)
            newCentroid = templateWcs.skyToPixel(newCoord)
            if not templateBBox.contains(newCentroid):
                deleteList.append(i)
                continue
            s.set(coordKey, newCoord)
            s.set(centroidKey, newCentroid)

        for i in reversed(deleteList):  # Delete from back so we don't change indices
            del alignedSources[i]

        return alignedSources
