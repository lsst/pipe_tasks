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

__all__ = ["MatchFakesTask",
           "MatchFakesConfig",
           "MatchVariableFakesConfig",
           "MatchVariableFakesTask"]

import astropy.units as u
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from lsst.geom import Box2D
import lsst.pex.config as pexConfig
from lsst.pipe.base import PipelineTask, PipelineTaskConnections, Struct
import lsst.pipe.base.connectionTypes as connTypes
from lsst.skymap import BaseSkyMap

from lsst.pipe.tasks.insertFakes import InsertFakesConfig


class MatchFakesConnections(PipelineTaskConnections,
                            defaultTemplates={"coaddName": "deep",
                                              "fakesType": "fakes_"},
                            dimensions=("instrument",
                                        "visit",
                                        "detector")):
    skyMap = connTypes.Input(
        doc="Input definition of geometry/bbox and projection/wcs for "
        "template exposures. Needed to test which tract to generate ",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        dimensions=("skymap",),
        storageClass="SkyMap",
    )
    fakeCats = connTypes.Input(
        doc="Catalog of fake sources inserted into an image.",
        name="{fakesType}fakeSourceCat",
        storageClass="DataFrame",
        dimensions=("tract", "skymap"),
        deferLoad=True,
        multiple=True
    )
    diffIm = connTypes.Input(
        doc="Difference image on which the DiaSources were detected.",
        name="{fakesType}{coaddName}Diff_differenceExp",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
    )
    associatedDiaSources = connTypes.Input(
        doc="A DiaSource catalog to match against fakeCat. Assumed "
            "to be SDMified.",
        name="{fakesType}{coaddName}Diff_assocDiaSrc",
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "detector"),
    )
    matchedDiaSources = connTypes.Output(
        doc="A catalog of those fakeCat sources that have a match in "
            "associatedDiaSources. The schema is the union of the schemas for "
            "``fakeCat`` and ``associatedDiaSources``.",
        name="{fakesType}{coaddName}Diff_matchDiaSrc",
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "detector"),
    )


class MatchFakesConfig(
        InsertFakesConfig,
        pipelineConnections=MatchFakesConnections):
    """Config for MatchFakesTask.
    """
    matchDistanceArcseconds = pexConfig.RangeField(
        doc="Distance in arcseconds to match within.",
        dtype=float,
        default=0.5,
        min=0,
        max=10,
    )

    doMatchVisit = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Match visit to trim the fakeCat"
    )

    trimBuffer = pexConfig.Field(
        doc="Size of the pixel buffer surrounding the image. Only those fake sources with a centroid"
        "falling within the image+buffer region will be considered matches.",
        dtype=int,
        default=100,
    )


class MatchFakesTask(PipelineTask):
    """Match a pre-existing catalog of fakes to a catalog of detections on
    a difference image.

    This task is generally for injected sources that cannot be easily
    identified by their footprints such as in the case of detector sources
    post image differencing.
    """

    _DefaultName = "matchFakes"
    ConfigClass = MatchFakesConfig

    def run(self, fakeCats, skyMap, diffIm, associatedDiaSources):
        """Compose fakes into a single catalog and match fakes to detected
        diaSources within a difference image bound.

        Parameters
        ----------
        fakeCats : `pandas.DataFrame`
            List of catalog of fakes to match to detected diaSources.
        skyMap : `lsst.skymap.SkyMap`
            SkyMap defining the tracts and patches the fakes are stored over.
        diffIm : `lsst.afw.image.Exposure`
            Difference image where ``associatedDiaSources`` were detected.
        associatedDiaSources : `pandas.DataFrame`
            Catalog of difference image sources detected in ``diffIm``.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results struct with components.

            - ``matchedDiaSources`` : Fakes matched to input diaSources. Has
              length of ``fakeCat``. (`pandas.DataFrame`)
        """
        fakeCat = self.composeFakeCat(fakeCats, skyMap)

        if self.config.doMatchVisit:
            fakeCat = self.getVisitMatchedFakeCat(fakeCat, diffIm)

        return self._processFakes(fakeCat, diffIm, associatedDiaSources)

    def _processFakes(self, fakeCat, diffIm, associatedDiaSources):
        """Match fakes to detected diaSources within a difference image bound.

        Parameters
        ----------
        fakeCat : `pandas.DataFrame`
            Catalog of fakes to match to detected diaSources.
        diffIm : `lsst.afw.image.Exposure`
            Difference image where ``associatedDiaSources`` were detected.
        associatedDiaSources : `pandas.DataFrame`
            Catalog of difference image sources detected in ``diffIm``.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results struct with components.

            - ``matchedDiaSources`` : Fakes matched to input diaSources. Has
              length of ``fakeCat``. (`pandas.DataFrame`)
        """
        trimmedFakes = self._trimFakeCat(fakeCat, diffIm)
        nPossibleFakes = len(trimmedFakes)

        fakeVects = self._getVectors(trimmedFakes[self.config.ra_col],
                                     trimmedFakes[self.config.dec_col])
        diaSrcVects = self._getVectors(
            np.radians(associatedDiaSources.loc[:, "ra"]),
            np.radians(associatedDiaSources.loc[:, "decl"]))

        diaSrcTree = cKDTree(diaSrcVects)
        dist, idxs = diaSrcTree.query(
            fakeVects,
            distance_upper_bound=np.radians(self.config.matchDistanceArcseconds / 3600))
        nFakesFound = np.isfinite(dist).sum()

        self.log.info("Found %d out of %d possible.", nFakesFound, nPossibleFakes)
        diaSrcIds = associatedDiaSources.iloc[np.where(np.isfinite(dist), idxs, 0)]["diaSourceId"].to_numpy()
        matchedFakes = trimmedFakes.assign(diaSourceId=np.where(np.isfinite(dist), diaSrcIds, 0))

        return Struct(
            matchedDiaSources=matchedFakes.merge(
                associatedDiaSources.reset_index(drop=True), on="diaSourceId", how="left")
        )

    def composeFakeCat(self, fakeCats, skyMap):
        """Concatenate the fakeCats from tracts that may cover the exposure.

        Parameters
        ----------
        fakeCats : `list` of `lst.daf.butler.DeferredDatasetHandle`
            Set of fake cats to concatenate.
        skyMap : `lsst.skymap.SkyMap`
            SkyMap defining the geometry of the tracts and patches.

        Returns
        -------
        combinedFakeCat : `pandas.DataFrame`
            All fakes that cover the inner polygon of the tracts in this
            quantum.
        """
        if len(fakeCats) == 1:
            return fakeCats[0].get()
        outputCat = []
        for fakeCatRef in fakeCats:
            cat = fakeCatRef.get()
            tractId = fakeCatRef.dataId["tract"]
            # Make sure all data is within the inner part of the tract.
            outputCat.append(cat[
                skyMap.findTractIdArray(cat[self.config.ra_col],
                                        cat[self.config.dec_col],
                                        degrees=False)
                == tractId])

        return pd.concat(outputCat)

    def getVisitMatchedFakeCat(self, fakeCat, exposure):
        """Trim the fakeCat to select particular visit

        Parameters
        ----------
        fakeCat : `pandas.core.frame.DataFrame`
                    The catalog of fake sources to add to the exposure
        exposure : `lsst.afw.image.exposure.exposure.ExposureF`
                    The exposure to add the fake sources to

        Returns
        -------
        movingFakeCat : `pandas.DataFrame`
            All fakes that belong to the visit
        """
        selected = exposure.getInfo().getVisitInfo().getId() == fakeCat["visit"]

        return fakeCat[selected]

    def _addPixCoords(self, fakeCat, image):

        """Add pixel coordinates to the catalog of fakes.

        Parameters
        ----------
        fakeCat : `pandas.core.frame.DataFrame`
            The catalog of fake sources to be input
        image : `lsst.afw.image.exposure.exposure.ExposureF`
            The image into which the fake sources should be added
        Returns
        -------
        fakeCat : `pandas.core.frame.DataFrame`
        """
        wcs = image.getWcs()
        ras = fakeCat[self.config.ra_col].values
        decs = fakeCat[self.config.dec_col].values
        xs, ys = wcs.skyToPixelArray(ras, decs)
        fakeCat["x"] = xs
        fakeCat["y"] = ys

        return fakeCat

    def _trimFakeCat(self, fakeCat, image):
        """Trim the fake cat to the exact size of the input image.

        Parameters
        ----------
        fakeCat : `pandas.core.frame.DataFrame`
            The catalog of fake sources that was input
        image : `lsst.afw.image.exposure.exposure.ExposureF`
            The image into which the fake sources were added
        Returns
        -------
        fakeCat : `pandas.core.frame.DataFrame`
            The original fakeCat trimmed to the area of the image
        """

        # fakeCat must be processed with _addPixCoords before trimming
        if ('x' not in fakeCat.columns) or ('y' not in fakeCat.columns):
            fakeCat = self._addPixCoords(fakeCat, image)

        # Prefilter in ra/dec to avoid cases where the wcs incorrectly maps
        # input fakes which are really off the chip onto it.
        ras = fakeCat[self.config.ra_col].values * u.rad
        decs = fakeCat[self.config.dec_col].values * u.rad

        isContainedRaDec = image.containsSkyCoords(ras, decs, padding=0)

        # now use the exact pixel BBox to filter to only fakes that were inserted
        xs = fakeCat["x"].values
        ys = fakeCat["y"].values

        bbox = Box2D(image.getBBox())
        isContainedXy = xs >= bbox.minX
        isContainedXy &= xs <= bbox.maxX
        isContainedXy &= ys >= bbox.minY
        isContainedXy &= ys <= bbox.maxY

        return fakeCat[isContainedRaDec & isContainedXy]

    def _getVectors(self, ras, decs):
        """Convert ra dec to unit vectors on the sphere.

        Parameters
        ----------
        ras : `numpy.ndarray`, (N,)
            RA coordinates in radians.
        decs : `numpy.ndarray`, (N,)
            Dec coordinates in radians.

        Returns
        -------
        vectors : `numpy.ndarray`, (N, 3)
            Vectors on the unit sphere for the given RA/DEC values.
        """
        vectors = np.empty((len(ras), 3))

        vectors[:, 2] = np.sin(decs)
        vectors[:, 0] = np.cos(decs) * np.cos(ras)
        vectors[:, 1] = np.cos(decs) * np.sin(ras)

        return vectors


class MatchVariableFakesConnections(MatchFakesConnections):
    ccdVisitFakeMagnitudes = connTypes.Input(
        doc="Catalog of fakes with magnitudes scattered for this ccdVisit.",
        name="{fakesType}ccdVisitFakeMagnitudes",
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "detector"),
    )


class MatchVariableFakesConfig(MatchFakesConfig,
                               pipelineConnections=MatchVariableFakesConnections):
    """Config for MatchFakesTask.
    """
    pass


class MatchVariableFakesTask(MatchFakesTask):
    """Match injected fakes to their detected sources in the catalog and
    compute their expected brightness in a difference image assuming perfect
    subtraction.

    This task is generally for injected sources that cannot be easily
    identified by their footprints such as in the case of detector sources
    post image differencing.
    """
    _DefaultName = "matchVariableFakes"
    ConfigClass = MatchVariableFakesConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        inputs["band"] = butlerQC.quantum.dataId["band"]

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, fakeCats, ccdVisitFakeMagnitudes, skyMap, diffIm, associatedDiaSources, band):
        """Match fakes to detected diaSources within a difference image bound.

        Parameters
        ----------
        fakeCat : `pandas.DataFrame`
            Catalog of fakes to match to detected diaSources.
        diffIm : `lsst.afw.image.Exposure`
            Difference image where ``associatedDiaSources`` were detected in.
        associatedDiaSources : `pandas.DataFrame`
            Catalog of difference image sources detected in ``diffIm``.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results struct with components.

            - ``matchedDiaSources`` : Fakes matched to input diaSources. Has
              length of ``fakeCat``. (`pandas.DataFrame`)
        """
        fakeCat = self.composeFakeCat(fakeCats, skyMap)
        self.computeExpectedDiffMag(fakeCat, ccdVisitFakeMagnitudes, band)
        return self._processFakes(fakeCat, diffIm, associatedDiaSources)

    def computeExpectedDiffMag(self, fakeCat, ccdVisitFakeMagnitudes, band):
        """Compute the magnitude expected in the difference image for this
        detector/visit. Modify fakeCat in place.

        Negative magnitudes indicate that the source should be detected as
        a negative source.

        Parameters
        ----------
        fakeCat : `pandas.DataFrame`
            Catalog of fake sources.
        ccdVisitFakeMagnitudes : `pandas.DataFrame`
            Magnitudes for variable sources in this specific ccdVisit.
        band : `str`
            Band that this ccdVisit was observed in.
        """
        magName = self.config.mag_col % band
        magnitudes = fakeCat[magName].to_numpy()
        visitMags = ccdVisitFakeMagnitudes["variableMag"].to_numpy()
        diffFlux = (visitMags * u.ABmag).to_value(u.nJy) - (magnitudes * u.ABmag).to_value(u.nJy)
        diffMag = np.where(diffFlux > 0,
                           (diffFlux * u.nJy).to_value(u.ABmag),
                           -(-diffFlux * u.nJy).to_value(u.ABmag))

        noVisit = ~fakeCat["isVisitSource"]
        noTemplate = ~fakeCat["isTemplateSource"]
        both = np.logical_and(fakeCat["isVisitSource"],
                              fakeCat["isTemplateSource"])

        fakeCat.loc[noVisit, magName] = -magnitudes[noVisit]
        fakeCat.loc[noTemplate, magName] = visitMags[noTemplate]
        fakeCat.loc[both, magName] = diffMag[both]
