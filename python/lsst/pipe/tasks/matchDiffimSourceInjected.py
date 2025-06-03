# This file is part of ap_pipe.
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

__all__ = ["MatchInjectedToDiaSourceTask",
           "MatchInjectedToDiaSourceConfig",
           "MatchInjectedToAssocDiaSourceTask",
           "MatchInjectedToAssocDiaSourceConfig"]

import astropy.units as u
import numpy as np
from scipy.spatial import cKDTree

from lsst.afw import table as afwTable
from lsst import geom as lsstGeom
import lsst.pex.config as pexConfig
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
import lsst.pipe.base.connectionTypes as connTypes
from lsst.meas.base import ForcedMeasurementTask, ForcedMeasurementConfig


class MatchInjectedToDiaSourceConnections(
        PipelineTaskConnections,
        defaultTemplates={"coaddName": "deep",
                          "fakesType": "fakes_"},
        dimensions=("instrument",
                    "visit",
                    "detector")):
    injectedCat = connTypes.Input(
        doc="Catalog of sources injected in the images.",
        name="{fakesType}_pvi_catalog",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit", "detector"),
    )
    diffIm = connTypes.Input(
        doc="Difference image on which the DiaSources were detected.",
        name="{fakesType}{coaddName}Diff_differenceExp",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
    )
    diaSources = connTypes.Input(
        doc="A DiaSource catalog to match against fakeCat.",
        name="{fakesType}{coaddName}Diff_diaSrc",
        storageClass="SourceCatalog",
        dimensions=("instrument", "visit", "detector"),
    )
    matchDiaSources = connTypes.Output(
        doc="A catalog of those fakeCat sources that have a match in "
            "diaSrc. The schema is the union of the schemas for "
            "``fakeCat`` and ``diaSrc``.",
        name="{fakesType}{coaddName}Diff_matchDiaSrc",
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "detector"),
    )


class MatchInjectedToDiaSourceConfig(
        PipelineTaskConfig,
        pipelineConnections=MatchInjectedToDiaSourceConnections):
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
        default=True,
        doc="Match visit to trim the fakeCat"
    )
    trimBuffer = pexConfig.Field(
        doc="Size of the pixel buffer surrounding the image."
            "Only those fake sources with a centroid"
            "falling within the image+buffer region will be considered matches.",
        dtype=int,
        default=50,
    )
    doForcedMeasurement = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Force measurement of the fakes at the injection locations."
    )
    forcedMeasurement = pexConfig.ConfigurableField(
        target=ForcedMeasurementTask,
        doc="Task to force photometer difference image at injection locations.",
    )


class MatchInjectedToDiaSourceTask(PipelineTask):

    _DefaultName = "matchInjectedToDiaSource"
    ConfigClass = MatchInjectedToDiaSourceConfig

    def run(self, injectedCat, diffIm, diaSources):
        """Match injected sources to detected diaSources within a difference image bound.

        Parameters
        ----------
        injectedCat : `astropy.table.table.Table`
            Table of catalog of synthetic sources to match to detected diaSources.
        diffIm : `lsst.afw.image.Exposure`
            Difference image where ``diaSources`` were detected.
        diaSources : `afw.table.SourceCatalog`
            Catalog of difference image sources detected in ``diffIm``.
        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results struct with components.

            - ``matchedDiaSources`` : Fakes matched to input diaSources. Has
              length of ``injectedCalexpCat``. (`pandas.DataFrame`)
        """

        if self.config.doMatchVisit:
            fakeCat = self._trimFakeCat(injectedCat, diffIm)
        else:
            fakeCat = injectedCat
        if self.config.doForcedMeasurement:
            self._estimateFakesSNR(fakeCat, diffIm)

        return self._processFakes(fakeCat, diaSources)

    def _estimateFakesSNR(self, injectedCat, diffIm):
        """Estimate the signal-to-noise ratio of the fakes in the given catalog.

        Parameters
        ----------
        injectedCat : `astropy.table.Table`
            Catalog of synthetic sources to estimate the S/N of. **This table
            will be modified in place**.
        diffIm : `lsst.afw.image.Exposure`
            Difference image where the sources were detected.
        """
        # Create a schema for the forced measurement task
        schema = afwTable.SourceTable.makeMinimalSchema()
        schema.addField("x", "D", "x position in image.", units="pixel")
        schema.addField("y", "D", "y position in image.", units="pixel")
        schema.addField("deblend_nChild", "I", "Need for minimal forced phot schema")

        pluginList = [
            "base_PixelFlags",
            "base_SdssCentroid",
            "base_CircularApertureFlux",
            "base_PsfFlux",
            "base_LocalBackground"
        ]
        forcedMeasConfig = ForcedMeasurementConfig(plugins=pluginList)
        forcedMeasConfig.slots.centroid = 'base_SdssCentroid'
        forcedMeasConfig.slots.shape = None

        # Create the forced measurement task
        forcedMeas = ForcedMeasurementTask(schema, config=forcedMeasConfig)

        # Specify the columns to copy from the input catalog to the output catalog
        forcedMeas.copyColumns = {"coord_ra": "ra", "coord_dec": "dec"}

        # Create an afw table from the input catalog
        outputCatalog = afwTable.SourceCatalog(schema)
        outputCatalog.reserve(len(injectedCat))
        for row in injectedCat:
            outputRecord = outputCatalog.addNew()
            outputRecord.setId(row['injection_id'])
            outputRecord.setCoord(lsstGeom.SpherePoint(row["ra"], row["dec"], lsstGeom.degrees))
            outputRecord.set("x", row["x"])
            outputRecord.set("y", row["y"])

        # Generate the forced measurement catalog
        forcedSources = forcedMeas.generateMeasCat(diffIm, outputCatalog, diffIm.getWcs())
        # Attach the PSF shape footprints to the forced measurement catalog
        forcedMeas.attachPsfShapeFootprints(forcedSources, diffIm)

        # Copy the x and y positions from the forced measurement catalog back
        # to the input catalog
        for src, tgt in zip(forcedSources, outputCatalog):
            src.set('base_SdssCentroid_x', tgt['x'])
            src.set('base_SdssCentroid_y', tgt['y'])

        # Define the centroid for the forced measurement catalog
        forcedSources.defineCentroid('base_SdssCentroid')
        # Run the forced measurement task
        forcedMeas.run(forcedSources, diffIm, outputCatalog, diffIm.getWcs())
        # Convert the forced measurement catalog to an astropy table
        forcedSources_table = forcedSources.asAstropy()

        # Add the forced measurement columns to the input catalog
        for column in forcedSources_table.columns:
            if "Flux" in column or "flag" in column:
                injectedCat["forced_"+column] = forcedSources_table[column]

        # Add the SNR columns to the input catalog
        for column in injectedCat.colnames:
            if column.endswith("instFlux"):
                flux = injectedCat[column]
                fluxErr = injectedCat[column+"Err"].copy()
                fluxErr = np.where(
                    (fluxErr <= 0) | (np.isnan(fluxErr)), np.nanmax(fluxErr), fluxErr)

                injectedCat[column+"_SNR"] = flux / fluxErr

    def _processFakes(self, injectedCat, diaSources):
        """Match fakes to detected diaSources within a difference image bound.

        Parameters
        ----------
        injectedCat : `astropy.table.table.Table`
            Catalog of injected sources to match to detected diaSources.
        diaSources : `afw.table.SourceCatalog`
            Catalog of difference image sources detected in ``diffIm``.
        associatedDiaSources : `pandas.DataFrame`
            Catalog of associated difference image sources detected in ``diffIm``.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results struct with components.

            - ``matchedDiaSources`` : Fakes matched to input diaSources. Has
              length of ``fakeCat``. (`pandas.DataFrame`)
        """
        # First match the diaSrc to the injected fakes
        injectedCat = injectedCat.to_pandas()
        nPossibleFakes = len(injectedCat)

        fakeVects = self._getVectors(
            np.radians(injectedCat.ra),
            np.radians(injectedCat.dec))
        diaSrcVects = self._getVectors(
            diaSources['coord_ra'],
            diaSources['coord_dec'])

        diaSrcTree = cKDTree(diaSrcVects)
        dist, idxs = diaSrcTree.query(
            fakeVects,
            distance_upper_bound=np.radians(self.config.matchDistanceArcseconds / 3600))
        nFakesFound = np.isfinite(dist).sum()

        self.log.info("Found %d out of %d possible in diaSources.", nFakesFound, nPossibleFakes)

        # assign diaSourceId to the matched fakes
        diaSrcIds = diaSources['id'][np.where(np.isfinite(dist), idxs, 0)]
        matchedFakes = injectedCat.assign(diaSourceId=np.where(np.isfinite(dist), diaSrcIds, 0))
        matchedFakes['dist_diaSrc'] = np.where(np.isfinite(dist), 3600*np.rad2deg(dist), -1)

        return Struct(matchDiaSources=matchedFakes)

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

    def _addPixCoords(self, fakeCat, image):
        """Add pixel coordinates to the catalog of fakes.

        Parameters
        ----------
        fakeCat : `astropy.table.table.Table`
            The catalog of fake sources to be input
        image : `lsst.afw.image.exposure.exposure.ExposureF`
            The image into which the fake sources should be added
        Returns
        -------
        fakeCat : `astropy.table.table.Table`
        """

        wcs = image.getWcs()

        # Get x/y pixel coordinates for injected sources.
        xs, ys = wcs.skyToPixelArray(
            fakeCat["ra"],
            fakeCat["dec"],
            degrees=True
        )
        fakeCat["x"] = xs
        fakeCat["y"] = ys

        return fakeCat

    def _trimFakeCat(self, fakeCat, image):
        """Trim the fake cat to the exact size of the input image.

        Parameters
        ----------
        fakeCat : `astropy.table.table.Table`
            The catalog of fake sources that was input
        image : `lsst.afw.image.exposure.exposure.ExposureF`
            The image into which the fake sources were added
        Returns
        -------
        fakeCat : `astropy.table.table.Table`
            The original fakeCat trimmed to the area of the image
        """

        # fakeCat must be processed with _addPixCoords before trimming
        fakeCat = self._addPixCoords(fakeCat, image)

        # Prefilter in ra/dec to avoid cases where the wcs incorrectly maps
        # input fakes which are really off the chip onto it.
        ras = fakeCat["ra"] * u.deg
        decs = fakeCat["dec"] * u.deg

        isContainedRaDec = image.containsSkyCoords(ras, decs, padding=0)

        # now use the exact pixel BBox to filter to only fakes that were inserted
        xs = fakeCat["x"]
        ys = fakeCat["y"]

        bbox = lsstGeom.Box2D(image.getBBox())
        isContainedXy = xs - self.config.trimBuffer >= bbox.minX
        isContainedXy &= xs + self.config.trimBuffer <= bbox.maxX
        isContainedXy &= ys - self.config.trimBuffer >= bbox.minY
        isContainedXy &= ys + self.config.trimBuffer <= bbox.maxY

        return fakeCat[isContainedRaDec & isContainedXy]


class MatchInjectedToAssocDiaSourceConnections(
    PipelineTaskConnections,
    defaultTemplates={"coaddName": "deep",
                      "fakesType": "fakes_"},
    dimensions=("instrument",
                "visit",
                "detector")):

    assocDiaSources = connTypes.Input(
        doc="An assocDiaSource catalog to match against fakeCat from the"
            "diaPipe run. Assumed to be SDMified.",
        name="{fakesType}{coaddName}Diff_assocDiaSrc",
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "detector"),
    )
    matchDiaSources = connTypes.Input(
        doc="A catalog of those fakeCat sources that have a match in "
            "diaSrc. The schema is the union of the schemas for "
            "``fakeCat`` and ``diaSrc``.",
        name="{fakesType}{coaddName}Diff_matchDiaSrc",
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "detector"),
    )
    matchAssocDiaSources = connTypes.Output(
        doc="A catalog of those fakeCat sources that have a match in "
            "associatedDiaSources. The schema is the union of the schemas for "
            "``fakeCat`` and ``associatedDiaSources``.",
        name="{fakesType}{coaddName}Diff_matchAssocDiaSrc",
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "detector"),
    )


class MatchInjectedToAssocDiaSourceConfig(
        PipelineTaskConfig,
        pipelineConnections=MatchInjectedToAssocDiaSourceConnections):
    """Config for MatchFakesTask.
    """


class MatchInjectedToAssocDiaSourceTask(PipelineTask):

    _DefaultName = "matchInjectedToAssocDiaSource"
    ConfigClass = MatchInjectedToAssocDiaSourceConfig

    def run(self, assocDiaSources, matchDiaSources):
        """Tag matched injected sources to associated diaSources.

        Parameters
        ----------
        matchDiaSources : `pandas.DataFrame`
            Catalog of matched diaSrc to injected sources
        assocDiaSources : `pandas.DataFrame`
            Catalog of associated difference image sources detected in ``diffIm``.
        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results struct with components.

            - ``matchAssocDiaSources`` : Fakes matched to associated diaSources. Has
              length of ``matchDiaSources``. (`pandas.DataFrame`)
        """
        # Match the fakes to the associated sources. For this we don't use the coordinates
        # but instead check for the diaSources. Since they were present in the table already
        nPossibleFakes = len(matchDiaSources)
        matchDiaSources['isAssocDiaSource'] = matchDiaSources.diaSourceId.isin(assocDiaSources.diaSourceId)
        assocNFakesFound = matchDiaSources.isAssocDiaSource.sum()
        self.log.info("Found %d out of %d possible in assocDiaSources."%(assocNFakesFound, nPossibleFakes))

        return Struct(
            matchAssocDiaSources=matchDiaSources.merge(
                assocDiaSources.reset_index(drop=True),
                on="diaSourceId",
                how="left",
                suffixes=('_ssi', '_diaSrc')
            )
        )
