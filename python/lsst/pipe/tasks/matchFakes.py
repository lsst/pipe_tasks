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


import numpy as np
from scipy.spatial import cKDTree

import lsst.geom as lsstGeom
import lsst.pex.config as pexConfig
from lsst.pipe.base import PipelineTask, PipelineTaskConnections, Struct
import lsst.pipe.base.connectionTypes as connTypes
from lsst.pipe.tasks.insertFakes import InsertFakesConfig

__all__ = ["MatchFakesTask",
           "MatchFakesConfig",
           "MatchFakesConnections"]


class MatchFakesConnections(PipelineTaskConnections,
                            defaultTemplates={"coaddName": "deep",
                                              "fakesType": "fakes_"},
                            dimensions=("tract",
                                        "skymap",
                                        "instrument",
                                        "visit",
                                        "detector")):
    fakeCat = connTypes.Input(
        doc="Catalog of fake sources inserted into an image.",
        name="{fakesType}fakeSourceCat",
        storageClass="DataFrame",
        dimensions=("tract", "skymap")
    )
    image = connTypes.Input(
        doc="Image on which the sources were detected.",
        name="{fakesType}{coaddName}Diff_differenceExp",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
    )
    detectedSources = connTypes.Input(
        doc="A source or DiaSource catalog to match against fakeCat.",
        name="{fakesType}{coaddName}Diff_assocDiaSrc",
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "detector"),
    )
    matchedSources = connTypes.Output(
        doc="A catalog of those fakeCat sources that have a match in "
            "detectedSources. The schema is the union of the schemas for "
            "``fakeCat`` and ``detectedSources``; if ``config.src_id_col`` is "
            "`None`, this schema also includes a ``matchedSourceId`` column.",
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
    src_id_col = pexConfig.Field(
        doc="Column in detectedSources containing the source ID. If None, "
            "detectedSources's index will be used as the ID instead (and must "
            "be 1 column)",
        dtype=str,
        default="diaSourceId",
        optional=True,
    )
    src_radec_units = pexConfig.ChoiceField(
        doc="The units for src_ra_col and src_dec_col.",
        dtype=str,
        allowed={
            "degrees": "Assume detectedSources gives coordinates in degrees.",
            "radians": "Assume detectedSources gives coordinates in radians.",
        },
        default="degrees",
        optional=False,
    )
    src_ra_col = pexConfig.Field(
        doc="Column in detectedSources containing the J2000 right ascension.",
        dtype=str,
        default="ra",
    )
    src_dec_col = pexConfig.Field(
        doc="Column in detectedSources containing the J2000 declination.",
        dtype=str,
        default="decl",
    )


class MatchFakesTask(PipelineTask):
    """Match a pre-existing catalog of fakes to a catalog of detections on
    a direct or difference image.
    """

    _DefaultName = "matchFakes"
    ConfigClass = MatchFakesConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, fakeCat, image, detectedSources):
        """Match fakes to detected sources within an image bound.

        Parameters
        ----------
        fakeCat : `pandas.DataFrame`
            Catalog of fakes to match to detected sources.
        image : `lsst.afw.image.Exposure`
            Difference image where ``detectedSources`` were detected.
        detectedSources : `pandas.DataFrame`
            Catalog of sources detected in ``image``.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results struct with components.

            - ``matchedSources`` : Fakes matched to input sources. Has
              length of ``fakeCat``. (`pandas.DataFrame`)

        Raises
        ------
        ValueError:
            Raised if this task is configured to use the index as the
            source IDs, but ``detectedSources`` has a multi-column index.
        """
        trimmedFakes = self._trimFakeCat(fakeCat, image)
        nPossibleFakes = len(trimmedFakes)

        fakeVects = self._getVectors(trimmedFakes[self.config.raColName],
                                     trimmedFakes[self.config.decColName])
        if self.config.src_radec_units == "degrees":
            srcVects = self._getVectors(
                np.radians(detectedSources.loc[:, self.config.src_ra_col]),
                np.radians(detectedSources.loc[:, self.config.src_dec_col]))
        else:
            srcVects = self._getVectors(
                detectedSources.loc[:, self.config.src_ra_col],
                detectedSources.loc[:, self.config.src_dec_col])

        srcTree = cKDTree(srcVects)
        dist, idxs = srcTree.query(
            fakeVects,
            distance_upper_bound=np.radians(self.config.matchDistanceArcseconds / 3600))
        nFakesFound = np.isfinite(dist).sum()

        self.log.info("Found %d out of %d possible.", nFakesFound, nPossibleFakes)
        rows = detectedSources.iloc[np.where(np.isfinite(dist), idxs, 0)]
        if self.config.src_id_col is not None:
            srcIds = rows[self.config.src_id_col].to_numpy()
            matchedFakes = trimmedFakes.assign(**{
                self.config.src_id_col: np.where(np.isfinite(dist), srcIds, 0)
            })
            matchedSources = matchedFakes.merge(
                detectedSources.reset_index(drop=True), on=self.config.src_id_col, how="left")
        else:
            if rows.index.nlevels != 1:
                raise ValueError(f"Expected 1-column index, got {rows.index.names}.")
            srcIds = rows.index.to_numpy()
            matchedFakes = trimmedFakes.assign(**{"matchedSourceId": np.where(np.isfinite(dist), srcIds, 0)})
            matchedSources = matchedFakes.merge(
                detectedSources, how="left",
                left_on="matchedSourceId", right_index=True)
            matchedSources.reset_index(drop=True, inplace=True)

        return Struct(matchedSources=matchedSources)

    def _trimFakeCat(self, fakeCat, image):
        """Trim the fake cat to about the size of the input image.

        Parameters
        ----------
        fakeCat : `pandas.core.frame.DataFrame`
            The catalog of fake sources to be input
        image : `lsst.afw.image.exposure.exposure.ExposureF`
            The image into which the fake sources should be added

        Returns
        -------
        fakeCat : `pandas.core.frame.DataFrame`
            The original fakeCat trimmed to the area of the image
        """
        wcs = image.getWcs()

        bbox = lsstGeom.Box2D(image.getBBox())

        def trim(row):
            coord = lsstGeom.SpherePoint(row[self.config.raColName],
                                         row[self.config.decColName],
                                         lsstGeom.radians)
            cent = wcs.skyToPixel(coord)
            return bbox.contains(cent)

        return fakeCat[fakeCat.apply(trim, axis=1)]

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
