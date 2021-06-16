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

"""Methods to match an input catalog to a set of fakes in AP.
"""

import numpy as np
from scipy.spatial import cKDTree

from lsst.geom import Box2D, radians, SpherePoint
import lsst.pex.config as pexConfig
from lsst.pipe.base import PipelineTask, PipelineTaskConnections, Struct
import lsst.pipe.base.connectionTypes as connTypes
from lsst.pipe.tasks.insertFakes import InsertFakesConfig

__all__ = ["MatchApFakesTask",
           "MatchApFakesConfig",
           "MatchApFakesConnections"]


class MatchApFakesConnections(PipelineTaskConnections,
                              defaultTemplates={"coaddName": "deep",
                                                "fakesType": "fakes_"},
                              dimensions=("tract",
                                          "skymap",
                                          "instrument",
                                          "visit",
                                          "detector")):
    fakeCat = connTypes.Input(
        doc="Catalog of fake sources to draw inputs from.",
        name="{fakesType}fakeSourceCat",
        storageClass="DataFrame",
        dimensions=("tract", "skymap")
    )
    diffIm = connTypes.Input(
        doc="Difference image on which the DiaSources were detected.",
        name="{fakesType}{coaddName}Diff_differenceExp",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
    )
    associatedDiaSources = connTypes.Input(
        doc="Optional output storing the DiaSource catalog after matching and "
            "SDMification.",
        name="{fakesType}{coaddName}Diff_assocDiaSrc",
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "detector"),
    )
    matchedDiaSources = connTypes.Output(
        doc="",
        name="{fakesType}{coaddName}Diff_matchDiaSrc",
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "detector"),
    )


class MatchApFakesConfig(
        InsertFakesConfig,
        pipelineConnections=MatchApFakesConnections):
    """Config for MatchApFakesTask.
    """
    matchDistanceArcseconds = pexConfig.RangeField(
        doc="Distance in arcseconds to ",
        dtype=float,
        default=0.5,
        min=0,
        max=10,
    )


class MatchApFakesTask(PipelineTask):
    """Create and store a set of spatially uniform star fakes over the sphere
    for use in AP processing. Additionally assign random magnitudes to said
    fakes and assign them to be inserted into either a visit exposure or
    template exposure.
    """

    _DefaultName = "matchApFakes"
    ConfigClass = MatchApFakesConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, fakeCat, diffIm, associatedDiaSources):
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
        trimmedFakes = self._trimFakeCat(fakeCat, diffIm)
        nPossibleFakes = len(trimmedFakes)

        fakeVects = self._getVectors(trimmedFakes[self.config.raColName],
                                     trimmedFakes[self.config.decColName])
        diaSrcVects = self._getVectors(
            np.radians(associatedDiaSources.loc[:, "ra"]),
            np.radians(associatedDiaSources.loc[:, "decl"]))

        diaSrcTree = cKDTree(diaSrcVects)
        dist, idxs = diaSrcTree.query(
            fakeVects,
            distance_upper_bound=np.radians(self.config.matchDistanceArcseconds / 3600))
        nFakesFound = np.isfinite(dist).sum()

        self.log.info(f"Found {nFakesFound} out of {nPossibleFakes} possible.")
        diaSrcIds = associatedDiaSources.iloc[np.where(np.isfinite(dist), idxs, 0)]["diaSourceId"].to_numpy()
        matchedFakes = trimmedFakes.assign(diaSourceId=np.where(np.isfinite(dist), diaSrcIds, 0))

        return Struct(
            matchedDiaSources=matchedFakes.merge(
                associatedDiaSources.reset_index(drop=True), on="diaSourceId", how="left")
        )

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

        bbox = Box2D(image.getBBox())

        def trim(row):
            coord = SpherePoint(row[self.config.raColName],
                                row[self.config.decColName],
                                radians)
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
