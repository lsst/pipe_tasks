# This file is part of pipe_tasks.

# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

"""Simple association algorithm for DRP. 
Adapted from http://github.com/LSSTDESC/dia_pipe
"""

import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from .associationUtils import query_disc, eq2xyz, toIndex


class SimpleAssociationConfig(pexConfig.Config):
    """Configuration parameters for the SimpleAssociationTask
    """
    tolerance = pexConfig.Field(
        dtype=float,
        doc='maximum distance to match sources together in arcsec',
        default=0.5
    )
    nside = pexConfig.Field(
        dtype=int,
        doc='Healpix nside value used for indexing',
        default=2**18,
    )


class SimpleAssociationTask(pipeBase.Task):
    """Construct DIAObjects from a DataFrame of DIASources
    """
    ConfigClass = SimpleAssociationConfig
    _DefaultName = "simpleAssociation"

    def run(self, diaSources, tractPatchId, skymapBits):
        """Associate DiaSources into a collection of DiaObjects using a
        brute force matching algorithm.

        Reproducible is for the same input data is assured by ordering the
        DiaSource data by ccdVisit ordering.

        Parameters
        ----------
        diaSources : `pandas.DataFrame`
            DiaSources to spatially associate into DiaObjects
        tractPatchId : `int`
            Unique identifier for the tract patch.
        skymapBits : `int`
            Maximum number of bits used the ``tractPatchId`` integer
            identifier.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            Results struct with attributes:

            ``assocDiaSources``
                Table of DiaSources with updated values for the DiaObjects
                they are spatially associated to (`pandas.DataFrame`).
            ``diaObjects``
                Table of DiaObjects from matching DiaSources
                (`pandas.DataFrame`).

        """
        # Sort by ccdVisit and diaSourceId to get a reproducible ordering for
        # the association.
        diaSources.set_index(["ccdVisit", "diaSourceId"], inplace=True)

        # Empty lists to store matching and location data.
        diaObjCat = []
        diaObjCoords = []
        healPixIndices = []

        # Create Id factory and catalog for creating DiaObjectIds.
        idFactory = afwTable.IdFactory.makeSource(tractPatchId,
                                                  64 - skymapBits)
        idCat = afwTable.SourceCatalog(
            afwTable.SourceTable.make(afwTable.SourceTable.makeMinimalSchema(),
                                      idFactory))

        for ccdVisit in diaSources.index.levels[0]:
            # For the first ccdVisit, just copy the DiaSource info into the
            # diaObject data to create the first set of Objects.
            ccdVisitSources = diaSources.loc[ccdVisit]
            if len(diaObjectCat) == 0:
                for diaSourceId, diaSrc in ccdVisitSources:
                    self.addNewDiaObject(diaSrc,
                                         diaSources,
                                         ccdVisit,
                                         diaSourceId,
                                         diaObjectCat,
                                         idCat,
                                         diaObjectCoordList,
                                         healPixIndexes)

            # Run over subsequent data.
            for diaSourceId, diaSrc in ccdVisitSources:
                # Find matches.
                dists, matches = self.findMatches(diaSrc["ra"],
                                                  diaSrc["decl"]
                                                  2*self.config.tolerance,
                                                  healPixIndices,
                                                  diaObjCat)
                # Create a new DiaObject if no match found.
                if dists is None:
                    self.addNewDiaObject(diaSrc,
                                         diaSources,
                                         ccdVisit,
                                         diaSourceId,
                                         diaObjectCat,
                                         idCat,
                                         diaObjectCoordList,
                                         healPixIndexes)
                    continue
                # If matched, update catalogs and arrays.
                if np.min(dists) < np.deg2rad(self.config.tolerance/3600):
                    match_dist_arg = np.argmin(dists)
                    match_index = np.where(matches)[0][match_dist_arg]
                    self.updateDiaObjectCat(match_index,
                                            diaSrc,
                                            diaSources,
                                            ccdVisit,
                                            diaSourceId,
                                            diaObjCat,
                                            diaObjCoords,
                                            healPixIndices)
                # Create new DiaObject if no match found within the matching
                # tolerance.
                else:
                    self.addNewDiaObject(diaSrc,
                                         diaSources,
                                         ccdVisit,
                                         diaSourceId,
                                         diaObjectCat,
                                         idCat,
                                         diaObjectCoordList,
                                         healPixIndexes)

        # Drop indices before returning association diaSource catalog.
        diaSources.reset_index(inplace=True)

        return pipeBase.Struct(
            assocDiaSources=diaSources,
            diaObjects=pd.DataFrame(data=diaObjectCat))

    def addNewDiaObject(self,
                        diaSrc,
                        diaSources,
                        ccdVisit,
                        diaSourceId,
                        diaObjCat,
                        idCat,
                        diaObjCoords,
                        healPixIndices):
        """Create a new DiaObject and append its data.

        Parameters
        ----------
        diaSrc : `pandas.Series`
            Full unassociated DiaSource to create a DiaObject from.
        diaSources : `pandas.DataFrame`
            DiaSource catalog to update information in.
        ccdVisit : `int`
            Unique identifier of the ccdVisit where ``diaSrc`` was observed.
        diaSourceId : `int`
            Unique identifier of the DiaSource.
        diaObjectCat : `list` of `dict`s
            Catalog of diaObjects to append the new object o.
        idCat : `lsst.afw.table.SourceCatalog`
            Catalog with the IdFactory used to generate unique DiaObject
            identifiers.
        DiaObjectCoords : `list` of `list`s of `lsst.geom.SpherePoint`s
            Set of coordinates of DiaSource locations that make up the
            DiaObject average coordinate.
        healPixIndices : `list` of `int`s
            HealPix indices representing the locations of each currently
            existing DiaObject.
        """
        hpIndex = toIndex(self.config.nside,
                          diaSrc["ra"],
                          diaSrc["decl"])
        healPixIndices.append(hpIndex)

        sphPoint = geom.SpherePoint(diaSrc["ra"],
                                    diaSrc["decl"],
                                    geom.degrees)
        diaObjCoords.append([sphPoint])

        diaObjId = idCat.addNew().get("id")
        diaObjCat.append(self.createDiaObject(diaObjId))
        diaSources.loc[(ccdVisit, diaSourceId), "diaObjectId"] = diaObjId

    def updateCatalogs(self,
                       match_index,
                       diaSrc,
                       diaSources,
                       ccdVisit,
                       diaSourceId,
                       diaObjCat,
                       diaObjCoords,
                       healPixIndices):
        """Create a new DiaObject and append its data.

        Parameters
        ----------
        match_index : `int`
            Array index location of the DiaObject that ``diaSrc`` was
            associated to.
        diaSrc : `pandas.Series`
            Full unassociated DiaSource to create a DiaObject from.
        diaSources : `pandas.DataFrame`
            DiaSource catalog to update information in.
        ccdVisit : `int`
            Unique identifier of the ccdVisit where ``diaSrc`` was observed.
        diaSourceId : `int`
            Unique identifier of the DiaSource.
        diaObjectCat : `list` of `dict`s
            Catalog of diaObjects to append the new object o.
        DiaObjectCoords : `list` of `list`s of `lsst.geom.SpherePoint`s
            Set of coordinates of DiaSource locations that make up the
            DiaObject average coordinate.
        healPixIndices : `list` of `int`s
            HealPix indices representing the locations of each currently
            existing DiaObject.
        """
        # Update location and healPix index.
        sphPoint = geom.SpherePoint(diaSrc["ra"],
                                    diaSrc["decl"],
                                    geom.degrees)
        diaObjCoords[match_index].append(sphPoint)
        aveCoord = geom.averageSpherePoint(diaObjCoords[match_index])
        diaObjCat[match_index]["ra"] = aveCoord.getRa().asDegrees()
        diaObjCat[match_index]["decl"] = aveCoord.getDec().asDegrees()
        healPixIndices[match_index] = toIndex(self.config.nside,
                                              diaObjCat[match_index]["ra"],
                                              diaObjCat[match_index]["decl"])
        # Add update DiaObject Id that this source is now associated to.
        diaSources.loc[(ccdVisit, diaSourceId), "diaObjectId"] = /
            diaObjCat[match_index]["diaObjectId"]

    def findMatches(self, src_ra, src_dec, tol, hpIndices, diaObjs):
        """Search healPixels around DiaSource locations for DiaObjects.

        Parameters
        ----------
        src_ra : `float`
            DiaSource RA location.
        src_dec : `float`
            DiaSource Dec location.
        tol : `flat`
            Size of annulus to convert to covering healPixels and search for
            DiaObjects.
        diaObjs : `list` of `dict`s
            Catalog diaObjects to with full location information for comparing
            to DiaSources.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            Results struct containing

            ``dists``
                Array of distances between the current DiaSource diaObjects.
                (`numpy.ndarray` or `None`)
            ``matches``
                Array of array indices of diaObjects this DiaSource matches to.
                (`numpy.ndarray` or `None`)
        """
        match_indices = query_disc(self.config.nside,
                                   src_ra,
                                   src_dec,
                                   np.deg2rad(tol/3600.))
        matches = np.isin(hpIndices, match_indices)

        if np.sum(matches) < 1:
            return pipeBase.Struct(dists=None, matches=None)

        dists = np.array(
            [np.sqrt(np.sum((eq2xyz(src_ra, src_dec) -
                             eq2xyz(diaObj["ra"], diaObj["decl"]))**2))
             for diaObj in diaObjs[matches]])
        return pipeBase.Struct(
            dists=dists,
            matches=matches)

    def createDiaObject(self, objId, ra, decl):
        """Create a simple empty DiaObject with location and id information.

        Parameters
        ----------
        objId : `int`
            Unique ID for this new DiaObject.
        ra : `float`
            RA location of this DiaObject.
        decl : `float`
            Dec location of this DiaObject

        Returns
        -------
        DiaObject : `dict`
            Dictionary of values representing a DiaObject.
        """
        new_dia_object = {"diaObjectId": objId,
                          "ra": ra,
                          "decl": decl,
                          "pmParallaxNdata": 0,
                          "nearbyObj1": 0,
                          "nearbyObj2": 0,
                          "nearbyObj3": 0,
                          "flags": 0}
        for f in ["u", "g", "r", "i", "z", "y"]:
            new_dia_object["%sPSFluxNdata" % f] = 0
        return new_dia_object
