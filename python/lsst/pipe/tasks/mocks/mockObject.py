#
# LSST Data Management System
# Copyright 2008, 2009, 2010, 2011, 2012 LSST Corporation.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

import numpy
import pdb

import lsst.pex.config
import lsst.afw.table
import lsst.afw.geom
import lsst.afw.image
import lsst.pipe.base

class MockObjectConfig(lsst.pex.config.Config):
    minMag = lsst.pex.config.Field(dtype=float, default=18.0, doc="Minimum magnitude for mock objects")
    maxMag = lsst.pex.config.Field(dtype=float, default=20.0, doc="Maximum magnitude for mock objects")
    maxRadius = lsst.pex.config.Field(
        dtype=float, default=10.0,
        doc=("Maximum radius of an object in arcseconds; only used "
             "when determining which objects are in an exposure.")
        )
    spacing = lsst.pex.config.Field(
        dtype=float, default=20.0,
        doc="Distance between objects (in arcseconds)."
        )
    seed = lsst.pex.config.Field(dtype=int, default=1, doc="Seed for numpy random number generator")

class MockObjectTask(lsst.pipe.base.Task):
    """Task that generates simple mock objects and draws them on images, intended as a subtask of
    MockCoaddTask.

    May be subclassed to generate things other than stars.
    """

    ConfigClass = MockObjectConfig

    def __init__(self, **kwds):
        lsst.pipe.base.Task.__init__(self, **kwds)
        self.schema = lsst.afw.table.SimpleTable.makeMinimalSchema()
        self.center = lsst.afw.table.Point2DKey.addFields(self.schema, "center",
                                           "center position in tract WCS", "pixel")
        self.magKey = self.schema.addField("mag", type=float, doc="exact true magnitude")
        self.rng = numpy.random.RandomState(self.config.seed)

    def run(self, tractInfo, catalog=None):
        """Add records to the truth catalog and return it, delegating to makePositions and defineObject.

        If the given catalog is not None, add records to this catalog and return it instead
        of creating a new one.

        Subclasses should generally not need to override this method.
        """
        if catalog is None:
            catalog = lsst.afw.table.SimpleCatalog(self.schema)
        else:
            if not catalog.getSchema().contains(self.schema):
                raise ValueError("Catalog schema does not match Task schema")
        for coord, center in self.makePositions(tractInfo):
            record = catalog.addNew()
            record.setCoord(coord)
            record.set(self.center, center)
            self.defineObject(record)
        return catalog

    def makePositions(self, tractInfo):
        """Generate the centers (as a (coord, point) tuple) of mock objects (the point returned is
        in the tract coordinate system).

        Default implementation puts objects on a grid that is square in the tract's image coordinate
        system, with spacing approximately given by config.spacings.

        The return value is a Python iterable over (coord, point) pairs; the default implementation
        is actually an iterator (i.e. the function is a "generator"), but derived-class overrides may
        return any iterable.
        """
        wcs = tractInfo.getWcs()
        spacing = self.config.spacing / wcs.pixelScale().asArcseconds() # get spacing in tract pixels
        bbox = tractInfo.getBBox()
        for y in numpy.arange(bbox.getMinY() + 0.5 * spacing, bbox.getMaxY(), spacing):
            for x in numpy.arange(bbox.getMinX() + 0.5 * spacing, bbox.getMaxX(), spacing):
                yield wcs.pixelToSky(x, y), lsst.afw.geom.Point2D(x, y),

    def defineObject(self, record):
        """Fill in additional fields in a truth catalog record (id and coord will already have
        been set).
        """
        mag = self.rng.rand() * (self.config.maxMag - self.config.minMag) + self.config.minMag
        record.setD(self.magKey, mag)

    def drawSource(self, record, exposure, buffer=0):
        """Draw the given truth catalog record on the given exposure, makings use of its Psf, Wcs,
        Calib, and possibly filter to transform it appropriately.

        The mask and variance planes of the Exposure are ignored.

        The 'buffer' parameter is used to expand the source's bounding box when testing whether it
        is considered fully part of the image.

        Returns 0 if the object does not appear on the given image at all, 1 if it appears partially,
        and 2 if it appears fully (including the given buffer).
        """
        wcs = exposure.getWcs()
        center = wcs.skyToPixel(record.getCoord())
        try:
            psfImage = exposure.getPsf().computeImage(center).convertF()
        except:
            return 0
        psfBBox = psfImage.getBBox()
        overlap = exposure.getBBox()
        overlap.clip(psfBBox)
        if overlap.isEmpty():
            return 0
        flux = exposure.getCalib().getFlux(record.getD(self.magKey))
        normalization = flux / psfImage.getArray().sum()
        if psfBBox != overlap:
            psfImage = psfImage.Factory(psfImage, overlap)
            result = 1
        else:
            result = 2
            if buffer != 0:
                bufferedBBox = lsst.afw.geom.Box2I(psfBBox)
                bufferedBBox.grow(buffer)
                bufferedOverlap = exposure.getBBox()
                bufferedOverlap.clip(bufferedBBox)
                if bufferedOverlap != bufferedBBox:
                    result = 1
        image = exposure.getMaskedImage().getImage()
        subImage = image.Factory(image, overlap)
        subImage.scaledPlus(normalization, psfImage)
        variance = exposure.getMaskedImage().getVariance()
        subVariance = variance.Factory(variance, overlap)
        subVariance.scaledPlus(normalization, psfImage)
        return result
