#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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
import math
import os
import sys

import lsst.afw.coord as afwCoord
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.skymap
import lsst.pipe.base as pipeBase
import idListOptions

class CoaddOptionParser(pipeBase.ArgumentParser):
    """OptionParser is an lsst.pipe.base.ArgumentParser specialized for coaddition.
    
    @warning this class contains camera-specific defaults for plate scale and tile overlap;
        additional camera support requires additional coding
    
    @todo:
    - Add defaults for:
      - HSC
      - CFHT Megacam
    - Set correct default scale for suprimecam
    
    """
    # default plate scale for coadds (arcsec/pixel); recommended value is camera plate scale/sqrt(2)
    _DefaultScale = dict(
        lsstSim = 0.14,
        suprimecam = 0.14,
    )

    # default overlap of sky tiles (deg); recommended value is one field width of the camera
    _DefaultOverlap = dict(
        lsstSim = 3.5,
        suprimecam = 1.5,
    )
    def __init__(self, usage="usage: %prog dataSource [options]", **kwargs):
        """Construct an option parser
        """
        pipeBase.ArgumentParser.__init__(self, usage=usage, **kwargs)

        self.add_argument("--fwhm", type=float, default=0.0,
            help="Desired FWHM, in science exposure pixels; for no PSF matching omit or set to 0")
        self.add_argument("--radec", nargs=2, type=float,
            help="RA Dec to find tileid; center of coadd unless llc specified (ICRS, degrees)")
        self.add_argument("--tileid", type=int,
            help="sky tile ID; if omitted, chooses the best sky tile for --radec")
        self.add_argument("--llc", nargs=2, type=int,
            help="x y index of lower left corner (pixels); if omitted, coadd is centered on --radec")
        self.add_argument("--size", nargs=2, type=int,
            help="x y size of coadd (pixels)")
        self.add_argument("--projection", default="STG",
            help="WCS projection used for sky tiles, e.g. STG or TAN")
        
    def _handleCamera(self, camera):
        """Set attributes based on camera
        
        Called by parse_args before the main parser is called
        """
        pipeBase.ArgumentParser._handleDataSource(self, camera)
        defaultScale = self._DefaultScale.get(camera)
        defaultOverlap = self._DefaultOverlap.get(camera)
        self.add_argument("--scale", type=float, default = defaultScale, required = (defaultScale == None),
            help="Pixel scale for skycell, in arcsec/pixel")
        self.add_argument("--overlap", type=float, default = defaultOverlap, required = (defaultScale==None),
            help="Overlap between adjacent sky tiles, in deg")

    def parse_args(self, config, argv=None):
        """Parse arguments for a command-line-driven task

        @params config: config for the task being run
        @params argv: argv to parse; if None then sys.argv[1:] is used
        @return namespace: an object containing an attribute for most command-line arguments and options.
            In addition to the standard attributes from pipeBase.ArgumentParser, adds the following
            coadd-specific attributes:
            - fwhm: Desired FWHM, in science exposure pixels; 0 for no PSF matching
            - radec: RA, Dec of center of coadd (an afwGeom.IcrsCoord)
            - coaddBBox: bounding box for coadd (an afwGeom.Box2I)
            - coaddWcs: WCS for coadd (an afwMath.Wcs)
            - skyMap: sky map for coadd (an lsst.skymap.SkyMap)
            - skyTileInfo: sky tile info for coadd (an lsst.skymap.SkyTileInfo)

            The following command-line options are NOT included in namespace:
            - llc (get from coaddBBox)
            - size (get from coaddBBox)
            - scale (get from skyTileInfo)
            - projection (get from skyTileInfo)
            - overlap (get from skyTileInfo)
            - tileid (get from skyTileInfo)
        """
        namespace = pipeBase.ArgumentParser.parse_args(self, config, argv)
        
        namespace.skyMap = lsst.skymap.SkyMap(
            projection = namespace.projection,
            pixelScale = afwGeom.Angle(namespace.scale, afwGeom.arcseconds),
            overlap = afwGeom.Angle(namespace.overlap, afwGeom.degrees),
        )
        del namespace.projection
        del namespace.scale
        del namespace.overlap

        if namespace.radec != None:
            radec = [afwGeom.Angle(ang, afwGeom.degrees) for ang in namespace.radec]
            namespace.radec = afwCoord.IcrsCoord(radec[0], radec[1])

        dimensions = afwGeom.Extent2I(namespace.size[0], namespace.size[1])
        
        tileId = namespace.tileid
        if tileId == None:
            if namespace.radec == None:
                raise RuntimeError("Must specify tileid or radec")
            tileId = namespace.skyMap.getSkyTileId(namespace.radec)

        namespace.skyTileInfo = namespace.skyMap.getSkyTileInfo(tileId)
        namespace.coaddWcs = namespace.skyTileInfo.getWcs()
        
        # determine bounding box
        if namespace.llc != None:
            llcPixInd = afwGeom.Point2I(namespace.llc[0], namespace.llc[1])
        else:
            if namespace.radec == None:
                raise RuntimeError("Must specify llc or radec")
            ctrPixPos = namespace.coaddWcs.skyToPixel(namespace.radec)
            ctrPixInd = afwGeom.Point2I(ctrPixPos)
            llcPixInd = ctrPixInd - (dimensions / 2)
        namespace.coaddBBox = afwGeom.Box2I(llcPixInd, dimensions)
        del namespace.llc
        del namespace.size
        if namespace.radec == None:
            namespace.radec = namespace.skyTileInfo.getCtrCoord()
        
        return namespace
