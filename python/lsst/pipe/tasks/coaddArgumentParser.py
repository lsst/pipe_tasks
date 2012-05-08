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
import lsst.afw.coord as afwCoord
import lsst.afw.geom as afwGeom
import lsst.skymap
import lsst.pipe.base as pipeBase

class CoaddArgumentParser(pipeBase.ArgumentParser):
    """A version of lsst.pipe.base.ArgumentParser specialized for coaddition.
    
    @warning this class contains camera-specific defaults for plate scale and tract overlap;
        additional camera support requires additional coding.
        In the long run this camera-specific support will not be required: the sky maps
        will be defined in the data repositories and the associated arguments for constructing
        a sky map will go away.
    
    @todo:
    - Add defaults for:
      - sdss
      - hscSim
      - cfht (Megacam)
    - Set correct default scale for suprimecam
    
    """
    # default plate scale for coadds (arcsec/pixel); recommended value is camera plate scale/sqrt(2)
    _DefaultScale = dict(
        lsstSim = 0.14,
        suprimecam = 0.14,
    )

    # default overlap of sky tracts (deg); recommended value is one field width of the camera
    _DefaultOverlap = dict(
        lsstSim = 3.5,
        suprimecam = 1.5,
    )
    def __init__(self, **kwargs):
        """Construct an option parser
        """
        pipeBase.ArgumentParser.__init__(self, **kwargs)

        self.add_argument("--fwhm", type=float, default=0.0,
            help="Desired FWHM, in science exposure pixels; for no PSF matching omit or set to 0")
        self.add_argument("--radec", nargs=2, type=float,
            help="RA Dec to find tractId; center of coadd unless llc specified (ICRS, degrees)")
        self.add_argument("--tract", type=int,
            help="sky tract ID; if omitted, chooses the best sky tract for --radec")
        self.add_argument("--patch", nargs=2, type=int,
            help="sky patch index; if omitted, chooses the best sky patch for --radec")
        self.add_argument("--projection", default="STG",
            help="WCS projection used for sky tracts, e.g. STG or TAN")
        
    def handleCamera(self, namespace):
        """Set attributes based on camera
        
        Called by parse_args before the main parser is called
        """
        camera = namespace.camera
        pipeBase.ArgumentParser.handleCamera(self, namespace)
        defaultScale = self._DefaultScale.get(camera)
        defaultOverlap = self._DefaultOverlap.get(camera)
        self.add_argument("--scale", type=float, default = defaultScale, required = (defaultScale is None),
            help="Pixel scale for skycell, in arcsec/pixel")
        self.add_argument("--overlap", type=float, default = defaultOverlap, required = (defaultScale==None),
            help="Overlap between adjacent sky tracts, in deg")

    def parse_args(self, config, args=None, log=None):
        """Parse arguments for a command-line-driven task

        @params config: config for the task being run
        @params args: argument list; if None use sys.argv[1:]
        @params log: log (instance pex_logging Log); if None use the default log

        @return namespace: an object containing an attribute for most command-line arguments and options.
        In addition to the standard attributes from pipeBase.ArgumentParser, adds the following
        coadd-specific attributes:
        - fwhm: Desired FWHM, in science exposure pixels; 0 for no PSF matching
        - radec: User-specified center of coadd (an afwGeom.IcrsCoord); None if user did not specify
        - skyMap: sky map for coadd (an lsst.skymap.DodecaSkyMap)
        - tractInfo: sky tract info for coadd (an lsst.skymap.TractInfo)
        - patchInfo: sky patch info for coadd (an lsst.skyMap.PatchInfo)

        The following command-line options are NOT included in namespace:
        - scale (get from tractInfo)
        - projection (get from tractInfo)
        - overlap (get from tractInfo)
        - tract (get from tractInfo)
        - patch (get from patchInfo)
        """
        namespace = pipeBase.ArgumentParser.parse_args(self, config=config, args=args, log=log)
        
        namespace.skyMap = lsst.skymap.DodecaSkyMap(
            projection = namespace.projection,
            pixelScale = afwGeom.Angle(namespace.scale, afwGeom.arcseconds),
            tractOverlap = afwGeom.Angle(namespace.overlap, afwGeom.degrees),
        )

        if namespace.radec != None:
            radec = [afwGeom.Angle(ang, afwGeom.degrees) for ang in namespace.radec]
            namespace.radec = afwCoord.IcrsCoord(radec[0], radec[1])

        tractId = namespace.tract
        if tractId is None:
            if namespace.radec is None:
                raise RuntimeError("Must specify tract (and patch) or radec")
            tractInfo = namespace.skyMap.findTract(namespace.radec)
        else:
            tractInfo = namespace.skyMap[tractId]
        namespace.tractInfo = tractInfo
        
        patchIndex = namespace.patch
        if patchIndex is None:
            if namespace.radec is None:
                raise RuntimeError("Must specify patch (and tract) or radec")
            patchInfo = tractInfo.findPatch(namespace.radec)
        else:
            patchInfo = tractInfo.getPatch(patchIndex)
        namespace.patchInfo = patchInfo
        
        del namespace.scale
        del namespace.projection
        del namespace.overlap
        del namespace.tract
        del namespace.patch
        
        return namespace
