from builtins import object
import re
import lsst.daf.base as dafBase
import lsst.afw.coord as afwCoord
import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable
from lsst.log import Log


class ObjectMaskCatalog(object):
    """Class to support bright object masks

    N.b. I/O is done by providing a readFits method which fools the butler.
    """

    def __init__(self):
        schema = afwTable.SimpleTable.makeMinimalSchema()
        schema.addField("radius", "Angle", "radius of mask")

        self._catalog = afwTable.SimpleCatalog(schema)
        self._catalog.table.setMetadata(dafBase.PropertyList())

        self.table = self._catalog.table
        self.addNew = self._catalog.addNew

    def __len__(self):
        return len(self._catalog)

    def __iter__(self):
        return iter(self._catalog)

    def __getitem__(self, i):
        return self._catalog.__getitem__(i)

    def __setitem__(self, i, v):
        return self._catalog.__setitem__(i, v)

    @staticmethod
    def readFits(fileName, hdu=0, flags=0):
        """Read a ds9 region file, returning a ObjectMaskCatalog object

        This method is called "readFits" to fool the butler. The corresponding mapper entry looks like
        brightObjectMask: {
            template:      "deepCoadd/BrightObjectMasks/%(tract)d/BrightObjectMask-%(tract)d-%(patch)s-%(filter)s.reg"
            python:        "lsst.obs.subaru.objectMasks.ObjectMaskCatalog"
            persistable:   "PurePythonClass"
            storage:       "FitsCatalogStorage"
        }
        and this is the only way I know to get it to read a random file type, in this case a ds9 region file

        This method expects to find files named as BrightObjectMask-%(tract)d-%(patch)s-%(filter)s.reg
        The files should be structured as follows:

        # Description of catalogue as a comment
        # CATALOG: catalog-id-string
        # TRACT: 0
        # PATCH: 5,4
        # FILTER: HSC-I

        wcs; fk5

        circle(RA, DEC, RADIUS) # ID: 1

        The commented lines must be present, with the relevant fields such as tract patch and filter filled
        in. The coordinate system must be listed as above. Each patch is specified as a circle, with an RA,
        DEC, and Radius specified in decimal degrees. Only circles are supported as region definitions
        currently.
        """

        log = Log.getLogger("ObjectMaskCatalog")

        brightObjects = ObjectMaskCatalog()
        checkedWcsIsFk5 = False

        with open(fileName) as fd:
            for lineNo, line in enumerate(fd.readlines(), 1):
                line = line.rstrip()

                if re.search(r"^\s*#", line):
                    #
                    # Parse any line of the form "# key : value" and put them into the metadata.
                    #
                    # The medatdata values must be defined as outlined in the above docstring
                    #
                    # The value of these three keys will be checked,
                    # so get them right!
                    #
                    mat = re.search(r"^\s*#\s*([a-zA-Z][a-zA-Z0-9_]+)\s*:\s*(.*)", line)
                    if mat:
                        key, value = mat.group(1).lower(), mat.group(2)
                        if key == "tract":
                            value = int(value)

                        brightObjects.table.getMetadata().set(key, value)

                line = re.sub(r"^\s*#.*", "", line)
                if not line:
                    continue

                if re.search(r"^\s*wcs\s*;\s*fk5\s*$", line, re.IGNORECASE):
                    checkedWcsIsFk5 = True
                    continue

                # This regular expression parses the regions file for each region to be masked,
                # with the format as specified in the above docstring.
                mat = re.search(r"^\s*circle(?:\s+|\s*\(\s*)"
                                "(\d+(?:\.\d*)([d]*))" "(?:\s+|\s*,\s*)"
                                "([+-]?\d+(?:\.\d*)([d]*))" "(?:\s+|\s*,\s*)"
                                "(\d+(?:\.\d*))([d'\"]*)" "(?:\s*|\s*\)\s*)"
                                "\s*#\s*ID:\s*(\d+)" "\s*$", line)
                if mat:
                    ra, raUnit, dec, decUnit, radius, radiusUnit, _id = mat.groups()

                    _id = int(_id)
                    ra = convertToAngle(ra, raUnit, "ra", fileName, lineNo)
                    dec = convertToAngle(dec, decUnit, "dec", fileName, lineNo)
                    radius = convertToAngle(radius, radiusUnit, "radius", fileName, lineNo)

                    rec = brightObjects.addNew()
                    # N.b. rec["coord"] = Coord is not supported, so we have to use the setter
                    rec["id"] = _id
                    rec.setCoord(afwCoord.Fk5Coord(ra, dec))
                    rec["radius"] = radius
                else:
                    log.warn("Unexpected line \"%s\" at %s:%d" % (line, fileName, lineNo))

        if not checkedWcsIsFk5:
            raise RuntimeError("Expected to see a line specifying an fk5 wcs")

        # This makes the deep copy contiguous in memory so that a ColumnView can be exposed to Numpy
        brightObjects._catalog = brightObjects._catalog.copy(True)

        return brightObjects


def convertToAngle(var, varUnit, what, fileName, lineNo):
    """Given a variable and its units, return an afwGeom.Angle

    what, fileName, and lineNo are used to generate helpful error messages
    """
    var = float(var)

    if varUnit in ("d", ""):
        pass
    elif varUnit == "'":
        var /= 60.0
    elif varUnit == '"':
        var /= 3600.0
    else:
        raise RuntimeError("unsupported unit \"%s\" for %s at %s:%d" %
                           (varUnit, what, fileName, lineNo))

    return var*afwGeom.degrees
