# This file is part of pipe_tasks.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import re
import os.path
import logging
import lsst.daf.base as dafBase
import lsst.geom as geom
import lsst.afw.table as afwTable
from lsst.daf.butler.formatters.file import FileFormatter


class ObjectMaskCatalog:
    """Class to support bright object masks
    """

    def __init__(self):
        schema = afwTable.SimpleTable.makeMinimalSchema()
        schema.addField("type", str, "type of region (e.g. box, circle)", size=10)
        schema.addField("radius", "Angle", "radius of mask (if type == circle")
        schema.addField("height", "Angle", "height of mask (if type == box)")
        schema.addField("width", "Angle", "width of mask (if type == box)")
        schema.addField("angle", "Angle", "rotation of mask (if type == box)")
        schema.addField("mag", float, "object's magnitude")

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
    def read(fileName):
        """Read a ds9 region file, returning a ObjectMaskCatalog object

        The files should be structured as follows:

        # Description of catalogue as a comment
        # CATALOG: catalog-id-string
        # TRACT: 0
        # PATCH: 5,4
        # FILTER: HSC-I

        wcs; fk5

        circle(RA, DEC, RADIUS)           # ID: 1, mag: 12.34
        box(RA, DEC, XSIZE, YSIZE, THETA) # ID: 2, mag: 23.45
        ...

        The ", mag: XX.YY" is optional

        The commented lines must be present, with the relevant fields such as
        tract patch and filter filled in. The coordinate system must be listed
        as above. Each patch is specified as a box or circle, with RA, DEC,
        and dimensions specified in decimal degrees (with or without an
        explicit "d").

        Only (axis-aligned) boxes and circles are currently supported as
        region definitions.
        """

        log = logging.getLogger("lsst.ObjectMaskCatalog")

        brightObjects = ObjectMaskCatalog()
        checkedWcsIsFk5 = False
        NaN = float("NaN")*geom.degrees

        nFormatError = 0                      # number of format errors seen
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
                mat = re.search(r"^\s*(box|circle)"
                                r"(?:\s+|\s*\(\s*)"   # open paren or space
                                r"([+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?)([d]*)"  # ra + units
                                r"(?:\s+|\s*,\s*)"  # sep
                                r"([+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?)([d]*)"  # dec + units
                                r"(?:\s+|\s*,\s*)"  # sep
                                r"([+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?)([d]*)"  # param1 + units
                                r"(?:"  # start optional 1
                                r"(?:\s+|\s*,\s*)"  # sep
                                r"([+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?)([d]*)"  # param2 + units
                                r"(?:"  # start optional 2
                                r"(?:\s+|\s*,\s*)"  # sep
                                r"([+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?)([d]*)"  # param3 + units
                                ")?"  # end optional 2
                                ")?"  # end optional 1
                                r"(?:\s*|\s*\)\s*)"   # close paren or space
                                r"#\s*ID:[\w\s]*(\d+)"  # start comment, ID
                                r"(?:\s*,?\s*mag:\s*([+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?))?"
                                r"\s*$", line)
                if mat:
                    _type, ra, raUnit, dec, decUnit, \
                        param1, param1Unit, param2, param2Unit, param3, param3Unit, \
                        _id, mag = mat.groups()

                    _id = int(_id)
                    if mag is None:
                        mag = NaN
                    else:
                        mag = float(mag)

                    ra = convertToAngle(ra, raUnit, "ra", fileName, lineNo)
                    dec = convertToAngle(dec, decUnit, "dec", fileName, lineNo)

                    radius = NaN
                    width = NaN
                    height = NaN
                    angle = 0.0*geom.degrees

                    if _type == "box":
                        width = convertToAngle(param1, param1Unit, "width", fileName, lineNo)
                        height = convertToAngle(param2, param2Unit, "height", fileName, lineNo)
                        if param3 is not None:
                            angle = convertToAngle(param3, param3Unit, "angle", fileName, lineNo)

                        if angle != 0.0:
                            log.warning("Rotated boxes are not supported: \"%s\" at %s:%d",
                                        line, fileName, lineNo)
                            nFormatError += 1
                    elif _type == "circle":
                        radius = convertToAngle(param1, param1Unit, "radius", fileName, lineNo)

                        if not (param2 is None and param3 is None):
                            log.warning("Extra parameters for circle: \"%s\" at %s:%d",
                                        line, fileName, lineNo)
                            nFormatError += 1

                    rec = brightObjects.addNew()
                    # N.b. rec["coord"] = Coord is not supported, so we have to use the setter
                    rec["type"] = _type
                    rec["id"] = _id
                    rec["mag"] = mag
                    rec.setCoord(geom.SpherePoint(ra, dec))

                    rec["angle"] = angle
                    rec["height"] = height
                    rec["width"] = width
                    rec["radius"] = radius
                else:
                    log.warning("Unexpected line \"%s\" at %s:%d", line, fileName, lineNo)
                    nFormatError += 1

        if nFormatError > 0:
            raise RuntimeError("Saw %d formatting errors in %s" % (nFormatError, fileName))

        if not checkedWcsIsFk5:
            raise RuntimeError("Expected to see a line specifying an fk5 wcs in %s" % fileName)

        # This makes the deep copy contiguous in memory so that a ColumnView can be exposed to Numpy
        brightObjects._catalog = brightObjects._catalog.copy(True)

        return brightObjects


def convertToAngle(var, varUnit, what, fileName, lineNo):
    """Given a variable and its units, return an geom.Angle

    what, fileName, and lineNo are used to generate helpful error messages
    """
    var = float(var)

    if varUnit in ("d", "", None):
        pass
    elif varUnit == "'":
        var /= 60.0
    elif varUnit == '"':
        var /= 3600.0
    else:
        raise RuntimeError("unsupported unit \"%s\" for %s at %s:%d" %
                           (varUnit, what, fileName, lineNo))

    return var*geom.degrees


class RegionFileFormatter(FileFormatter):
    """Plugin for reading DS9 region file catalogs with Gen3 Butler.
    """
    extension = ".reg"

    def _readFile(self, path, pytype):
        # Docstring inherited from FileFormatter._readFile
        if not os.path.exists(path):
            return None

        return pytype.read(path)

    def _writeFile(self, inMemoryDataset, fileDescriptor):
        # Docstring inherited from FileFormatter._writeFile
        raise NotImplementedError("Write not implemented.")
