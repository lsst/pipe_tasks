import numpy
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.coord as afwCoord

__all__ = ["SkyPolygon", "ImagePolygon"]

class Polygon(object):
    """Base class for a polygon, which is specified by a list of vertices"""
    def __init__(self, vertices):
        """Constructor

        @param vertices: List of polygon vertices, in order
        """
        self._vertices = vertices

    def edges(self):
        """Generator for polygon edges

        @return Yields vertices of edges
        """
        for v1, v2 in zip(self._vertices, self._vertices[1:] + [self._vertices[0]]):
            yield v1, v2

    def __iter__(self):
        """Iterator over polygon vertices"""
        return iter(self._vertices)

    def __len__(self):
        """Number of polygon vertices"""
        return len(self._vertices)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self._vertices)


class SkyPolygon(Polygon):
    def calculateCenter(self):
        """Calculate center of polygon

        @return ICRS coordinates of polygon center
        """
        xyz = afwGeom.Extent3D(0,0,0)
        for coord in self:
            xyz += afwGeom.Extent3D(coord.toIcrs().getVector())
        xyz /= len(self)
        return afwCoord.IcrsCoord(afwGeom.Point3D(xyz))

    def toImage(self, wcs):
        pointList = [wcs.skyToPixel(coord) for coord in self]
        return ImagePolygon(pointList)

    def calculateWcs(self, scale, zero=afwGeom.Point2D(0,0)):
        """Generate a Wcs for the polygon

        The Wcs is zeroed at the center of the polygon.

        @param scale: Pixel scale (Angle)
        @param zero: Zero position, pixels (CRPIX)
        @return Wcs for polygon
        """
        return afwImage.makeWcs(self.calculateCenter(), zero, scale.asDegrees(), 0.0, 0.0, scale.asDegrees())


class ImagePolygon(Polygon):
    def __init__(self, *args, **kwargs):
        super(ImagePolygon, self).__init__(*args, **kwargs)
        self._center = None

    def calculateCenter(self):
        """Calculate center of polygon"""
        center = afwGeom.Extent2D(0,0) # Center of image in patch x,y coordinates
        for pixel in self:
            center += afwGeom.Extent2D(pixel)
        center /= len(self)
        return center

    @property
    def center(self):
        """Return center of polygon"""
        if self._center is None:
            self._center = self.calculateCenter()
        return self._center

    def contains(self, point):
        """Determine whether a point is contained by the polygon

        To do this, we cast a ray and count the number of intersections.
        If the number of intersections is odd, then the point is
        enclosed.

        See:
        * http://en.wikipedia.org/wiki/Point_in_polygon
        * http://erich.realtimerendering.com/ptinpoly/

        @param point: Point of interest, (x,y) tuple
        @return Whether the point is enclosed (boolean)
        """
        x0, y0 = point
        numIntersections = 0
        for p1, p2 in self.edges():
            # Put point of interest at center
            x1, y1 = p1[0] - x0, p1[1] - y0
            x2, y2 = p2[0] - x0, p2[1] - y0
            # Cast ray right (along y=0)
            # Note that we test equality only for point 1, so that we only count a point on y=0 once.
            if (x1 <= 0 and x2 < 0) or (y1 <= 0 and y2 < 0) or (y1 >= 0 and y2 > 0): # Not intersected
                continue
            if x1 >= 0 and x2 > 0: # Must intersect somewhere
                numIntersections += 1
                continue
            if x1 - y1*(x1 - x2)/(y1 - y2) >= 0: # Intersects
                numIntersections += 1
        return True if numIntersections % 2 else False

    def _intersects(self, other):
        """Determine whether any edges of two polygons intersect

        @param other: Polygon to test for intersecting edges
        @return whether any of the edges of the polygons intersect (boolean)
        """
        coords = numpy.array([(p1.getX(), p1.getY(), p2.getX(), p2.getY()) for p1, p2 in self.edges()])
        p1x = coords[:,0]
        p1y = coords[:,1]
        p2x = coords[:,2]
        p2y = coords[:,3]
        pdx = p2x - p1x
        pdy = p2y - p1y
        for q1, q2 in other.edges():
            q1x = q1.getX()
            q1y = q1.getY()
            q2x = q2.getX()
            q2y = q2.getY()
            qdx = q2x - q1x
            qdy = q2y - q1y

            # Line 1: y = m1(x-x1) + y1
            # Line 2: y = m2(x-x2) + y2
            # Intersection: x = (m1*x1 - m2*x2 - y1 + y2)/(m1 - m2)
            mq = qdy/qdx if qdx != 0 else numpy.inf # Slope of q
            mp = numpy.where(pdx == 0, numpy.inf, pdy/pdx) # Slope of p
            x = (mp*p1x - mq*q1x - p1y + q1y)/(mp - mq) # x coordinate of intersection
            # Intersection x must be between q1x and q2x AND between p1x and p2x
            # If either slope is infinity, then dx=0, so test for intersection in the appropriate x,y range
            if (numpy.any((x < max(q1x, q2x)) & (x > min(q1x, q2x)) &
                         ((x > p1x) | (x > p2x)) & ((x < p1x) | (x < p2x))) or
                numpy.any(numpy.isinf(mp) & (p1x < max(q1x, q2x)) & (p1x > min(q1x, q2x)) &
                          ((p1y < max(q1y, q2y)) | (p2y < max(q1y, q2y))) &
                          ((p1y > min(q1y, q2y)) | (p2y > min(q1y, q2y)))) or
                numpy.any(numpy.isinf(mq) & ((q1x > p1x) | (q1x > p2x)) & ((q1x < p1x) | (q1x < p2x)) &
                          ((p1y < max(q1y, q2y)) | (p2y < max(q1y, q2y))) &
                          ((p1y > min(q1y, q2y)) | (p2y > min(q1y, q2y))))
                ):
                return True
        return False

    def intersects(self, other):
        """Determine whether any edges of two polygons intersect

        Disables warnings from numpy, because we're expecting there
        may be NANs and Infs (from dx=0).

        @param other: Polygon to test for intersecting edges
        @return whether any of the edges of the polygons intersect (boolean)
        """
        old = numpy.seterr(all="ignore")
        try:
            return self._intersects(other)
        finally:
            numpy.seterr(**old)

    def overlaps(self, other):
        """Determine whether two polygons overlap

        Polygons overlap if any of:
        1. The center of polygon 1 lies in polygon 2
        2. The center of polygon 2 lines in polygon 1
        3. Any of the edges of polygon 1 intersect any of the edges in polygon 2
        """
        return self.contains(other.center) or other.contains(self.center) or self.intersects(other)


