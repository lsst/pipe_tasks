import unittest
import numpy as np

import lsst.utils.tests as utilsTests
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.afw.geom as afwGeom
import lsst.afw.detection as afwDetect
from lsst.afw.geom import makeSkyWcs
from lsst.geom import Point2D, Angle, degrees, SpherePoint

from lsst.ip.diffim.maskReferenceSources import (
    MaskReferenceSourcesTask,
    MaskReferenceSourcesConfig,
)

# ----------------- helpers -----------------


def make_exposure(
    w=300, h=300, scale_arcsec=0.2, crpix=(150.0, 150.0), crval_deg=(10.0, -5.0)
):
    exp = afwImage.ExposureF(w, h)
    scale_deg = scale_arcsec / 3600.0
    cd = np.array([[scale_deg, 0.0], [0.0, scale_deg]], dtype=float)
    wcs = makeSkyWcs(
        crpix=Point2D(*crpix),
        crval=SpherePoint(Angle(crval_deg[0], degrees), Angle(crval_deg[1], degrees)),
        cdMatrix=cd,
    )
    exp.setWcs(wcs)
    return exp


def circular_footprint(xc, yc, r=4):
    spans = []
    for y in range(yc - r, yc + r + 1):
        dy = y - yc
        dx = int(np.floor(np.sqrt(max(0, r * r - dy * dy))))
        spans.append(afwGeom.Span(y, xc - dx, xc + dx))
    ss = afwGeom.SpanSet(spans)
    return afwDetect.Footprint(ss)


def make_src_cat(wcs, xy_list, radius=4):

    schema = afwTable.SourceTable.makeMinimalSchema()

    xKey = schema.addField(afwTable.FieldD("centroid_x", "centroid x", "pixel"))
    yKey = schema.addField(afwTable.FieldD("centroid_y", "centroid y", "pixel"))

    schema.getAliasMap().set("slot_Centroid", "centroid")
    table = afwTable.SourceTable.make(schema)

    cat = afwTable.SourceCatalog(table)
    for x, y in xy_list:
        rec = cat.addNew()
        rec.set(xKey, float(x))
        rec.set(yKey, float(y))
        rec.setCoord(wcs.pixelToSky(Point2D(float(x), float(y))))
        rec.setFootprint(circular_footprint(int(x), int(y), radius))
    return cat


def make_ref_cat_from_sources(wcs, src_xy, arcsec_offset, offset_axis="ra"):
    """Make a reference catalog by offsetting each source's sky coord."""
    schema = afwTable.SourceTable.makeMinimalSchema()
    table = afwTable.SourceTable.make(schema)
    cat = afwTable.SourceCatalog(table)
    deg_off = arcsec_offset / 3600.0
    for x, y in src_xy:
        sp = wcs.pixelToSky(Point2D(float(x), float(y)))
        ra = sp.getRa().asDegrees()
        dec = sp.getDec().asDegrees()
        if offset_axis == "ra":
            ra += deg_off
        else:
            dec += deg_off
        rec = cat.addNew()
        rec.setCoord(SpherePoint(Angle(ra, degrees), Angle(dec, degrees)))
    return cat


def mask_bit(exp, plane):
    try:
        return exp.mask.addMaskPlane(plane)
    except Exception:
        return exp.mask.getPlaneBitMask(plane)


def ensure_plane_bitmask(mask, name):
    try:
        bit_index = mask.addMaskPlane(name)
        return 1 << bit_index
    except Exception:
        return mask.getPlaneBitMask(name)


def paint_footprint(mask, footprint, bit):
    """OR the given bit into mask pixels inside the footprint (clip to bounds)."""
    bbox = mask.getBBox()
    for span in footprint.getSpans():
        y = span.getY()
        for x in range(span.getMinX(), span.getMaxX() + 1):
            if bbox.contains(x, y):
                mask.array[y, x] |= bit


# ----------------- tests -----------------


class TestMaskReferenceSourcesMany(utilsTests.TestCase):
    """Build 300x300 exposure with 8 DETECTED sources.
    Make 5 reference objects close enough to match
    Verify REFERENCE is set exactly over DETECTED for matches only.
    """

    def setUp(self):
        self.exp = make_exposure(300, 300, scale_arcsec=0.2)
        self.cfg = MaskReferenceSourcesConfig()
        self.cfg.mask_plane_name = "REFERENCE"
        self.cfg.matching_radius = 1.2  # arcsec
        self.task = MaskReferenceSourcesTask(config=self.cfg)

        # 8 sources, well-separated (no overlapping footprints)
        self.src_xy = [
            (40, 40),
            (90, 60),
            (140, 80),
            (190, 100),
            (240, 120),
            (60, 200),
            (160, 220),
            (260, 240),
        ]
        self.src_cat = make_src_cat(self.exp.getWcs(), self.src_xy, radius=4)

        self.detected_bit = ensure_plane_bitmask(self.exp.mask, "DETECTED")
        for rec in self.src_cat:
            paint_footprint(self.exp.mask, rec.getFootprint(), self.detected_bit)

        # Build reference catalog:
        # - First 5: +0.8" in RA (=> should match within 1.2")
        # - Last 3: not in the reference catalog
        refs = make_ref_cat_from_sources(
            self.exp.getWcs(), self.src_xy[:5], arcsec_offset=0.8, offset_axis="ra"
        )

        # Concatenate (order shouldn't matter)
        self.ref_cat = afwTable.SourceCatalog(refs.getTable())
        for rec in refs:
            self.ref_cat.append(rec)

        self.reference_bit = ensure_plane_bitmask(
            self.exp.mask, self.cfg.mask_plane_name
        )

    def test_masking(self):
        self.assertEqual(int((self.exp.mask.array & self.reference_bit).sum()), 0)

        # Run the algorithm
        result = self.task.run(
            ref_catalog=self.ref_cat,
            difference_image=self.exp,
            matching_sources=self.src_cat,
            matching_image=self.exp,
        )
        out = result.masked_difference_image
        mask_arr = out.mask.array

        # 1) REFERENCE must be a subset of DETECTED everywhere
        ref_only = (mask_arr & self.reference_bit) & ~(
            (mask_arr & self.detected_bit) > 0
        )
        self.assertEqual(
            int(ref_only.sum()), 0, "REFERENCE set outside DETECTED footprint(s)"
        )

        # 2) Exactly 5 sources matched; REFERENCE pixels == sum of their footprint areas
        matches = self.task._matchSources(self.ref_cat, self.src_cat, self.exp.getWcs())
        self.assertEqual(len(matches), 5, "Expected 5 matches within 1.2 arcsec")
        matched_src_ids = {id(src) for _, src in matches}

        expected_pixels = 0
        for src in self.src_cat:
            if id(src) in matched_src_ids:
                expected_pixels += src.getFootprint().getArea()

        actual_pixels = ((mask_arr & self.reference_bit) == self.reference_bit).sum()
        self.assertEqual(
            actual_pixels,
            expected_pixels,
            "REFERENCE mask pixels should equal sum of matched footprint areas",
        )

        # 3) Unmatched sources: DETECTED-only, no REFERENCE bits inside their footprints
        for src in self.src_cat:
            if id(src) not in matched_src_ids:
                # Collect REFERENCE pixels inside this footprint
                in_fp = 0
                bbox = self.exp.mask.getBBox()
                for span in src.getFootprint().getSpans():
                    y = span.getY()
                    for x in range(span.getMinX(), span.getMaxX() + 1):
                        if bbox.contains(x, y) and (
                            mask_arr[y, x] & self.reference_bit
                        ):
                            in_fp += 1
                self.assertEqual(
                    in_fp, 0, "Unmatched source has REFERENCE bits in its footprint"
                )


# --------------- boilerplate ---------------


def setup_module(module):
    utilsTests.init()


if __name__ == "__main__":
    utilsTests.init()
    unittest.main()
