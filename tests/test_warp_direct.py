# This file is part of pipe_tasks.
#
# LSST Data Management System
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
# See COPYRIGHT file at the top of the source tree.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#
"""Unit tests for the shared direct-warp capability gate
(``lsst.pipe.tasks._warpDirect``).

The gate is the single decision point every "rust" direct-mode consumer
(meta-mosaic TAN pairs, rgb2hips HPX tiles) shares, so it is pinned here
directly on the shared module: the pure-FITS-TAN verdict table (isFits
short-circuit, exact SIP-card rejection ahead of the numeric probe,
fail-closed on unreadable accessors) and the HPX builder accept/reject
table (plain HiPS headers accepted with the inert PV2_1=4/PV2_2=3 cards
treated as absent; isFits-False, TAN, off-regime-dec, exotic-card —
non-numeric PV values, SIP coefficient and linear D cards — and
missing-CTYPE headers all declined for the maps ladder).
"""
import numpy as np
import pytest

from lsst.afw.geom import makeHpxWcs, makeSkyWcs
from lsst.daf.base import PropertyList
from lsst.geom import Box2I, Extent2I, Point2I, Point2D, SpherePoint, degrees
from stellaRGB.warp import DirectWarp

from lsst.pipe.tasks._warpDirect import (
    _is_sip_card,
    hpx_direct_wcs_from_afw,
    pure_fits_tan,
    tan_wcs_from_afw,
)

# An order-8 HiPS tile (nested id within the 12*Nside^2 range) far enough
# from the poles to exercise the equatorial zone; makeHpxWcs(8, id, 12) is
# exactly the header shape rgb2hips warps onto.
HPX_ID = 569328


def _tan_wcs():
    return makeSkyWcs(
        crpix=Point2D(0.0, 0.0),
        crval=SpherePoint(10.0 * degrees, -30.0 * degrees),
        cdMatrix=np.array([[-1.0e-4, 0.0], [0.0, 1.0e-4]]),
    )


def _fits_stub(real, extra=None, drop=(), ctype=None, sky=None):
    """Delegate a SkyWcs stub's whole numeric surface to ``real`` while
    replacing selected getFitsMetadata() cards (and optionally the reported
    sky origin) — the same stub idiom the meta-mosaic battery uses to pin
    card-based rejection ahead of any sampling."""
    extra = extra or {}

    class _Stub:
        def __getattr__(self, name):
            if name == "getSkyOrigin" and sky is not None:
                return lambda: sky
            return getattr(real, name)

        def getFitsMetadata(self):
            ps = PropertyList()
            for name in real.getFitsMetadata().names():
                if name not in drop:
                    ps.set(name, real.getFitsMetadata().get(name))
            if ctype is not None:
                ps.set("CTYPE1", ctype[0])
                ps.set("CTYPE2", ctype[1])
            for name, value in extra.items():
                ps.set(name, value)
            return ps

    return _Stub()


class TestIsSipCard:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("A_ORDER", True), ("B_ORDER", True), ("C_ORDER", True), ("D_ORDER", True),
            ("A_2_0", True), ("B_0_1", True), ("C_1_1", True), ("D_3_2", True),
            # Linear-distortion cards of the SIP standard: not index-shaped
            # (the split path's digit test misses them).
            ("A_D1", True), ("A_D2", True), ("B_D1", True), ("B_D2", True),
            ("A_1", False),  # no second index: not an indexed coefficient card
            ("AB_1_1", False),
            ("CTYPE1", False), ("PV2_1", False), ("CRVAL1", False),
            ("A_P_Q", False),  # literal template names are not digits
            ("A_P1", False), ("AP1", False), ("D1", False),  # D-card lookalikes
        ],
        ids=["A_ORDER", "B_ORDER", "C_ORDER", "D_ORDER", "A_2_0", "B_0_1", "C_1_1",
             "D_3_2", "A_D1", "A_D2", "B_D1", "B_D2", "A_1", "AB_1_1", "CTYPE1",
             "PV2_1", "CRVAL1", "A_P_Q", "A_P1", "AP1", "D1"],
    )
    def test_table(self, name, expected):
        assert _is_sip_card(name) is expected


class TestPureFitsTan:
    def test_accepts_clean_tan(self):
        box = Box2I(Point2I(0, 0), Extent2I(64, 64))
        assert pure_fits_tan(_tan_wcs(), box) is True

    def test_isfits_short_circuits(self):
        """isFits False vetoes before any accessor is touched (the stub's
        would-be accessors raise if the short-circuit fails)."""
        class _NonFits:
            isFits = False

            def getPixelOrigin(self):
                raise AssertionError("isFits gate must short-circuit")

        assert pure_fits_tan(_NonFits(), Box2I(Point2I(0, 0), Extent2I(64, 64))) is False

    def test_sip_card_rejection_beats_clean_sampling(self):
        """A real A_p_q card rejects even though the delegated numeric TAN
        surface would pass the 9-point probe to machine precision; the same
        stub without the card passes — the card check does the rejecting."""
        real = _tan_wcs()
        box = Box2I(Point2I(0, 0), Extent2I(64, 64))
        poisoned = _fits_stub(real, extra={"A_ORDER": 2, "A_2_0": 1.0e-12})
        assert pure_fits_tan(poisoned, box) is False
        clean = _fits_stub(real)
        assert pure_fits_tan(clean, box) is True

    @pytest.mark.parametrize("fault", ["cd_nan", "sky_nan"],
                             ids=["cd_nan", "sky_nan"])
    def test_nonfinite_origin_triple_rejected(self, fault):
        real = _tan_wcs()
        box = Box2I(Point2I(0, 0), Extent2I(64, 64))
        if fault == "cd_nan":
            class _NanCd:
                def __getattr__(self, name):
                    return getattr(real, name)

                def getCdMatrix(self):
                    return np.array([[np.nan, 0.0], [0.0, 1.0e-4]])

            assert pure_fits_tan(_NanCd(), box) is False
        else:
            stub = _fits_stub(real, sky=SpherePoint(10.0 * degrees, float("nan") * degrees))
            # getSkyOrigin delegates through the stub override:
            assert pure_fits_tan(stub, box) is False

    def test_legacy_skymap_wcs_rejected(self):
        """The raw DiscreteSkyMap fixture WCS (legacy RA---STG frame dict)
        is provably not the TAN of its own triple and must be declined —
        the pin behind "raw-fixture rust batteries exercise the maps
        path" in both consumer suites."""
        import lsst.skymap

        config = lsst.skymap.DiscreteSkyMap.ConfigClass()
        tb = config.tractBuilder["cells"]
        tb.cellInnerDimensions = [100, 100]
        tb.cellBorder = 20
        tb.numCellsPerPatchInner = 4
        tb.numCellsInPatchBorder = 1
        config.tractBuilder.name = "cells"
        config.raList = [10.0, 10.05]
        config.decList = [-30.0, -30.0]
        config.radiusList = [0.15, 0.15]
        skymap = lsst.skymap.DiscreteSkyMap(config)
        info = skymap[0][0]
        assert pure_fits_tan(info.getWcs(), info.getOuterBBox()) is False


class TestTanWcsFromAfw:
    def test_accessor_mapping(self):
        wcs = makeSkyWcs(
            crpix=Point2D(20.0, 30.0),
            crval=SpherePoint(150.0 * degrees, -30.0 * degrees),
            cdMatrix=np.array([[-1.0e-3, 0.0], [0.0, 1.0e-3]]),
        )
        tan = tan_wcs_from_afw(wcs)
        assert tuple(tan.crpix) == pytest.approx((20.0, 30.0))
        assert tuple(tan.crval_deg) == pytest.approx((150.0, -30.0))
        np.testing.assert_allclose(tan.cd_deg_per_pix, [[-1.0e-3, 0.0], [0.0, 1.0e-3]])


class TestHpxDirectWcsFromAfw:
    """Accept/reject table for the HiPS header builder. The accept row also
    pins that the engine itself accepts the built DirectWcs (construction
    of a DirectWarp is the engine-side validation)."""

    @staticmethod
    def _hpx_wcs():
        return makeHpxWcs(8, HPX_ID, 12, False)

    def test_accepts_make_hpx_wcs_and_engine_validates(self):
        wcs = self._hpx_wcs()
        # Precondition (measured): the production header really carries the
        # inert PV2_1=4/PV2_2=3 pair — the case the builder must treat as
        # absent rather than feed to the engine's empty-pv contract.
        md = wcs.getFitsMetadata()
        assert md.get("PV2_1") == 4.0 and md.get("PV2_2") == 3.0
        built = hpx_direct_wcs_from_afw(wcs)
        assert built is not None and built.kind == "hpx"
        assert built.pv == ()
        # dec snapped to exactly 0.0 (afw answers ~1e-15 deg for a
        # header CRVAL2 of exactly 0; the engine adapter rejects anything
        # not exactly 0).
        assert built.crval_deg[1] == 0.0
        origin = wcs.getPixelOrigin()
        assert built.crpix == pytest.approx((origin.getX(), origin.getY()))
        np.testing.assert_allclose(built.cd_deg_per_pix, np.asarray(wcs.getCdMatrix()))
        # Engine-side validation: the built spec constructs a DirectWarp
        # against a TAN source (the rgb2hips production pairing).
        DirectWarp(dest=built, src=tan_wcs_from_afw(_tan_wcs()))

    def test_accepts_lowercase_ctype(self):
        real = self._hpx_wcs()
        stub = _fits_stub(real, ctype=("ra---hpx", "dec--hpx"))
        assert hpx_direct_wcs_from_afw(stub) is not None

    def test_isfits_short_circuits(self):
        """First tier mirroring pure_fits_tan: an isFits-False WCS is
        declined even though the same underlying header (delegated whole)
        is accepted — only the representability flag differs, so the card
        scan demonstrably never runs on a non-FITS representation."""
        real = self._hpx_wcs()
        assert hpx_direct_wcs_from_afw(real) is not None  # baseline

        class _NonFits:
            isFits = False

            def __getattr__(self, name):
                return getattr(real, name)

        assert hpx_direct_wcs_from_afw(_NonFits()) is None

    def test_rejects_tan(self):
        assert hpx_direct_wcs_from_afw(_tan_wcs()) is None

    def test_rejects_missing_ctype(self):
        real = self._hpx_wcs()
        stub = _fits_stub(real, drop=("CTYPE1", "CTYPE2"))
        assert hpx_direct_wcs_from_afw(stub) is None

    def test_rejects_mixed_ctype(self):
        real = self._hpx_wcs()
        stub = _fits_stub(real, ctype=("RA---HPX", "DEC--TAN"))
        assert hpx_direct_wcs_from_afw(stub) is None

    def test_rejects_nonzero_dec(self):
        real = self._hpx_wcs()
        stub = _fits_stub(real, sky=SpherePoint(real.getSkyOrigin().getRa(), 0.5 * degrees))
        assert hpx_direct_wcs_from_afw(stub) is None

    def test_snaps_noise_dec_to_zero(self):
        """|dec| within construction noise is a CRVAL2=0 header and is
        accepted with the declination snapped to exactly 0.0 (the value the
        engine adapter demands)."""
        real = self._hpx_wcs()
        ra = real.getSkyOrigin().getRa()
        stub = _fits_stub(real, sky=SpherePoint(ra, 1.0e-12 * degrees))
        built = hpx_direct_wcs_from_afw(stub)
        assert built is not None
        assert built.crval_deg[1] == 0.0

    @pytest.mark.parametrize(
        "extra",
        [
            {"PV2_1": 5.0},  # non-default H: inexpressible zone scheme
            {"PV2_2": 2.0},  # non-default K
            {"PV1_1": 1.0},  # PV-family card outside the inert pair
            {"PV2_1": "4"},  # string card: '4' is not numeric evidence
            {"PV2_1": True},  # bool: int subclass, never a zone constant
            {"A_ORDER": 2, "A_2_0": 1.0e-12},  # SIP on an HPX header
            {"A_D1": 0.01},  # SIP linear-distortion card on an HPX header
        ],
        ids=["PV2_1_5", "PV2_2_2", "PV1_1", "PV2_1_str", "PV2_1_bool", "SIP", "A_D1"],
    )
    def test_rejects_exotic_cards(self, extra):
        real = self._hpx_wcs()
        stub = _fits_stub(real, extra=extra)
        assert hpx_direct_wcs_from_afw(stub) is None

    def test_rejects_broken_accessor(self):
        """Fail closed (D19): an unreadable origin triple declines the WCS
        rather than letting the exception escape into the task."""
        real = self._hpx_wcs()

        class _Broken:
            def __getattr__(self, name):
                return getattr(real, name)

            def getPixelOrigin(self):
                raise RuntimeError("afw exploded")

        assert hpx_direct_wcs_from_afw(_Broken()) is None


class TestModuleSurface:
    def test_meta_mosaic_resolves_shared_symbols(self):
        """The meta-mosaic battery imports the gate through its historical
        module — the refactor to the shared module must leave those names
        resolving to the shared implementations."""
        from lsst.pipe.tasks.prettyPictureMaker import _meta_mosaic

        assert _meta_mosaic._pure_fits_tan is pure_fits_tan
        assert _meta_mosaic._tan_wcs_from_afw is tan_wcs_from_afw
