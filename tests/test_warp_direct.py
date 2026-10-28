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
table (plain HiPS headers accepted through the raw-card AST probe with
the inert PV2_1=4/PV2_2=3 cards treated as absent; isFits-False, TAN,
off-regime-dec, exotic-card — non-numeric PV values, SIP coefficient and
linear D cards — and missing-CTYPE headers all declined for the maps
ladder before the engine is ever asked).
"""
import numpy as np
import pytest

from lsst.afw.geom import makeHpxWcs, makeSkyWcs
from lsst.daf.base import PropertyList
from lsst.geom import Box2I, Extent2I, Point2I, Point2D, SpherePoint, degrees
from stellaRGB.warp import AstWarp, AstWcs

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
        stub without the card passes — the card check does the rejecting.
        The poisoned set is the COMPLETE closed-form pair (A/B forward plus
        AP/BP inverse): the very header the stellaRGB gate selectively
        unlocks into a ``DirectWcs.sip`` layer since T8.4 — pipe policy
        still keeps ALL SIP distortion on the fitted-maps path."""
        real = _tan_wcs()
        box = Box2I(Point2I(0, 0), Extent2I(64, 64))
        complete_pair = {
            "A_ORDER": 2,
            "A_2_0": 1.0e-12,
            "B_ORDER": 2,
            "B_0_2": 1.0e-12,
            "AP_ORDER": 2,
            "AP_2_0": -1.0e-12,
            "BP_ORDER": 2,
            "BP_0_2": -1.0e-12,
        }
        poisoned = _fits_stub(real, extra=complete_pair)
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
    pins that the engine itself accepts the built frame (construction of an
    AstWarp is the engine-side validation; the HPX route is raw-card —
    T10.5)."""

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
        # Raw-card route (T10.5): the verdict is an AST-probed card frame,
        # not a classified spec — the cards are the afw header verbatim,
        # so CRPIX stays FITS 1-based (the 0-based getPixelOrigin accessor
        # and its isFits-quirk mismatch are no longer consulted at all).
        assert isinstance(built, AstWcs)
        assert any(card.startswith("CTYPE1  = 'RA---HPX'") for card in built.cards)
        assert any(card.startswith("CRPIX1  =") for card in built.cards)
        np.testing.assert_allclose(md.get("CRPIX2"), wcs.getPixelOrigin().getY() + 1.0)
        # Engine-side validation: the probed raw-card dest constructs an
        # AstWarp against a TAN source (the rgb2hips production pairing).
        warp = AstWarp(dest=built, src=tan_wcs_from_afw(_tan_wcs()))
        assert warp.mode == "direct"

    def test_get_fits_metadata_round_trips_through_the_probe(self):
        """The raw-card premise (T10.5 probe result): the production tile
        header passes the engine's AST probe directly — no astropy
        detour, no value re-basing — with every afw card rendered into the
        80-column image AST reads."""
        wcs = self._hpx_wcs()
        md = wcs.getFitsMetadata()
        built = AstWcs.from_cards(md)
        assert len(built.cards) == len(list(md.names()))
        assert all(len(card) == 80 for card in built.cards)

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
        # Card-based stub (T10.5 retarget): the raw-card route reads the
        # declination off the CRVAL2 card — the frame the engine renders —
        # so the off-regime value is poisoned there; the skyOrigin accessor
        # the classifier route consulted is no longer an input.
        real = self._hpx_wcs()
        stub = _fits_stub(real, extra={"CRVAL2": 0.5})
        assert hpx_direct_wcs_from_afw(stub) is None

    def test_accepts_construction_noise_dec(self):
        """A CRVAL2 within construction noise of the HiPS regime is
        accepted. Unlike the classifier route there is no snap to exactly
        0.0 (T10.5): the raw-card route hands the header verbatim, and
        1e-9 deg is far below render precision — production headers state
        CRVAL2 = 0.0 exactly (the accept row's header is makeHpxWcs)."""
        real = self._hpx_wcs()
        stub = _fits_stub(real, extra={"CRVAL2": 1.0e-12})
        assert hpx_direct_wcs_from_afw(stub) is not None

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
            # The complete closed-form inverse pair (T8.4's selective
            # unlock set): pipe still declines it — maps-only policy.
            {"AP_ORDER": 2, "AP_2_0": -1.0e-12, "BP_ORDER": 2, "BP_0_2": -1.0e-12},
        ],
        ids=["PV2_1_5", "PV2_2_2", "PV1_1", "PV2_1_str", "PV2_1_bool", "SIP", "A_D1", "AP_pair"],
    )
    def test_rejects_exotic_cards(self, extra):
        real = self._hpx_wcs()
        stub = _fits_stub(real, extra=extra)
        assert hpx_direct_wcs_from_afw(stub) is None

    def test_sip_veto_fires_before_the_probe(self):
        """Policy ordering (T10.5): the ALL-SIP veto runs BEFORE the AST
        probe — unlike the old classifier, the AST backend reads SIP from
        the cards themselves and would happily accept this header, so the
        pipe's veto must not merely outrank the probe's verdict but
        suppress the probe call entirely (spy records zero calls)."""
        real = self._hpx_wcs()
        stub = _fits_stub(real, extra={"A_ORDER": 2, "A_2_0": 1.0e-12})
        calls = []

        class _Spy:
            @classmethod
            def from_cards(cls, cards):
                calls.append(cards)
                raise AssertionError("SIP-poisoned header reached the probe")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr("lsst.pipe.tasks._warpDirect.AstWcs", _Spy)
        try:
            assert hpx_direct_wcs_from_afw(stub) is None
            assert calls == []
        finally:
            monkeypatch.undo()

    def test_probe_decline_fails_closed(self):
        """(T10.5) A header the regime screen cannot decline but the AST
        probe can: a zero CD matrix is finite (every card passes the
        screen) yet the frame cannot round trip — from_cards raises
        ValueError with AST's reason, and the fail-closed contract
        swallows it to None so the maps ladder serves the WCS (D19)."""
        real = self._hpx_wcs()
        stub = _fits_stub(
            real, extra={"CD1_1": 0.0, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": 0.0}
        )
        assert hpx_direct_wcs_from_afw(stub) is None

    def test_rejects_broken_accessor(self):
        """Fail closed (D19): an unreadable card surface declines the WCS
        rather than letting the exception escape into the task.
        (T10.5 retarget: the HPX route's input surface is
        getFitsMetadata() — the origin accessors it used to read are now
        only the TAN route's inputs.)"""
        real = self._hpx_wcs()

        class _Broken:
            def __getattr__(self, name):
                return getattr(real, name)

            def getFitsMetadata(self):
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
