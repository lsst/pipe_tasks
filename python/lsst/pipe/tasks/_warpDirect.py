# This file is part of pipe_tasks.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Shared capability gate for the ``stellaRGB.warp`` direct (per-pixel) engine.

The "rust" warp backends of :class:`~lsst.pipe.tasks.prettyPictureMaker.
MetaMosaicTask` and ``lsst.pipe.tasks.rgb2hips.HighOrderHipsTask`` render a
destination grid directly from two WCS *header triples* instead of fitting
coordinate maps — but only when each side's ``lsst.afw.geom.SkyWcs`` is
provably a plain FITS projection the engine's closed-form registry models.
This module owns the genuine afw glue of that decision (D16/D18/D19): the
pure-FITS-TAN verdict (:func:`pure_fits_tan`), the TAN header-to-``TanWcs``
builder (:func:`tan_wcs_from_afw`), and the HPX builder
(:func:`hpx_direct_wcs_from_afw`). The TAN route stays the builtin
closed-form triple (``crpix``/``crval``/``CD``, extracted from the afw
accessors); the HPX route is the raw-card route (T10.5): the tile WCS's own
``getFitsMetadata()`` card set handed verbatim to
:class:`stellaRGB.warp.AstWcs`, whose ``from_cards`` runs the engine's
starlink-ast probe. The FITS *card vocabulary* — the SIP-card scan (quartet
+ digit patterns) — lives in :mod:`stellaRGB.warp`
(:func:`~stellaRGB.warp.sip_cards_present`) so non-pipeline callers get the
same trap book for free; what stays HERE is pipeline risk defense, not FITS
knowledge: accessor extraction, the ``isFits``-quirk short-circuit, the
9-point ``makeSkyWcs`` agreement probe, the TAN-only policy and the
plain-HiPS-header screen :func:`_plain_hips_header` (the AST probe answers
only "AST can render these cards", so the pipe's own regime vetoes — exact
HPX CTYPE pair, inert PV pair only, declination on the HiPS regime — are
stated here, and they are what keeps headers the probe happily reads out of
the direct path). Every gate fails CLOSED: a WCS that cannot be fully read,
carries distortion cards, or sits outside a modeled regime is declined with
``False``/``None`` and the caller's fitted-maps ladder (or afw fallback)
serves it — a direct-mode engine hiccup must never crash a task and a direct
render must never be entered on a partially-verified transform.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from stellaRGB.warp import AstWcs, TanWcs, sip_cards_present

from lsst.afw.geom import makeSkyWcs
from lsst.geom import Point2D

if TYPE_CHECKING:
    from lsst.afw.geom import SkyWcs
    from lsst.geom import Box2I


def _is_sip_card(name: str) -> bool:
    """True for FITS SIP metadata card names. The vocabulary (order
    declarations, indexed coefficient cards, the A_D1/A_D2/B_D1/B_D2
    linear quartet and the digit patterns) is single-sourced in
    :func:`stellaRGB.warp.sip_cards_present` (T8.8 promotion); this
    per-name wrapper keeps the gate-local name — and the ``None``-value
    presence filtering stays at the call sites below.
    """
    return sip_cards_present({name: 0.0})


# The HPX plain-header regime restated pipe-side for the raw-card route
# (T10.5). Values mirror the classifier's HPX arm in
# :mod:`stellaRGB.warp` (``_HPX_INERT_PV`` / ``_HPX_CRVAL2_TOL_DEG``);
# the AST probe itself is deliberately policy-free — it answers only
# "AST can render these cards", happily reading non-inert PV values,
# non-numeric cards and even a missing CTYPE pair — so without this
# screen the pipe's regime policy would silently widen.
_HPX_CTYPE_PAIR = ("RA---HPX", "DEC--HPX")
"""The exact CTYPE1/CTYPE2 pair of a HiPS HPX tile header, matched after
``strip().upper()`` like the classifier matches it."""

_HPX_INERT_PV = {"PV2_1": 4.0, "PV2_2": 3.0}
"""The only PV cards the engine's fixed zone scheme models (H=4/K=3),
treated as absent — any other PV card means an inexpressible zone scheme
or non-numeric evidence and is declined."""

_HPX_CRVAL2_TOL_DEG = 1e-9
"""Construction-noise tolerance on the HiPS-regime declination
(CRVAL2 = 0); mirrors the classifier constant of the same name."""

_HPX_FRAME_CARDS = (
    "CRPIX1",
    "CRPIX2",
    "CRVAL1",
    "CRVAL2",
    "CD1_1",
    "CD1_2",
    "CD2_1",
    "CD2_2",
)
"""The frame cards the raw-card route consumes; each must be present and
carry a finite number."""


def _hips_numeric_card_value(cards, name):
    """The value of ``cards[name]`` when it is a real finite number.

    ``bool`` is excluded although it is an ``int`` subclass: a boolean
    card is never numeric evidence of a zone constant or frame value —
    this restates the classifier's ``_numeric_card_value`` rule (bool and
    ``np.bool_`` out, real numeric types in, non-finite out)."""
    try:
        value = cards.get(name)
    except Exception:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        return None
    out = float(value)
    return out if np.isfinite(out) else None


def _plain_hips_header(cards) -> bool:
    """True when a ``names()``/``get()`` card set (afw
    ``getFitsMetadata()``'s result) is a plain HiPS HPX header under the
    pipe's regime policy — the vetoes the old
    :func:`stellaRGB.warp.direct_spec_from_cards` call applied for free,
    restated for the raw-card route (T10.5): the exact HPX CTYPE pair,
    the inert H=4/K=3 ``PV2_1 = 4``/``PV2_2 = 3`` pair as the only PV
    cards (non-numeric or off-value PV declined), a finite frame card
    set, and declination on the HiPS regime within construction noise.

    Fail-closed by construction: any unreadable card or off-regime value
    answers ``False`` and the maps ladder serves the WCS. Declination is
    NOT snapped here — the raw-card route hands the header verbatim
    (``makeHpxWcs`` states ``CRVAL2 = 0`` exactly; within-tolerance
    noise is far below render precision and stays the header's own)."""
    try:
        names = list(cards.names())
        ctype = []
        for card in ("CTYPE1", "CTYPE2"):
            value = cards.get(card)
            if not isinstance(value, str):
                return False
            ctype.append(value.strip().upper())
        if tuple(ctype) != _HPX_CTYPE_PAIR:
            return False
        for name in names:
            if name.startswith("PV"):
                inert = _HPX_INERT_PV.get(name)
                number = _hips_numeric_card_value(cards, name)
                if inert is None or number is None or number != inert:
                    return False
        for card in _HPX_FRAME_CARDS:
            if _hips_numeric_card_value(cards, card) is None:
                return False
        return abs(_hips_numeric_card_value(cards, "CRVAL2")) <= _HPX_CRVAL2_TOL_DEG
    except Exception:
        return False


def pure_fits_tan(wcs: SkyWcs, box: Box2I) -> bool:
    """True when ``wcs`` is accepted as pure FITS TAN for direct mode.

    This is a gate of increasing cost, not a proof:

    1. ``isFits`` (a *property* in this stack, answering "FITS-metadata
       representable" — TAN+SIP also satisfies it, SIP keywords being
       FITS-standard metadata; ``hasFitsApproximation()`` is no use, it
       only reports a factory approximation slot) rejects exotic
       non-FITS representations outright.
    2. Exact SIP-card rejection on ``getFitsMetadata()``: ANY SIP-family
       card vetoes the WCS before any sampling — A/B/C/D/AP/BP order
       declarations, indexed A_p_q/AP_p_q-style coefficient cards, and
       the A_D1/A_D2/B_D1/B_D2 linear quartet (single-sourced from
       :func:`stellaRGB.warp.sip_cards_present`, whose vocabulary since
       T8.4 also covers the AP/BP closed-form inverse spelling). The
       order matters: a numeric probe can never PROVE the absence of
       polynomial distortion — an adversarial SIP polynomial with terms
       chosen to vanish at every sample point agrees to the probe's
       tolerance at all of them while reaching 0.05-0.14 px between the
       samples. The header cards are exact evidence and SIP files must
       carry them. NOTE (T8.4): this gate stays stricter than
       :func:`stellaRGB.warp.direct_spec_from_cards`, which since T8.4
       selectively accepts a complete A/B + AP/BP closed-form pair as a
       ``DirectWcs.sip`` layer. The pipeline policy keeps ALL SIP
       distortion on the fitted-maps path: this gate's contract is the
       pure-FITS-TAN verdict (the T7.3 bit-identical pin), and no
       pipe-side battery yet covers a distortion-carrying direct
       render. Widen condition: a pipe_tasks battery pinning
       SIP-direct renders against afw/drizzled truth on real SIP
       exposures — until then, do not relax this veto.
    3. The numeric heuristic last: re-derive the TAN-only WCS from the
        same three accessors the direct engine consumes (pixelOrigin,
        skyOrigin, CD) and require pixel->sky agreement at a 9-point
        lattice over ``box`` (corners, edge midpoints, center) to a
        microarcsecond. This catches non-pure frame-dict geometries that
        are not the plain TAN of their own reported triple (e.g. the
        legacy azimuthal-equidistant-like DiscreteSkyMap tract WCS,
        measured ~1.2 arcsec off gnomonic at the fixture bbox corners)
        and NaN-origin distortion variants. A WCS that passes the cards
        and the lattice is accepted as TAN for rendering purposes;
        residual risk of distortion hidden between samples exists in
        principle, mitigated by card 2 removing its realistic source.
    """
    if not wcs.isFits:
        return False
    try:
        fits_md = wcs.getFitsMetadata()
        if any(fits_md.get(name) is not None for name in fits_md.names() if _is_sip_card(name)):
            return False
        origin = wcs.getPixelOrigin()
        sky = wcs.getSkyOrigin()
        cd = np.asarray(wcs.getCdMatrix())
        # Distortion-carrying WCSs report isFits==True (SIP keywords are
        # FITS-standard metadata) yet answer NaN for the origin triple
        # (no pixel maps exactly to crval under SIP) — reject before
        # makeSkyWcs chokes on NaN CRPIX. Any failure here means the WCS
        # is not provably the plain TAN its accessors claim, so the
        # fitted-maps path (which drives the WCS's own frame dictionary)
        # must serve it: fail closed.
        if not (
            np.isfinite(origin.getX())
            and np.isfinite(origin.getY())
            and bool(np.isfinite(cd).all())
            and np.isfinite(sky.getRa().asRadians())
            and np.isfinite(sky.getDec().asRadians())
        ):
            return False
        tan = makeSkyWcs(crpix=origin, crval=sky, cdMatrix=cd)
        cx = (box.getMinX() + box.getMaxX()) / 2.0
        cy = (box.getMinY() + box.getMaxY()) / 2.0
        for x in (box.getMinX(), cx, box.getMaxX()):
            for y in (box.getMinY(), cy, box.getMaxY()):
                p = Point2D(x, y)
                # ``not (sep < tol)`` fails closed on NaN separations.
                if not wcs.pixelToSky(p).separation(tan.pixelToSky(p)).asArcseconds() < 1e-6:
                    return False
    except Exception:
        return False
    return True


def tan_wcs_from_afw(wcs: SkyWcs) -> TanWcs:
    """Map a (checked pure-FITS-TAN) afw SkyWcs onto the direct engine's
    :class:`stellaRGB.warp.TanWcs` field-by-field (D16): ``getPixelOrigin``
    is the 0-based pixel-center reference pixel (afw's pixel origin is
    FITS ``CRPIX - 1``), ``getSkyOrigin`` the tangent point (degrees), and
    ``getCdMatrix`` the degrees-per-pixel CD matrix."""
    origin = wcs.getPixelOrigin()
    sky = wcs.getSkyOrigin()
    return TanWcs(
        crpix=(origin.getX(), origin.getY()),
        crval_deg=(sky.getRa().asDegrees(), sky.getDec().asDegrees()),
        cd_deg_per_pix=np.asarray(wcs.getCdMatrix(), dtype=float),
    )


def hpx_direct_wcs_from_afw(wcs: SkyWcs) -> AstWcs | None:
    """Map an afw HiPS-tile SkyWcs onto the direct engine's raw-card frame
    (:class:`stellaRGB.warp.AstWcs`), or return ``None`` when the WCS is
    not a plain HiPS HPX header.

    Raw-card route (T10.5, replacing the T8.8 classifier route): the
    tile's own ``getFitsMetadata()`` card set is handed to
    :meth:`stellaRGB.warp.AstWcs.from_cards`, whose AST probe is the
    engine's own go/no-go (the same fail-closed read + round-trip gate
    :class:`~stellaRGB.warp.AstWarp` runs at render). No value is
    re-based and none is re-stated: the cards speak the FITS 1-based
    convention natively, so the ``getPixelOrigin``/``getSkyOrigin``/
    ``getCdMatrix`` accessors — and with them the isFits-quirk mismatch
    between accessor and card reference pixels — leave this route
    entirely. Because the probe is policy-free (it reads non-inert PV
    values and even a missing CTYPE pair), the pipe's regime vetoes are
    restated HERE — the ``isFits`` short-circuit first (a WCS that is
    not FITS-representable at all is declined outright, whatever
    approximation its ``getFitsMetadata()`` may fabricate), then the
    explicit ALL-SIP veto (the pipe's distortion-free policy, mirroring
    :func:`pure_fits_tan` — see its NOTE for the widen condition; the
    AST backend reads SIP from the cards themselves, so the pipe must
    veto explicitly rather than inherit the library's acceptance), then
    :func:`_plain_hips_header` (exact HPX CTYPE pair, inert PV pair
    only, HiPS-regime declination). This gate is HPX-only by policy:
    TAN headers stay on the builtin triple route
    (:func:`tan_wcs_from_afw`), never here.

    Fail-closed like :func:`pure_fits_tan`: any unreadable card, an
    accessor exception, or a probe decline (``from_cards`` raises
    ``ValueError`` with AST's own reason — including frames whose
    coordinate round trip does not close) returns ``None`` and the
    caller's maps ladder (or afw fallback) serves the WCS.
    """
    if not wcs.isFits:
        return None
    try:
        fits_md = wcs.getFitsMetadata()
        # Explicit ALL-SIP veto (pipe policy, T8.4) — evaluated BEFORE the
        # probe: a distortion-carrying header must never reach the direct
        # path even though the AST backend can read SIP from the cards.
        # The LTP/LPT vocabulary afw has never applied rides out with it.
        if sip_cards_present(fits_md):
            return None
        # Regime screen (T10.5): the AST probe answers only "AST can
        # render these cards"; the pipe's plain-HiPS regime policy is
        # stated here, so a non-FITS-like header never reaches the probe
        # and a header the probe would happily read cannot enter the
        # direct path off-regime.
        if not _plain_hips_header(fits_md):
            return None
        # The AST probe proper: AST's own reader plus the engine's
        # round-trip gate. A decline raises AST's reason verbatim; the
        # fail-closed contract (None, not an exception) is the job of the
        # except below.
        return AstWcs.from_cards(fits_md)
    except Exception:
        # Fail closed (D19): a WCS whose cards cannot be fully read, or
        # whose frame AST declines (or cannot round trip), is not
        # provably a plain HiPS header.
        return None
