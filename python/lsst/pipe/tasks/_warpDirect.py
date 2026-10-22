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
This module owns that decision (D16/D18/D19): the pure-FITS-TAN verdict
(:func:`pure_fits_tan`), the TAN and HPX header-to-``DirectWcs`` builders
(:func:`tan_wcs_from_afw`, :func:`hpx_direct_wcs_from_afw`), and the SIP-card
scanner they share. Every gate fails CLOSED: a WCS that cannot be fully read,
carries distortion cards, or sits outside a modeled regime is declined with
``False``/``None`` and the caller's fitted-maps ladder (or afw fallback)
serves it — a direct-mode engine hiccup must never crash a task and a
direct render must never be entered on a partially-verified transform.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsst.afw.geom import makeSkyWcs
from lsst.geom import Point2D
from stellaRGB.warp import DirectWcs, TanWcs

if TYPE_CHECKING:
    from lsst.afw.geom import SkyWcs
    from lsst.geom import Box2I

# The inert H=4/K=3 zone-scheme constants (C&R 2007) that every HiPS /
# afw ``makeHpxWcs`` header writes as PV2_1/PV2_2. The engine models only
# that fixed scheme and rejects any non-empty ``pv``, so these two cards —
# at exactly these values — are treated as ABSENT by the HPX gate; any other
# PV-family card (or these names at other values) expresses a non-default
# scheme the engine cannot represent and the WCS is declined.
_HPX_INERT_PV = {"PV2_1": 4.0, "PV2_2": 3.0}

# Tolerance on the HPX tangent-point declination, degrees. The engine's HPX
# arm is realized at crval_dec = 0 exactly (the HiPS regime; its Python
# adapter rejects any other value), and afw's ``makeHpxWcs`` answers a
# |dec| ~ 1e-15 deg SpherePoint for a header CRVAL2 of exactly 0.0 (internal
# radian round-trip). Anything within this tolerance is construction noise
# for a CRVAL2-of-0 header and is snapped to exactly 0.0; anything beyond is
# a different regime (astropy is degenerate there) and is declined.
_HPX_CRVAL2_TOL_DEG = 1e-9


def _is_sip_card(name: str) -> bool:
    """True for FITS SIP metadata cards: the A/B/C/D polynomial order
    declarations, the A_p_q/B_p_q/C_p_q/D_p_q coefficient cards (A/B
    forward distortion, C/D its inverse), and the A_D1/A_D2/B_D1/B_D2
    linear-distortion cards of the SIP standard."""
    if name in ("A_ORDER", "B_ORDER", "C_ORDER", "D_ORDER"):
        return True
    # The linear-distortion cards are not index-shaped ("A_D1".split("_")
    # -> ["A", "D1"] fails the digit test), so they need their own exact
    # match; lookalikes (A_P1, AP1, a bare D1) are not SIP and stay out.
    if name in ("A_D1", "A_D2", "B_D1", "B_D2"):
        return True
    parts = name.split("_")
    return (
        len(parts) == 3
        and parts[0] in ("A", "B", "C", "D")
        and parts[1].isdigit()
        and parts[2].isdigit()
    )


def pure_fits_tan(wcs: SkyWcs, box: Box2I) -> bool:
    """True when ``wcs`` is accepted as pure FITS TAN for direct mode.

    This is a gate of increasing cost, not a proof:

    1. ``isFits`` (a *property* in this stack, answering "FITS-metadata
       representable" — TAN+SIP also satisfies it, SIP keywords being
       FITS-standard metadata; ``hasFitsApproximation()`` is no use, it
       only reports a factory approximation slot) rejects exotic
       non-FITS representations outright.
    2. Exact SIP-card rejection on ``getFitsMetadata()``: any A/B/C/D
       order card or indexed A_p_q-style coefficient card vetoes the
       WCS before any sampling. The order to matters: a numeric probe
       can never PROVE the absence of polynomial distortion — an
       adversarial SIP polynomial with terms chosen to vanish at every
       sample point agrees to the probe's tolerance at all of them
       while reaching 0.05-0.14 px between the samples. The header
       cards are exact evidence and SIP files must carry them.
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


def hpx_direct_wcs_from_afw(wcs: SkyWcs) -> DirectWcs | None:
    """Map an afw HiPS-tile SkyWcs onto the direct engine's HPX arm
    (:class:`stellaRGB.warp.DirectWcs` with ``kind="hpx"``), or return
    ``None`` when the WCS is not a plain HiPS HPX header.

    Accepted shape (all exact evidence from ``isFits`` and
    ``getFitsMetadata()`` plus the origin triple the engine consumes):

    * ``isFits`` True — the same first tier as :func:`pure_fits_tan`: a
      WCS that is not FITS-representable at all is declined outright,
      whatever approximation its ``getFitsMetadata()`` may fabricate.
    * ``CTYPE1``/``CTYPE2`` exactly ``RA---HPX``/``DEC--HPX`` (any case);
      any other CTYPE — including TAN — is declined (callers probe the TAN
      side with :func:`pure_fits_tan`, never here).
    * No SIP cards and no PV-family cards other than the inert H=4/K=3
      pair ``PV2_1 = 4``/``PV2_2 = 3`` that every HiPS / ``makeHpxWcs``
      header carries: those are treated as ABSENT (the engine models only
      that fixed C&R 2007 zone scheme and rejects any non-empty ``pv``,
      so the builder hands it ``pv=()``); any other PV card, or these at
      other values, expresses an inexpressible scheme and is declined.
    * Tangent-point declination 0 (the HiPS regime), within
      ``_HPX_CRVAL2_TOL_DEG`` of float noise, snapped to exactly ``0.0``
      (the value the header states and the engine requires); a meaningfully
      non-zero dec0 is a different regime (astropy is degenerate there)
      and is declined.
    * A finite origin triple.

    Fail-closed like :func:`pure_fits_tan`: any unreadable card, accessor
    exception, or off-shape value returns ``None`` and the caller's maps
    ladder (or afw fallback) serves the WCS.
    """
    if not wcs.isFits:
        return None
    try:
        fits_md = wcs.getFitsMetadata()
        cards = {name.upper(): fits_md.get(name) for name in fits_md.names()}
        ctype1 = cards.get("CTYPE1")
        ctype2 = cards.get("CTYPE2")
        if (
            not isinstance(ctype1, str)
            or not isinstance(ctype2, str)
            or ctype1.strip().upper() != "RA---HPX"
            or ctype2.strip().upper() != "DEC--HPX"
        ):
            return None
        for name, value in cards.items():
            if _is_sip_card(name):
                return None
            if name.startswith("PV"):
                if name in _HPX_INERT_PV:
                    # Inert only at exactly its declared constant, and
                    # only from a genuine numeric card: bool is an int
                    # subclass but True/False is no zone-scheme constant,
                    # and a string card ('4') is not the numeric evidence
                    # this gate's pv contract reads — both fail closed.
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        return None
                    if not np.isfinite(float(value)) or float(value) != _HPX_INERT_PV[name]:
                        return None
                else:
                    return None
        origin = wcs.getPixelOrigin()
        sky = wcs.getSkyOrigin()
        cd = np.asarray(wcs.getCdMatrix(), dtype=float)
        ra_deg = sky.getRa().asDegrees()
        dec_deg = sky.getDec().asDegrees()
        if not (
            np.isfinite(origin.getX())
            and np.isfinite(origin.getY())
            and bool(np.isfinite(cd).all())
            and np.isfinite(ra_deg)
            and np.isfinite(dec_deg)
            and abs(dec_deg) <= _HPX_CRVAL2_TOL_DEG
        ):
            return None
        return DirectWcs(
            kind="hpx",
            crpix=(origin.getX(), origin.getY()),
            # Snapped (see the tolerance note above): the header states
            # CRVAL2 = 0 and the engine arm is realized at exactly that.
            crval_deg=(ra_deg, 0.0),
            cd_deg_per_pix=cd,
            pv=(),
        )
    except Exception:
        # Fail closed (D19): a WCS whose cards or origin triple cannot be
        # fully read is not provably a plain HiPS header.
        return None
