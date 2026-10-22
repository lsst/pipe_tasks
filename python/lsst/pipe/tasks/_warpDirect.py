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
pure-FITS-TAN verdict (:func:`pure_fits_tan`), the TAN header-to-``DirectWcs``
builder (:func:`tan_wcs_from_afw`), and the HPX builder
(:func:`hpx_direct_wcs_from_afw`). The FITS *card vocabulary* — the CTYPE
pair → kind table, the SIP-card scan (quartet + digit patterns), the inert
``PV2_1=4``/``PV2_2=3`` rule and the CRVAL-dec construction-noise snap —
lives in :mod:`stellaRGB.warp` (:func:`~stellaRGB.warp.direct_spec_from_cards`,
:func:`~stellaRGB.warp.sip_cards_present`) so non-pipeline callers get the
same trap book for free; what stays HERE is pipeline risk defense, not FITS
knowledge: accessor extraction, the ``isFits``-quirk short-circuit, the
9-point ``makeSkyWcs`` agreement probe and the TAN-only policy. Every gate
fails CLOSED: a WCS that cannot be fully read, carries distortion cards, or
sits outside a modeled regime is declined with ``False``/``None`` and the
caller's fitted-maps ladder (or afw fallback) serves it — a direct-mode
engine hiccup must never crash a task and a direct render must never be
entered on a partially-verified transform.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from stellaRGB.warp import DirectWcs, TanWcs, direct_spec_from_cards, sip_cards_present

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

    Genuine afw glue only (T8.8): the ``isFits`` short-circuit first (a
    WCS that is not FITS-representable at all is declined outright,
    whatever approximation its ``getFitsMetadata()`` may fabricate),
    then the card set and the origin triple the engine consumes are
    extracted and handed to
    :func:`stellaRGB.warp.direct_spec_from_cards` — which owns the
    whole accepted shape (exact ``RA---HPX``/``DEC--HPX`` CTYPE pair,
    SIP-card rejection, the inert H=4/K=3 ``PV2_1 = 4``/``PV2_2 = 3``
    pair treated as ABSENT with every other PV card declined, and the
    CRVAL-dec construction-noise tolerance snapped to exactly ``0.0``).
    This gate is HPX-only: the classifier accepts every registry kind,
    so any other kind (TAN included — callers probe the TAN side with
    :func:`pure_fits_tan`, never here) is filtered back to ``None``.

    Fail-closed like :func:`pure_fits_tan`: any unreadable card, accessor
    exception (``direct_spec_from_cards`` raises its normal validation
    family for a malformed frame triple — e.g. a NaN origin), or
    declined verdict returns ``None`` and the caller's maps ladder (or
    afw fallback) serves the WCS.
    """
    if not wcs.isFits:
        return None
    try:
        origin = wcs.getPixelOrigin()
        sky = wcs.getSkyOrigin()
        cd = np.asarray(wcs.getCdMatrix(), dtype=float)
        spec = direct_spec_from_cards(
            wcs.getFitsMetadata(),
            # getPixelOrigin() is afw's 0-based reference pixel (FITS
            # CRPIX - 1) — pass it through; the -1 belongs to callers
            # holding raw 1-based header values, not here.
            (origin.getX(), origin.getY()),
            (sky.getRa().asDegrees(), sky.getDec().asDegrees()),
            cd,
        )
    except Exception:
        # Fail closed (D19): a WCS whose cards or origin triple cannot be
        # fully read is not provably a plain HiPS header.
        return None
    if spec is None or spec.kind != "hpx":
        return None
    return spec
