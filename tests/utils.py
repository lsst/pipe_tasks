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

__all__ = ["makeTestVisitInfo", "make_exposure_record"]

import astropy.units as u

from lsst.afw.coord import Observatory
import lsst.afw.image as afwImage
import lsst.daf.base as dafBase
import lsst.daf.butler
import lsst.geom as geom


def makeTestVisitInfo(id=1):
    """Return a non-NaN visitInfo."""
    return afwImage.VisitInfo(id=id,
                              date=dafBase.DateTime(65321.1, dafBase.DateTime.MJD, dafBase.DateTime.TAI),
                              era=45.1*geom.degrees,
                              boresightRaDec=geom.SpherePoint(23.1, 73.2, geom.degrees),
                              boresightAzAlt=geom.SpherePoint(134.5, 33.3, geom.degrees),
                              boresightRotAngle=73.2*geom.degrees,
                              rotType=afwImage.RotType.SKY,
                              observatory=Observatory(11.1*geom.degrees, 22.2*geom.degrees, 0.333))


def make_exposure_record(exposure, instrument="testCam"):
    """Return an ``exposure`` dimension record that describes a test exposure.

    Parameters
    ----------
    exposure : `lsst.afw.image.Exposure`
        Exposure to take the observation metadata from.
    instrument : `str`, optional
        Name of the instrument to record.

    Returns
    -------
    record : `lsst.daf.butler.DimensionRecord`
        Record holding the observation metadata that
        `lsst.images.VisitImage.from_legacy` reads.
    """
    visit_info = exposure.visitInfo
    begin = visit_info.date.toAstropy()
    universe = lsst.daf.butler.DimensionUniverse()
    return universe["exposure"].RecordClass(
        instrument=instrument,
        id=visit_info.id,
        obs_id=f"test_{visit_info.id}",
        group=str(visit_info.id),
        physical_filter=exposure.filter.physicalLabel,
        day_obs=int(begin.strftime("%Y%m%d")),
        exposure_time=visit_info.exposureTime,
        seq_num=1,
        seq_start=1,
        seq_end=1,
        can_see_sky=True,
        timespan=lsst.daf.butler.Timespan(begin=begin, end=begin + visit_info.exposureTime*u.s),
    )
