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


from collections import defaultdict
import logging
from sqlalchemy.exc import IntegrityError

from lsst.daf.butler import Butler


_log = logging.getLogger(__name__)


registeredMsg = "Registered subfilters {subfilters} for filter band \"{band}\"."
notRegisteredMsg =  \
    "Not registering subfilters for filter band \"{band}\"; subfilters {subfilters} already existed."


class InsertResults:
    """Represents the results of adding subfilters to a filter band."""

    class InsertedSubfilters:
        """Keeps track of the inserted and existing subfilter numbers.

        Attributes
        ----------
        inserted : `list` [`int`]
            The inserted subfilters.
        existing : `list` [`int`]
            The subfilters that already existed.
        """
        def __init__(self):
            self.inserted = []
            self.existing = []

    def __init__(self):
        self.filters = defaultdict(self.InsertedSubfilters)

    def add(self, filterName, subfilter, inserted):
        """Add result information about attemping to add a subfilter to a
        filter band.

        Parameters
        ----------
        filterName : `str`
            The name of the filter band.
        subfilter : `int`
            The subfilter id.
        inserted : `bool`
            `True` if the subfilter was inserted, or `False` if this is the id
            of a subfilter that already existed.
        """
        if inserted:
            self.filters[filterName].inserted.append(subfilter)
        else:
            self.filters[filterName].existing.append(subfilter)

    def __str__(self):
        """Get the results formated for CLI output.

        Returns
        -------
        results : `str`
            The results formatted for CLI output.
        """
        ret = ""
        for filterName, subs in self.filters.items():
            if ret:
                ret += "\n"
            if subs.inserted:
                ret += registeredMsg.format(band=filterName, subfilters=subs.inserted)
            if subs.existing:
                subs.existing.sort()
                ret += notRegisteredMsg.format(band=filterName, subfilters=subs.existing)
        return ret


def registerDcrSubfilters(repo, num_subfilters, band_names):
    """Construct a set of subfilters for chromatic modeling and add them to a
    registry.

    Parameters
    ----------
    repo : `str`
        URI to the location to read the repo.
    num_subfilters : `int`
        The number of subfilters to add.
    band_names : `list` [`str`]
        The filter band names to add.

    Returns
    -------
    insertResults : ``InsertResults``
        A class that contains the results of the subfilters that were inserted
        or already exist in each filter band, that has a __str__ method so it
        can be easily printed to the CLI output.
    """
    results = InsertResults()
    with Butler.from_config(repo, writeable=True) as butler:
        for filterName in band_names:
            try:
                with butler.registry.transaction():
                    for sub in range(num_subfilters):
                        butler.registry.insertDimensionData(
                            "subfilter", {"band": filterName, "subfilter": sub}
                        )
                        results.add(filterName, sub, True)
            except IntegrityError:
                records = butler.registry.queryDimensionRecords("subfilter", dataId={"band": filterName})
                for record in records:
                    results.add(filterName, record.id, False)

    return results
