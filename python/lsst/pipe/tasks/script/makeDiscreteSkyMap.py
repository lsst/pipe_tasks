# This file is part of obs_base.
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

from lsst.daf.butler import Butler
from lsst.pipe.tasks.makeDiscreteSkyMap import MakeDiscreteSkyMapTask, MakeDiscreteSkyMapConfig
from lsst.obs.base.utils import getInstrument


def makeDiscreteSkyMap(repo, config_file, collections, instrument, out_collection):
    """Implements the command line interface `butler make-discrete-skymap` subcommand,
    should only be called by command line tools and unit test code that tests
    this function.

    Constructs a skymap from calibrated exposure in the butler repository

    Parameters
    ----------
    repo : `str`
        URI to the location to read the repo.
    config_file : `str` or `None`
        Path to a config file that contains overrides to the skymap config.
    collections : `list` [`str`]
        An expression specifying the collections to be searched (in order) when
        reading datasets, and optionally dataset type restrictions on them.
        At least one collection must be specified.  This is the collection
        with the calibrated exposures.
    insrument : `str`
        The name or fully-qualified class name of an instrument.
    append : `bool`
        If True, an existing skymap will be loaded to seed the skymap.
    """
    butler = Butler(repo, collections=collections, writeable=True)
    instr = getInstrument(instrument, butler.registry)
    config = MakeDiscreteSkyMapConfig()
    instr.applyConfigOverrides(MakeDiscreteSkyMapTask._DefaultName, config)


    if config_file is not None:
        config.load(config_file)
    oldSkyMap = None
    if config.doAppend:
        # Add the default collection if appending
        coll_set = set(collections)
        coll_set.add("skymaps")
        collections = list(coll_set)
        try:
            oldSkyMap = butler.get(config.coaddName + "Coadd_skyMap", collections=collections)
        except ValueError:
            raise ValueError(f"Could not find seed sky map for {config.coaddName}Coadd_skyMap, "
                             "but doAppend is set to True.  Aborting.")
    calexp_md_list = []
    calexp_wcs_list = []
    id_list = butler.registry.queryDatasets('calexp', collections=collections)
    for i in id_list:
        calexp_md_list.append(butler.get('calexp.md', dataId=i, collections=collections))
        calexp_wcs_list.append(butler.get('calexp.wcs', dataId=i, collections=collections))
    task = MakeDiscreteSkyMapTask(config=config, butler=butler)
    task.run(calexp_md_list, calexp_wcs_list, oldSkyMap, isGen3=True)
