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

from lsst.daf.butler import Butler, DatasetType
from lsst.pipe.tasks.makeDiscreteSkyMap import MakeDiscreteSkyMapTask, MakeDiscreteSkyMapConfig
from lsst.obs.base.utils import getInstrument


def makeDiscreteSkyMap(repo, config_file, collections, instrument,
                       out_collection='skymaps', skymap_id='discrete'):
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
    out_collection : `str`, optional
        The name of the collection to save the skymap to.  Default is 'skymaps'.
    skymap_id : `str`, optional
        The identifier of the skymap to save.  Default is 'discrete'.
    """
    butler = Butler(repo, collections=collections, writeable=True, run=out_collection)
    instr = getInstrument(instrument, butler.registry)
    config = MakeDiscreteSkyMapConfig()
    instr.applyConfigOverrides(MakeDiscreteSkyMapTask._DefaultName, config)

    if config_file is not None:
        config.load(config_file)
    skymap_name = config.coaddName + "Coadd_skyMap"
    oldSkyMap = None
    if config.doAppend:
        if out_collection in collections:
            raise ValueError(f"Cannot overwrite dataset.  If appending, specify an output "
                             f"collection not in the input collections.")
        dataId = {'skymap': skymap_id}
        try:
            oldSkyMap = butler.get(skymap_name, collections=collections, dataId=dataId)
        except LookupError as e:
            msg = (f"Could not find seed skymap for {skymap_name} with dataId {dataId} "
                   f"in collections {collections} but doAppend is {config.doAppend}.  Aborting...")
            raise LookupError(msg, *e.args[1:])

    wcs_md_tuple_list = []
    id_list = butler.registry.queryDatasets('calexp', collections=collections)
    for i in id_list:
        wcs_md_tuple_list.append((butler.get('calexp.metadata', dataId=i.dataId, collections=collections),
                                  butler.get('calexp.wcs', dataId=i.dataId, collections=collections)))
    task = MakeDiscreteSkyMapTask(config=config)
    result = task.run(wcs_md_tuple_list, oldSkyMap)
    skymap_dataset_type = DatasetType(skymap_name, dimensions=["skymap", ],
                                      universe=butler.registry.dimensions,
                                      storageClass="SkyMap")
    butler.registry.registerDatasetType(skymap_dataset_type)
    if config.doAppend:
        # By definition if appending the dataset has already been registered
        result.skyMap.register(skymap_id, butler.registry)
    butler.put(result.skyMap, skymap_name, dataId={'skymap': skymap_id})
