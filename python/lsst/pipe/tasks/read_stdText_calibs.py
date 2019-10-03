from lsst.meas.algorithms import Defects
from lsst.meas.algorithms.simple_curve import Curve

import os
import glob
import dateutil.parser


def read_one_chip(root, chip_name, chip_id):
    """Read data for a particular sensor from the standard format at a particular root.

    Parameters
    ----------
    root : str
        Path to the top level of the data tree.  This is expected to hold directories
        named after the sensor names.  They are expected to be lower case.
    chip_name : str
        The name of the sensor for which to read data.
    chip_id : int
        The identifier for the sensor in question.

    Returns
    -------
    dict
        A dictionary of objects constructed with ``factory``.
        The key is the validity start time as a `datetime` object.
    """
    factory_map = {'qe_curves': Curve, 'defects': Defects}
    files = glob.glob(os.path.join(root, chip_name, '*.ecsv'))
    parts = os.path.split(root)
    instrument = os.path.split(parts[0])[1]  # convention is that these reside at <instrument>/<data_name>
    data_name = os.path.split(parts[0])[0]
    factory = factory_map[data_name]
    data_dict = {}
    for f in files:
        date_str = os.path.splitext(os.path.basename(f))[0]
        valid_start = dateutil.parser.parse(date_str)
        data_dict[valid_start] = factory.readText(f)
        check_metadata(data_dict[valid_start], valid_start, instrument, chip_id, f, data_name)
    return data_dict


def check_metadata(obj, valid_start, instrument, chip_id, f, data_name):
    """Check that the metadata is complete and self consistent

    Parameters
    ----------
    obj : object of same type as the factory
        Object to retrieve metadata from in order to compare with
        metadata inferred from the path.
    valid_start : datetime
        Start of the validity range for data
    instrument : str
        Name of the instrument in question
    chip_id : int
        Identifier of the sensor in question
    f : str
        Path of the file read to construct the data
    data_name : str
        Name of the type of data being read

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the metadata from the path and the metadata encoded
        in the path do not match for any reason.
    """
    md = obj.getMetadata()
    finst = md.get('INSTRUME')
    fchip_id = md.get('DETECTOR')
    fcalib_date = md.get('CALIBDATE')
    fdata_name = md.get('OBSTYPE')
    if not (finst.lower(), int(fchip_id), fcalib_date, fdata_name) ==
           (instrument.lower(), chip_id, valid_start.isoformat(), data_name):
        st_time = valid_start.isoformat()
        raise ValueError("Path and file metadata do not agree:\n" +
                         "Path metadata: %s, %s, %s, %s\n"%(instrument, chip_id, st_time, fdata_name) +
                         "File metadata: %s, %s, %s, %s\n"%(finst, fchip_id, fcalib_date, data_name) +
                         "File read from : %s\n"%(f)
                         )


def read_all(root, camera):
    """Read all data from the standard format at a particular root.

    Parameters
    ----------
    root : str
        Path to the top level of the data tree.  This is expected to hold directories
        named after the sensor names.  They are expected to be lower case.
    camera : `lsst.afw.cameraGeom.Camera`
        The camera that goes with the data being read.

    Returns
    -------
    dict
        A dictionary of dictionaries of objects constructed with ``factory``.
        The first key is the sensor name, and the second is the validity
        start time as a `datetime` object.
    """
    root = os.path.normpath(root)
    dirs = os.listdir(root)  # assumes all directories contain data
    dirs = [d for d in dirs if os.path.isdir(os.path.join(root, d))]
    data_by_chip = {}
    name_map = {det.getName().lower(): det.getName() for
                det in camera}  # we assume the directories have been lowered
    calib_type_old = None
    for d in dirs:
        chip_name = os.path.basename(d)
        chip_id = camera[name_map[chip_name]].getId()
        data_by_chip[chip_name], calib_type = read_one_chip(root, chip_name, chip_id, factory)
        if calib_type_old is not None:
            if calib_type_old != calib_type:
                raise ValueError(f'Calibration types do not agree: {calib_type_old} != {calib_type}')
    return data_by_chip, calib_type
