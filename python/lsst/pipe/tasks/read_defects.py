from lsst.meas.algorithms import Defects
import os
import glob
import dateutil.parser


def read_defects_one_chip(root, chip_name, chip_id):
    """Read defects for a particular sensor from the standard format at a particular root.

    Parameters
    ----------
    root : str
        Path to the top level of the defects tree.  This is expected to hold directories
        named after the sensor names.  They are expected to be lower case.
    chip_name : str
        The name of the sensor for which to read defects.
    chip_id : int
        The identifier for the sensor in question.

    Returns
    -------
    dict
        A dictionary of `lsst.meas.algorithms.Defects`.
        The key is the validity start time as a `datetime` object.
    """
    files = glob.glob(os.path.join(root, chip_name, '*.ecsv'))
    parts = os.path.split(root)
    instrument = os.path.split(parts[0])[1]  # convention is that these reside at <instrument>/defects
    defect_dict = {}
    for f in files:
        date_str = os.path.splitext(os.path.basename(f))[0]
        valid_start = dateutil.parser.parse(date_str)
        defect_dict[valid_start] = Defects.readText(f)
        check_metadata(defect_dict[valid_start], valid_start, instrument, chip_id, f)
    return defect_dict


def check_metadata(defects, valid_start, instrument, chip_id, f):
    """Check that the metadata is complete and self consistent

    Parameters
    ----------
    defects : `lsst.meas.algorithms.Defects`
        Object to retrieve metadata from in order to compare with
        metadata inferred from the path.
    valid_start : datetime
        Start of the validity range for defects
    instrument : str
        Name of the instrument in question
    chip_id : int
        Identifier of the sensor in question
    f : str
        Path of the file read to produce ``defects``

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the metadata from the path and the metadata encoded
        in the path do not match for any reason.
    """
    md = defects.getMetadata()
    finst = md.get('INSTRUME')
    fchip_id = md.get('DETECTOR')
    fcalib_date = md.get('CALIBDATE')
    if not (finst.lower(), int(fchip_id), fcalib_date) == (instrument.lower(),
                                                           chip_id, valid_start.isoformat()):
        raise ValueError("Path and file metadata do not agree:\n" +
                         "Path metadata: %s, %s, %s\n"%(instrument, chip_id, valid_start.isoformat()) +
                         "File metadata: %s, %s, %s\n"%(finst, fchip_id, fcalib_date) +
                         "File read from : %s\n"%(f)
                         )


def read_all_defects(root, camera):
    """Read all defects from the standard format at a particular root.

    Parameters
    ----------
    root : str
        Path to the top level of the defects tree.  This is expected to hold directories
        named after the sensor names.  They are expected to be lower case.
    camera : `lsst.afw.cameraGeom.Camera`
        The camera that goes with the defects being read.

    Returns
    -------
    dict
        A dictionary of dictionaries of `lsst.meas.algorithms.Defects`.
        The first key is the sensor name, and the second is the validity
        start time as a `datetime` object.
    """
    root = os.path.normpath(root)
    dirs = os.listdir(root)  # assumes all directories contain defects
    dirs = [d for d in dirs if os.path.isdir(os.path.join(root, d))]
    defects_by_chip = {}
    name_map = {det.getName().lower(): det.getName() for
                det in camera}  # we assume the directories have been lowered
    for d in dirs:
        chip_name = os.path.basename(d)
        chip_id = camera[name_map[chip_name]].getId()
        defects_by_chip[chip_name] = read_defects_one_chip(root, chip_name, chip_id)
    return defects_by_chip
