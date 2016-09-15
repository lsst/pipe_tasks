import lsst.meas.algorithms.ingestIndexReferenceTask
assert type(config) == lsst.meas.algorithms.ingestIndexReferenceTask.IngestIndexedReferenceConfig, 'config is of type %s.%s instead of lsst.meas.algorithms.ingestIndexReferenceTask.IngestIndexedReferenceConfig' % (
    type(config).__module__, type(config).__name__)

# Name of column stating if satisfactory for photometric calibration (optional).
config.is_photometric_name = 'photometric'

# Default HTM level.  Level 8 gives ~0.08 sq deg per trixel.
config.level = 8

# Name of RA column
config.ra_name = 'ra'

# Name of Dec column
config.dec_name = 'dec'

import lsst.meas.algorithms.readFitsCatalogTask
config.file_reader.retarget(target=lsst.meas.algorithms.readFitsCatalogTask.ReadFitsCatalogTask,
                            ConfigClass=lsst.meas.algorithms.readFitsCatalogTask.ReadFitsCatalogConfig)
# HDU containing the desired binary table, 0-based but a binary table never occurs in HDU 0
config.file_reader.hdu = 1

# Mapping of input column name: output column name; each specified column
# must exist, but additional columns in the input data are written using
# their original name.
config.file_reader.column_map = {}

# Name of column stating if the object is resolved (optional).
config.is_resolved_name = 'resolved'

# Name of column to use as an identifier (optional).
config.id_name = 'id'

# The values in the reference catalog are assumed to be in AB magnitudes.
# List of column names to use for photometric information.  At least one
# entry is required.
config.mag_column_list = ['u', 'g', 'r', 'i', 'z']

# A map of magnitude column name (key) to magnitude error column (value).
config.mag_err_column_map = {'i': 'i_err', 'r': 'r_err', 'u': 'u_err', 'z': 'z_err', 'g': 'g_err'}

# String to pass to the butler to retrieve persisted files.
config.ref_dataset_name = 'cal_ref_cat'

# Extra columns to add to the reference catalog.
config.extra_col_names = []

# Name of column stating if the object is measured to be variable (optional).
config.is_variable_name = None
