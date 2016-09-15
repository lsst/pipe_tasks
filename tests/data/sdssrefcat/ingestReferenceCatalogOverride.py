from lsst.meas.algorithms.readFitsCatalogTask import ReadFitsCatalogTask
config.file_reader.retarget(ReadFitsCatalogTask)
config.ra_name = "ra"
config.dec_name = "dec"
config.mag_column_list = ["u", "g", "r", "i", "z"]
config.mag_err_column_map = {"u": "u_err", "g": "g_err", "r": "r_err", "i": "i_err", "z": "z_err"}
config.is_photometric_name = "photometric"
config.is_resolved_name = "resolved"
config.id_name = "id"
