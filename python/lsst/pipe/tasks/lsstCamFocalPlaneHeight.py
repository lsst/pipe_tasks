import numpy as np
from astropy.table import Table
import re
from sklearn.neighbors import KNeighborsRegressor
import tqdm
from lsst.daf.butler import Butler
import lsst.afw.cameraGeom as cameraGeom
import lsst.geom as geom
from lsst.obs.lsst import LsstCam
import lsst.afw.image as afwImage
from lsst.geom import Point2D
from lsst.afw.cameraGeom import PIXELS, FOCAL_PLANE

from astropy.io import fits
from astropy.table import Table


def make_metrology_table(file='/sdf/home/l/laliotis/rubin-user/lsst_dev/meas_extensions_piff/python/lsst/meas/extensions/piff/metrology/LSST_FP_cold_b_measurement_4col_bysurface.fits', rsid=None, write=False):
    """
    Make an astropy table of the height measurement data. 
    Inputs: 
    file: string, file path for measurement file
    rsid: string (optional) like R##_S## if you want data for just one sensor
    write: bool (default False), whether to write out the table as a fits file
    Outputs:
    bigtable: One large astropy table with focal plane x and y coordinates, modeled and measured z values, and the RSID for which detector each fpx,fpy coord pair is on
    """

    rows = []
    with fits.open(file) as hdulist:
        for hdu in tqdm.tqdm(hdulist):
            if isinstance(hdu, fits.BinTableHDU):
                table = Table(hdu.data)
                extname = hdu.header['EXTNAME']
                if rsid is not None:
                    if extname == rsid: # filter to the single det , 172
                        extname = re.sub(r'(R\d\d)(S\d\d)', r'\1_\2', extname)
                        for x, y, z_mod, z_meas in zip(table['X_CCS'], table['Y_CCS'], table['Z_CCS_MODEL'], table['Z_CCS_MEASURED']):
                            fpx = y
                            fpy = x
                            rows.append([fpx, fpy, z_mod, z_meas, extname])
                else:
                    if re.fullmatch(r'R\d\dS\d\d', extname):
                        extname = re.sub(r'(R\d\d)(S\d\d)', r'\1_\2', extname)
                        for x, y, z_mod, z_meas in zip(table['X_CCS'], table['Y_CCS'], table['Z_CCS_MODEL'], table['Z_CCS_MEASURED']):
                                fpx = y
                                fpy = x
                                rows.append([fpx, fpy, z_mod, z_meas, extname]) 

        bigtable = Table(rows=rows, names=['fpx', 'fpy', 'z_mod', 'z_meas', 'det'])
        if write: bigtable.write('metrology_fp.fits', format='fits', overwrite=True)

    return bigtable



def get_height_interpolator(metrology_table, k=3, weight_type='distance', ztype='measured'):
    """
    Create a KNN interpolator for height given the metrology table created above
    Inputs:
    table: astropy.table.Table, Table with columns 'fpx', 'fpy', and 'z_meas'.
    k : int, Number of neighbors to use.
    weight_type: str, 'uniform' or 'distance' weighting for KNN.
    Returns:
    interp_func : function, function that takes (fpx, fpy) and returns interpolated z_meas.
    """

    x=np.column_stack((metrology_table['fpx'], metrology_table['fpy']))
    if ztype=='measured': y=np.array(metrology_table['z_meas'])
    elif ztype=='model': y=np.array(metrology_table['z_mod'])
    else: raise ValueError('Specify a valid z type (measured, model)')

    knn = KNeighborsRegressor(n_neighbors=k, weights=weight_type)
    knn.fit(x, y)

    def interp_func(x, y):
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        point = np.column_stack((x, y))
        z = knn.predict(point)
        return z[0] #because z is an array


    return interp_func



def detector_fromcoords(x, y, camera):
    """
    Function to get the detector ID for a given coordinate pair
    Inputs:
    x: float, x-coordinate
    y: float, y-coordinate
    camera: lsst.obs.lsst object, the camera being used (lsstcam or comcam)
    """

    cam = camera.getCamera()
    point = geom.Point2D(x, y)

    for det in cam:
        bbox = det.getBBox()
        transform = det.getTransform(FOCAL_PLANE, det.PIXELS)
        try:
            pixel_point = transform.applyForward(point)
            if bbox.contains(pixel_point):
                return det.getId()
        except Exception:
            continue

    raise ValueError(f"No detector contains point ({x},{y}) in focal plane coords.")


def pixel_to_focal(x_arr, y_arr, det_arr):
    """
    Convert pixel coordinates to focal plane coordinates for a given detector.
    
    Inputs:
    x: float, x-coordinate in pixel space
    y: float, y-coordinate in pixel space
    det: int, detector ID
    
    Returns:
    focal_point: lsst.geom.Point2D, focal plane coordinates
    """
    
    cam = LsstCam.getCamera()

    # Cache transforms to avoid redundant lookups
    transform_cache = {}
    fpx_list = []
    fpy_list = []

    for x, y, d in zip(x_arr, y_arr, det_arr):
        if d not in transform_cache:
            det = cam[d]
            transform_cache[d] = det.getTransform(PIXELS, FOCAL_PLANE)

        tx = transform_cache[d]
        pt = tx.applyForward(Point2D(x, y))
        fpx_list.append(pt.getX())
        fpy_list.append(pt.getY())

    return np.array(fpx_list), np.array(fpy_list)

