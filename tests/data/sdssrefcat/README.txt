Go to SDSS SkyServer using a web browser
Go to the SQL query page
Select "FITS" as the output format
Run this query:
    select objid as id, ra, dec, type, u, g, r, i, z, err_u as u_err, err_g as g_err, err_r as r_err, err_i as i_err, err_z as z_err from PhotoPrimary where ra between 215.1 and 216.1 and dec between 52.6 and 53.6 order by id
Your browser will download a file; rename it to photocal.fits
Make a copy replacing "type" with a "resolved" flag, which is set for objects that are not stars (SDSS type != 6);
"photometric" is set for stars (SDSS type == 6) that have high S/N (r_err < 0.1). This uses "fitscopy",
which is a cfitsio example program that is easy to download and install:
    fitscopy photocal.fits"[col id;ra;dec;u;g;r;i;z;u_err;g_err;r_err;i_err;z_err;resolved=(type!=6);photometric=(type==6 && r_err<0.1)]" photocal2.fits
Set up pipe_tasks
Ingest the catalog:
    ingestReferenceCatalog.py . photocal2.fits --configfile ingestReferenceCatalogOverride.py
Check the resulting config file
    data/sdssrefcat/config/IngestIndexedReferenceTask.py
and remove unwanted imports, if present (see ticket DM-7002)
