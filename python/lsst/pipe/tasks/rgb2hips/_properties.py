from __future__ import annotations

__all__ = ("HipsPropertiesSpectralTerm", "HipsPropertiesConfig", "_write_property")

from lsst.pex.config import Config, Field, ListField, ConfigDictField


class HipsPropertiesSpectralTerm(Config):
    lambda_min = Field[float](
        doc="Minimum wavelength (nm)",
    )
    lambda_max = Field[float](
        doc="Maximum wavelength (nm)",
    )


class HipsPropertiesConfig(Config):
    """Configuration parameters for writing a HiPS properties file."""

    creator_did_template = Field[str](
        doc=("Unique identifier of the HiPS - Format: IVOID. Use ``{band}`` to substitute the band name."),
        dtype=str,
        optional=False,
    )
    obs_collection = Field[str](
        doc="Short name of original data set - Format: one word",
        optional=True,
    )
    obs_description_template = Field[str](
        doc=(
            "Data set description - Format: free text, longer free text "
            "description of the dataset.  Use ``{band}`` to substitute "
            "the band name."
        ),
    )
    prov_progenitor = ListField[str](
        doc="Provenance of the original data - Format: free text",
        default=[],
    )
    obs_title_template = Field[str](
        doc=(
            "Data set title format: free text, but should be short. "
            "Use ``{band}`` to substitute the band name."
        ),
        optional=False,
    )
    spectral_ranges = ConfigDictField(
        doc=("Mapping from band to lambda_min, lamba_max (nm).  May be approximate."),
        keytype=str,
        itemtype=HipsPropertiesSpectralTerm,
        default={},
    )
    initial_ra = Field[float](
        doc="Initial RA (deg) (default for HiPS viewer).  If not set will use a point in MOC.",
        optional=True,
    )
    initial_dec = Field[float](
        doc="Initial Declination (deg) (default for HiPS viewer).  If not set will use a point in MOC.",
        optional=True,
    )
    initial_fov = Field[float](
        doc="Initial field-of-view (deg).  If not set will use ~1 healpix tile.",
        optional=True,
    )
    obs_ack = Field[str](
        doc="Observation acknowledgements (free text).",
        optional=True,
    )
    t_min = Field[float](
        doc="Time (MJD) of earliest observation included in HiPS",
        optional=True,
    )
    t_max = Field[float](
        doc="Time (MJD) of latest observation included in HiPS",
        optional=True,
    )

    def validate(self):
        super().validate()

        if self.obs_collection is not None:
            if re.search(r"\s", self.obs_collection):
                raise ValueError("obs_collection cannot contain any space characters.")

    def setDefaults(self):
        # Values here taken from
        # https://github.com/lsst-dm/dax_obscore/blob/44ac15029136e2ec15/configs/dp02.yaml#L46
        u_term = HipsPropertiesSpectralTerm()
        u_term.lambda_min = 330.0
        u_term.lambda_max = 400.0
        self.spectral_ranges["u"] = u_term
        g_term = HipsPropertiesSpectralTerm()
        g_term.lambda_min = 402.0
        g_term.lambda_max = 552.0
        self.spectral_ranges["g"] = g_term
        r_term = HipsPropertiesSpectralTerm()
        r_term.lambda_min = 552.0
        r_term.lambda_max = 691.0
        self.spectral_ranges["r"] = r_term
        i_term = HipsPropertiesSpectralTerm()
        i_term.lambda_min = 691.0
        i_term.lambda_max = 818.0
        self.spectral_ranges["i"] = i_term
        z_term = HipsPropertiesSpectralTerm()
        z_term.lambda_min = 818.0
        z_term.lambda_max = 922.0
        self.spectral_ranges["z"] = z_term
        y_term = HipsPropertiesSpectralTerm()
        y_term.lambda_min = 970.0
        y_term.lambda_max = 1060.0
        self.spectral_ranges["y"] = y_term


# WARNING: In general PipelineTasks are not allowed to do any outputs
# outside of the butler.  This task has been given (temporary)
# Special Dispensation because of the nature of HiPS outputs until
# a more controlled solution can be found.
def _write_property(fh, name, value):
    """Write a property name/value to a file handle.

    Parameters
    ----------
    fh : file handle (blah)
        Open for writing.
    name : `str`
        Name of property
    value : `str`
        Value of property
    """
    # This ensures that the name has no spaces or space-like characters,
    # per the HiPS standard.
    if re.search(r"\s", name):
        raise ValueError(f"``{name}`` cannot contain any space characters.")
    if "=" in name:
        raise ValueError(f"``{name}`` cannot contain an ``=``")

    fh.write(f"{name:25}= {value}\n")
