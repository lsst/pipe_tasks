from lsst.daf.butler import FormatterV2
from lsst.resources import ResourcePath
from io import BytesIO
from typing import Any

__all__ = ["SSPAuxiliaryFile", "SSPAuxiliaryFileFormatter"]


class SSPAuxiliaryFile():
    """Class to hold information about auxiliary files needed for
    solar system object ephemeris calculations.
    """
    fileContents = None
    def __init__(self, fileContents):
        self.fileContents = fileContents

class SSPAuxiliaryFileFormatter(FormatterV2):
    """Formatter for SSP Auxiliary Files.
    """
    can_read_from_uri = True

    def read_from_uri(self, uri: ResourcePath, component: str | None = None, expected_size: int = -1) -> Any:
        """Read a dataset.

        Parameters
        ----------
        uri : `lsst.ResourcePath`
            Location of the file to read.
        component : `str` or `None`, optional
            Component to read from the file.
        expected_size : `int`, optional
            Expected size of the file.

        Returns
        -------
        payload : `SSPAuxiliaryFile`
            The requested data as a Python object.
        """
        return SSPAuxiliaryFile(BytesIO(uri.read()))

    def to_bytes(self, in_memory_dataset: Any) -> bytes:
        return in_memory_dataset.fileContents