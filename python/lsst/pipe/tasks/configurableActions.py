import warnings

warnings.warn(
    "lsst.pipe.tasks.configurableActions is deprecated; "
    "it has been moved to lsst.pex.config.configurableActions. "
    "Accessing though lsst.pipe.tasks will be removed from "
    "Science Pipelines after release 26.0.0",
    stacklevel=2,  # Issue warning from caller.
    category=FutureWarning
)

from lsst.pex.config.configurableActions import *  # noqa: F401 E402, F403
