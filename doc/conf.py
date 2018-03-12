"""Sphinx configuration file for an LSST stack package.

This configuration only affects single-package Sphinx documenation builds.
"""

from documenteer.sphinxconfig.stackconf import build_package_configs
import lsst.pipe.tasks


_g = globals()
_g.update(build_package_configs(
    project_name='pipe_tasks',
    version=lsst.pipe.tasks.version.__version__))
