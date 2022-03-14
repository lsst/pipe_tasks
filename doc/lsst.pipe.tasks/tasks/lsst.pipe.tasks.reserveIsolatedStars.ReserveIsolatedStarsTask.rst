.. lsst-task-topic:: lsst.pipe.tasks.reserveIsolatedStars.ReserveIsolatedStarsTask

########################
ReserveIsolatedStarsTask
########################

`ReserveIsolatedStarsTask` is a simple task to generate a consistent selection of reserved stars.

.. _lsst.pipe.tasks.reserveIsolatedStars.ReserveIsolatedStarsTask-summary:

Processing summary
==================

`ReserveIsolatedStarsTask` uses a hashed string to generate a random seed to select a random subset of the total number of stars.
The hash is generated from the lower 32 bits of the SHA-256 hash of a combination of the configured ``reserve_name`` and an optional ``extra`` string.
In this way, consistent reserve selections can be done based on human-readable names, bands, etc, rather than relying on hard-to-remember numbers.
The return value is a boolean array of the desired length, with ``reserve_fraction`` of the array set to ``True``.

.. _lsst.pipe.tasks.reserveIsolatedStars.ReserveIsolatedStarsTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.pipe.tasks.reserveIsolatedStars.ReserveIsolatedStarsTask

.. _lsst.pipe.tasks.reserveIsolatedStars.ReserveIsolatedStarsTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.pipe.tasks.reserveIsolatedStars.ReserveIsolatedStarsTask
