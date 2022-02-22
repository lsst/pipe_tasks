.. lsst-task-topic:: lsst.pipe.tasks.isolatedStarAssociation.IsolatedStarAssociationTask

###########################
IsolatedStarAssociationTask
###########################

``IsolatedStarAssociationTask`` matches unresolved star sources from the ``sourceTable_visit`` datasets into a pure (but not complete) catalog of isolated stars.
The output datasets are a catalog of isolated stars (dataset ``isolated_star_cat``) and the associated sources with configured columns from the input catalogs (dataset ``isolated_star_sources``).

.. _lsst.pipe.tasks.isolatedStarAssociation.IsolatedStarAssociationTask-summary:

Processing summary
==================

``IsolatedStarAssociationTask`` reads in all the visit-level source table parquet catalogs in a given tract and matches (associates) the sources together.
In particular:

- All ``sourceTable_visit`` catalogs overlapping a given tract are used as inputs.
  The configuration variable ``extra_columns`` can be used to specify which columns from the inputs are persisted (in addition to the default flux column, input row number, and source id).
- Unflagged, unresolved stars above a configured signal-to-noise are selected from each input catalog.
- The input stars are associated within a configured ``match_radius`` arcseconds.
  This association is done band-by-band (with the order configured by ``band_order``), and then each band is matched such that all sources are associated with a unique list of stars which cover all the bands in the tract.
- The star catalog is matched against itself, and groups of stars that match neighbor(s) within ``isolation_radius`` arcseconds are removed from the final isolated star catalog (``isolated_star_cat``).
- All individual sources are associated with the isolated star catalog within ``match_radius``, to create the ``isolated_star_sources``.
- The data are collated such that each isolated star has an index to the ``isolated_star_sources`` table for quick access of all the individual sources for that star.

.. _lsst.pipe.tasks.isolatedStarAssociation.IsolatedStarAssociationTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.pipe.tasks.isolatedStarAssociation.IsolatedStarAssociationTask

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.pipe.tasks.isolatedStarAssociation.IsolatedStarAssociationTask

.. _lsst.pipe.tasks.isolatedStarAssociation.IsolatedStarAssociationTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.pipe.tasks.isolatedStarAssociation.IsolatedStarAssociationTask
