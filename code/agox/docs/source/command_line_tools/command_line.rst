Command-line tools
================================

Upon installation AGOX adds a few convenient command-line tools

:code:`agox-convert` can be used to convert AGOX database files to ASE trajectories 

.. code-block:: console

    agox-convert <DB_FILES> -n <TRAJECTORY_NAME>

and :code:`agox-analysis` runs the analysis program on directories containing 
db-files

.. code-block:: console

    agox-analysis -d <DIRECTORIES>

