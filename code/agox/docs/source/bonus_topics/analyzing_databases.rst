Analyzing databases 
====================

As discussed in the Getting Started section :code:`agox/utils/batch_analysis.py` 
may be used to calculate statistics about a run. 

However if a more detailed analysis of a search is required extracting all 
structures from a database is generally the first step, this can be done as such 

.. code-block:: python 

    from agox.databases import Database
    from ase.io import write

    database = Database(filename=path_to_db)

    database.restore_to_memory()

    candidates = database.get_all_candidates()

    write('database.traj', candidates)

Which produces a ASE trajectory file that can be worked with as any other ASE 
trajectory file. 


.. note:: 

    It is also possible to write a trajectory at the end of a search by simply 
    adding::

        candidates = database.get_all_candidates()
        write('database.traj', candidates)

    After::

        agox.run(n_iterations=...)

    If using a BoxConstraint this may lead to trouble when trying 
    to read the file, so remove the constraints first::

        candidates = database.get_all_candidates()

        for candidate in candidates:
            candidate.set_constraints([])

        write('database.traj', candidates)

