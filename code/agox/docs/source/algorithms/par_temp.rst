Parallel Tempering
===================

The script below implements a parallel-tempering basin hopping (PT-BH) algorithm 
where several basin-hopping runs are run concurrently at different temperatures. 
Workers with adjacent temperatures swap their Metropolis accepted candidate.

.. literalinclude:: ../../../agox/test/run_tests/script_pt.py
