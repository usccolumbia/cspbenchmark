Random structure search 
=========================

The script below runs a 'random structure search' type run, consisting of the 
following elemenets in each iteration. 

- Generate a random candidate. 
- Locally optimize that candidate. 
- Add the fully relaxed candidate to the database. 

.. literalinclude:: ../../../agox/test/run_tests/script_rss.py