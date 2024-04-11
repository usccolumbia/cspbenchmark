Basin hopping
==============

In basin hopping a sampler is used to keep track of a previously evaluated 
candidate which informs the generation of a new candidate. The algorithm 
involves the following steps in each iteration 

- Generate a candidate by rattling a previous candidate. 
- Locally optimize the generated candidate. 
- Check Metropolis criterion to determine if the new candidate is accepted as the starting point for generation. 
- Store the candidate in the database. 

The script below implements basin-hopping in AGOX.

.. literalinclude:: ../../../agox/test/run_tests/script_bh.py

    
