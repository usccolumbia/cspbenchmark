Evolutionary algorithm
=======================

In an evolutionary algorithm a population is 'evolved' following a fitness 
criteria, this is kept track of in AGOX by a sampler. The algorithm is rather 
similar to a basin-hopping algorithm in terms of the overall elements, but some 
of the individual elements are slightly different. If the population has size *N*
the script below does the following: 

- Generate *N* candidates using the population. 
- Locally optimize the *N* candidates. 
- Store *N* candidates in the database. 
- Update the population. 

A key difference in the script below compared to basin-hopping script is that 
the sampler is not given as an argument to AGOX but rather attached to database. 

.. literalinclude:: ../../../agox/test/run_tests/script_ea.py
