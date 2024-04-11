Database and AGOX observer
===========================

.. literalinclude:: database_and_main_observer.py
    :language: python
    :lines: 51-77

This is similar to the :ref:`two method observer` but with the key difference 
that this :code:`DatabaseAndMainObserver` uses the :code:`handler_identifier`
argument to define which :code:`ObserverHandler` each method will be attached to. 
This defaults in such a way that the method will attach to any :code:`ObserverHandler`, 
but as in this example we can overwrite that. 

The output of an iteration using this observer, looks like so:: 

    |=============================== Iteration: 1 ================================|
    |========================== DatabaseAndMainObserver ==========================|
    | I am run from main, I see only the state: <agox.main.State object at        |
    | 0x7f0fe525da30>                                                             |
    |======================== LocalOptimizationEvaluator =========================|
    | Trying candidate - remaining 1                                              |
    | Step 0: 8.325                                                               |
    | Step 1: 8.158                                                               |
    | Step 2: 8.094                                                               |
    | Final energy of candidate = 8.094                                           |
    | Succesful calculation of candidate.                                         |
    | Calculated required number of candidates.                                   |
    |========================== DatabaseAndMainObserver ==========================|
    | Database size: 1                                                            |
    | I am run from the database, I see the database                              |
    | <agox.databases.database.Database object at 0x7f0fe88568b0> and             |
    | <agox.main.State object at 0x7f0fe525da30>.                                 |
    |================================= Database ==================================|
    | Energy 000001: 8.09437941190291                                             |
    |================================== Logger ===================================|
    | Total time 00.02 s                                                          |
    |  DatabaseAndMainObserver.main_observer_method - 00.00 s                     |
    |  RandomGenerator.generate - 00.01 s                                         |
    |  LocalOptimizationEvaluator.evaluate - 00.01 s                              |
    |  Database.store_in_database - 00.00 s                                       |
    |      DatabaseAndMainObserver.database_observer_method - 00.00 s             |
    |============================ Iteration finished =============================|

Which shows the expected output. 

.. note:: 

    In the currently implemented modules of AGOX the methods that observer the 
    database generally do not get or set anything, they simply read from the database 
    and do something with that e.g. train a model or setup a sampler. The 
    check that happens in the beginning of an AGOX-run does not take the 
    :code:`gets` or :code:`sets` of database-observers into account and may 
    thus be incorrect in cases where this is actually used.
    