Database observer
==================

Instances of the :code:`database`-module can also take observers, as is e.g. 
done with models - that are then trained whenever data is added to the database. 

.. literalinclude:: database_observer.py
    :language: python
    :lines: 50-68

Note that both the database and the state will be passed to the :code:`DatabaseObserver`
when it is called. 

In this case we will not attach to observer to the main :code:`AGOX`-loop but rather 
manually attached to the database: 

.. literalinclude:: database_observer.py
    :language: python
    :lines: 70-71

And so it is not given to the instantiation: 

.. literalinclude:: database_observer.py
    :language: python
    :lines: 73-73

If we want to make sure the :code:`DataObserver` does what we want we can 
use the following methods: 

.. literalinclude:: database_observer.py
    :language: python
    :lines: 77-78

Which in this case produces this output::

    |========================= Observers set/get reports =========================|
    |   DatabaseObserver.basic_observer_method                                    |
    |       Doesnt set/get anything                                               |
    |                                                                             |
    |   Overall:                                                                  |
    |   Get keys: set()                                                           |
    |   Set keys: set()                                                           |
    |   Key match: True                                                           |
    |================================= Observers =================================|
    |   Order 3 - Name: DatabaseObserver.basic_observer_method                    |

During a search run we will also see the timing information of the observer 
from the :code:`Logger`:: 

    |================================== Logger ===================================|
    | Total time 00.37 s                                                          |
    |  RandomGenerator.generate - 00.01 s                                         |
    |  LocalOptimizationEvaluator.evaluate - 00.37 s                              |
    |  Database.store_in_database - 00.00 s                                       |
    |      DatabaseObserver.database_observer_method - 00.00 s                    |
    |============================ Iteration finished =============================|

.. note::
    In order for the timings to work properly the order of every observer 
    must be different!
