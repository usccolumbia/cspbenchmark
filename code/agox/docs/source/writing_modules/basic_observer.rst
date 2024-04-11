Basic Observer
===============

At first we will implement a very basic observer that does nothing more than 
informs us of its existences. We will add it to the script that we used 
in the :ref:`Getting started` section, that is repeated here: 

.. literalinclude:: ../getting_started/rss_script_single.py

This very basic Observer can be written like so 

.. literalinclude:: basic_observer.py
    :language: python
    :lines: 50-66

We have the :code:`BasicObserver` inherit from both :code:`Observer` and :code:`Writer`,
which does make it look a little more daunting. The method we want the observer
to execute is :code:`basic_observer_method` and hence that method is passed to 
:code:`add_observer_method`. The decorators :code:`agox_writer` and :code:`Observer.observer_method`
give this method added functionality behind the scenes, so they should always 
be used on methods that will attached as observer-methods.

We include it in the AGOX as any other observer, 

.. literalinclude:: basic_observer.py
    :language: python
    :lines: 68-70

When run we will get the following output::

    |========================== Initialization starting ==========================|
    |================================= Observers =================================|
    |   Order 1 - Name: RandomGenerator.generate                                  |
    |   Order 2 - Name: LocalOptimizationEvaluator.evaluate                       |
    |   Order 3 - Name: Database.store_in_database                                |
    |   Order 3 - Name: BasicObserver.basic_observer_method                       |
    |   Order 4 - Name: Logger.report_logs                                        |
    |========================= Observers set/get reports =========================|
    |   RandomGenerator.generate                                                  |
    |       Sets 'candidates'                                                     |
    |   LocalOptimizationEvaluator.evaluate                                       |
    |       Gets 'candidates'                                                     |
    |       Sets 'evaluated_candidates'                                           |
    |   Database.store_in_database                                                |
    |       Gets 'evaluated_candidates'                                           |
    |   BasicObserver.basic_observer_method                                       |
    |       Doesnt set/get anything                                               |
    |   Logger.report_logs                                                        |
    |       Doesnt set/get anything                                               |
    |                                                                             |
    |   Overall:                                                                  |
    |   Get keys: {'evaluated_candidates', 'candidates'}                          |
    |   Set keys: {'evaluated_candidates', 'candidates'}                          |
    |   Key match: True                                                           |
    |========================== Initialization finished ==========================|

Which tells us that the BasicObserver will run with order = 3 and does not 
try to set or get anything. 