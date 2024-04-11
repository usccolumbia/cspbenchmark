Two method observer
===========================

An observer may add any number of observer-methods, as shown in the example below. 

.. literalinclude:: two_methods_observer.py
    :language: python
    :lines: 50-101

This naturally becomes a little more complicated, the main things to consider are

- A list of two dicts for the :code:`sets` (and it would similar for :code:`gets`).
- The :code:`order`-argument becomes a list. 
- Both methods are added as observer-methods using :code:`add_observer_method`.

The output in this case looks like so:: 

    |========================== Initialization starting ==========================|
    |================================= Observers =================================|
    |   Order 1 - Name: RandomGenerator.generate                                  |
    |   Order 2 - Name: TwoMethodObserver.observer_method_1                       |
    |   Order 2 - Name: LocalOptimizationEvaluator.evaluate                       |
    |   Order 3 - Name: Database.store_in_database                                |
    |   Order 4 - Name: TwoMethodObserver.observer_method_2                       |
    |   Order 5 - Name: Logger.report_logs                                        |
    |========================= Observers set/get reports =========================|
    |   RandomGenerator.generate                                                  |
    |       Sets 'candidates'                                                     |
    |   TwoMethodObserver.observer_method_1                                       |
    |       Gets 'candidates'                                                     |
    |   LocalOptimizationEvaluator.evaluate                                       |
    |       Gets 'candidates'                                                     |
    |       Sets 'evaluated_candidates'                                           |
    |   Database.store_in_database                                                |
    |       Gets 'evaluated_candidates'                                           |
    |   TwoMethodObserver.observer_method_2                                       |
    |       Gets 'evaluated_candidates'                                           |
    |   Logger.report_logs                                                        |
    |       Doesnt set/get anything                                               |
    |                                                                             |
    |   Overall:                                                                  |
    |   Get keys: {'evaluated_candidates', 'candidates'}                          |
    |   Set keys: {'evaluated_candidates', 'candidates'}                          |
    |   Key match: True                                                           |
    |========================== Initialization finished ==========================|

So the output reflects what we expected - that the first method gets :code:`candidates`
and the second method gets :code:`evaluated_candidates`. Any number of 
methods can be added in this way, and adding something to the :code:`State`-cache 
works in the same way. 

.. note::
    Generally this should be avoided unless it leads to a simpler implementation, 
    such as when saving an intermediate result as an attribute of the observer 
    in the first method that is then used by the second method. 