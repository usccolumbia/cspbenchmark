Data-flow
==========

In the previous section we explored how the observer pattern can be used 
to put together the modular pieces of the algorithm. In this section 
we will look at how the modules can communicate without imposing limitations 
on the allowed order of execution. 

As a first example we will again take a look at the generator ABC and its 
:code:`generate`-method

.. literalinclude:: ../../../agox/generators/ABC_generator.py
    :linenos:
    :lineno-start: 135
    :lines: 135-143
    :emphasize-lines: 6

In particular this method calls a method :code:`add_to_cache` parsing along 
an attribute :code:`self.set_key` and a list of generated :code:`candidates`. 
This attribute is defined by the initialization, 

.. literalinclude:: ../../../agox/generators/ABC_generator.py
    :linenos:
    :lineno-start: 17
    :lines: 17-21

In particular we are interested in the argument 

.. code-block:: python 

    sets={'set_key':'candidates'}

Which is parsed along to the Observer class that the generator ABC inherits from 
where it is turned into an attribute :code:`self.set_key = 'candidates'`. So, the call 
to :code:`add_to_cache` is given the string :code:`'candidates'`, which it uses a key 
to a dictionary where it stores the list that was also given as an argument. 
This may all seem rather complicated, but there are benefits - one of which is that 
the program is able to tell us beforehand what each observer has told it it will 
set or get, for the script from the Getting Started section this looks like so:: 

    |========================= Observers set/get reports =========================|
    |   RandomGenerator.generate                                                  |
    |       Sets 'candidates'                                                     |
    |   LocalOptimizationEvaluator.evaluate_candidates                            |
    |       Gets 'candidates'                                                     |
    |       Sets 'evaluated_candidates'                                           |
    |   Database.store_in_database                                                |
    |       Gets 'evaluated_candidates'                                           |
    |   Logger.report_logs                                                        |
    |       Doesnt set/get anything                                               |
    |                                                                             |
    |   Overall:                                                                  |
    |   Get keys: {'evaluated_candidates', 'candidates'}                          |
    |   Set keys: {'evaluated_candidates', 'candidates'}                          |
    |   Key match: True                                                           |
    ===============================================================================

Which tells us that

1. The RandomGenerator sets something with a key candidates. 
2. The LocalOptimizationEvaluator gets this and then sets with the key evaluated_candidates
3. The database gets with the key evaluate_candidates. 
4. The logger does not get or set anything. 

It also compares the set and get keys and tells us if they match, make sure to 
check this when building new algorithms - if they dont match it is very likely 
that the script is not doing as intended. 

As an example we could change what key the RandomGenerator is setting with, 
this can be done in the script like so

.. code-block:: python

    random_generator = RandomGenerator(**environment.get_confinement(), 
        environment=environment, may_nucleate_at_several_places=True, 
        order=1, sets={'set_key':'not_candidates'})


Which would result in this print out::

    |========================= Observers set/get reports =========================|
    |   RandomGenerator.generate                                                  |
    |       Sets 'not_candidates'                                                 |
    |   LocalOptimizationEvaluator.evaluate_candidates                            |
    |       Gets 'candidates'                                                     |
    |       Sets 'evaluated_candidates'                                           |
    |   Database.store_in_database                                                |
    |       Gets 'evaluated_candidates'                                           |
    |   Logger.report_logs                                                        |
    |       Doesnt set/get anything                                               |
    |                                                                             |
    |   Overall:                                                                  |
    |   Get keys: {'evaluated_candidates', 'candidates'}                          |
    |   Set keys: {'evaluated_candidates', 'not_candidates'}                      |
    |   Key match: False                                                          |
    |   Sets do not match, this can be problematic!                               |
    |   Umatched keys ['candidates', 'not_candidates']                            |
    |========================== Initialization finished ==========================|

And a not working algorithm - but it does show case that we can control how 
data flows through an algorithm. 

Lets try an example without breaking the algorithm, we could move the candidates 
to the center of the computational cell before evaluation, as this may help 
with the local optimization, so we add a Postprocessors

.. code-block:: python

    centering = CenteringPostProcess(gets={'get_key':'candidates'}, sets={'set_key':'centered_candidates'}, order=2)

and change the :code:`gets` of the evaluator

.. code-block:: python

    evaluator = LocalOptimizationEvaluator(calc, gets={'get_key':'centered_candidates'}, ...)

Now the output looks like so:: 

    |========================= Observers set/get reports =========================|
    |   RandomGenerator.generate                                                  |
    |       Sets 'candidates'                                                     |
    |   CenteringPostProcess.postprocess_candidates                               |
    |       Gets 'candidates'                                                     |
    |       Sets 'centered_candidates'                                            |
    |   LocalOptimizationEvaluator.evaluate_candidates                            |
    |       Gets 'centered_candidates'                                            |
    |       Sets 'evaluated_candidates'                                           |
    |   Database.store_in_database                                                |
    |       Gets 'evaluated_candidates'                                           |
    |   Logger.report_logs                                                        |
    |       Doesnt set/get anything                                               |
    |                                                                             |
    |   Overall:                                                                  |
    |   Get keys: {'candidates', 'evaluated_candidates', 'centered_candidates'}   |
    |   Set keys: {'candidates', 'evaluated_candidates', 'centered_candidates'}   |
    |   Key match: True                                                           |
    ===============================================================================

Which tell us that now the generator generates candidates, that the postprocessor then 
centers before the evaluator does its thing. Without changing any of the underlying c
code we have added an additional module and changed the way data flows through 
the program.