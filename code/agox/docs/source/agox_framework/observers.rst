Observers
==========

We need modules to be executed in a specific order to build a specific algorithm, 
but generally we do not want to impose any general limits on the order in 
which modules can be used. This is handled by an observer-pattern that the main 
iterative-loop of :code:`main.py` uses

.. literalinclude:: ../../../agox/main.py
    :lines: 103-113

Which calls the method :code:`get_observers_in_execution_order()`. 

Observers are added to this loop in the search script, rss-script defined 
in the Getting Started section had three observers: 

.. literalinclude:: ../getting_started/rss_script_slurm.py
    :lines: 44-45

.. literalinclude:: ../getting_started/rss_script_slurm.py
    :lines: 47-49

.. literalinclude:: ../getting_started/rss_script_slurm.py
    :lines: 37-38

Which were given to the :code:`AGOX` object, note that any number of 
modules can be passed in this way and order in which they are given is not important

.. literalinclude:: ../getting_started/rss_script_slurm.py
    :lines: 55

The keyword argument :code:`order` decides the order of execution of these 
three modules, so from the script we expect the following to happen

1. The generator does something. 
2. The evaluator does something. 
3. The database does something. 

When executing the script the program tells us what it will do:: 

    |================================= Observers =================================|
    |   Order 1 - Name: RandomGenerator.generate                                  |
    |   Order 2 - Name: LocalOptimizationEvaluator.evaluate_candidates            |
    |   Order 3 - Name: Database.store_in_database                                |
    |   Order 4 - Name: Logger.report_logs                                        |

Which lucky agrees with our prediction from looking at the script, with the addition 
with andditional observer, 'Logger.report_logs' which is added by default to time 
the execution speed of other observers. The print-out also indicates more about 
what the observers are actually doing than our previous guess, but to really understand 
we are going to have to dive a bit deeper into the code. 

All modules that can act as an observer inherit from the :code:`Observer`-class, 
we can again take a look at the generator ABC to see this 

.. literalinclude:: ../../../agox/generators/ABC_generator.py
    :linenos:
    :lineno-start: 17
    :lines: 17-28

So the order argument, along with some additional ones that we will discuss in the 
next section are passed to the observer class. In order to understand why the 
generate method is the one being reported on we have to look a little bit further 

.. literalinclude:: ../../../agox/generators/ABC_generator.py
    :linenos:
    :lineno-start: 60
    :lines: 60-61

Which is saying that the :code:`generate` method should be called as an observer 
with the given order. If we take a look at this method 

.. literalinclude:: ../../../agox/generators/ABC_generator.py
    :linenos:
    :lineno-start: 186
    :lines: 186-194

Which indeed uses the :code:`get_candidates` method that we saw in the previous 
section that we were obliged to write. The :code:`if` statement ensures that 
the generator can procedure a candidate even if a :code:`Sampler` is not ready. 
The :code:`add_to_cache`-call is part how modules communicate, which is the topic 
of the next section. In general all ABCs have methods such as this one implemented 
that define what the module will do when attached as an observer, these methods 
may be overwritten in specific implementations or other methods may be attached 
instead or in addition using the :code:`add_observer_method`-syntax. 


