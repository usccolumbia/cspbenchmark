Gets Basic Observer
====================

We will often want to at least look at some of things that are going on 
during a search so we can update the :code:`BasicObserver` from the previous 
example to a :code:`GetsBasicObserver` that can do so. 

.. literalinclude:: gets_basic_observer.py
    :language: python
    :lines: 50-72

Compared to the :code:`BasicObserver` we have added :code:`gets={'get_key':'evaluated_candidates'}`
as an argument and we pass that argument onto :code:`Observer.__init__`. This allows 
the :code:`LessBasicObserver` to get from the dict stored on the :code:`State`-object, 
in this case it gets the evaluated candidates of the current iteration.
