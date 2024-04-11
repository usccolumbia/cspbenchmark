Gets & Sets Basic Observer
===========================

Sometimes we will want an observer to actually change candidates, in which 
case we need to both get from and set to the cache. We can do this like so: 

.. literalinclude:: gets_sets_basic_observer.py
    :language: python
    :lines: 50-74

.. note:: 
    In this case the observer is doing what something that may be considered a 
    postprocessor, e.g. it does essentially the same thing as the :code:`CenteringPostProcessor`.
    Often it will easist to implement something if it can be implemented as 
    one of the common modules - usually avoids having to deal with :code:`gets`
    or :code:`sets` etc. 
