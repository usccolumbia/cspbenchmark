Modularity
===========

AGOX is founded on the idea of modularity in global search algorithms, recognizing 
that optimization algorithms can be expressed in terms of building blocks 
that are reused for many different algorithms. Most search algorithms are 
an iterative procedure consisting of generating candidates and evaluating them.  

AGOX implements abstract base classes (ABCs) for these modules, which currently 
are: 

* :code:`Candidate`: An extension of the ASE atoms object that holds the properties, e.g. 
  positions and atomic numbers of trial solutions. This is the type of object 
  the other modules act on or create. 
* :code:`Environment`: Defines the search settings, e.g. the amount and species of atoms, 
  any already present atoms, the computational cell and any constraints. 

The remaining modules make, interact with or act on candidates:

* :code:`Generators`: Generates candidate structures. 
* :code:`Evaluators`: Evaluates the target property, such as the total energy. 
* :code:`Collectors`: Collects generated candidates for algorithms that used multiple 
  candidates per. iteration 
* :code:`Samplers``: Determines seed structures for generators, such as the population 
  in an evolutionary algorithms or Metropolis accepted structure in basin-hopping. 
* :code:`Postprocessors`: Postprocesss generated candidates, this is an extension of 
  the generation procedure, e.g. to center the candidate structures to the center 
  of the cell or locally optimize in a surrogate potential. 
* :code:`Acquisitors`: Some algorithms, such as GOFEE, may select which candidates are 
  selected for evaluation by applying an acquisition function. 
* :code:`Models`: These are machine-learning models, such as a Gaussian Process surrogate 
  energy model. 
* :code:`Database`: Stores the evaluated structures so they can be analyzed after a search 
  and for training models. 

Abstract Base Classes
_____________________

The ABC of each modules defines the skeleton of that module, specifically it 
tells us which methods must be implemented to create a version of the module. 
This is done by the :code:`@abstractmethod` decorator, telling us that 
a method of that name has to be implemented in order to use the module. 

As an example, we can look at the ABC for generator modules, :code:`ABC_generator.py`
where we will find two methods decorated with :code:`@abstractmethod`. 

.. literalinclude:: ../../../agox/generators/ABC_generator.py
    :linenos:
    :lineno-start: 63
    :lines: 63-70

Which are the only two methods that we have to implement to write to make a 
generator. 

