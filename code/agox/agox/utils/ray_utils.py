import os
import joblib
import ray
import numpy as np

from timeit import default_timer as dt

from agox.writer import pretty_print, header_print

def ray_startup(cpu_count, memory, tmp_dir):
    """
    Start a Ray instance on Slurm.

    Parameters
    ----------
    cpu_count : int, optional
        Number of CPU cores to use, by default None in which case defaults are 
        read from SLURM.
    memory : int, optional
        Amount of memeory to use in bits, by default None in which case environment
        variables are used to guess a suitable amount. 
    """

    if ray.is_initialized():
        return

    # CPU Count:
    if cpu_count is None:
        try:
            cpu_count = int(os.environ['SLURM_NTASKS'])
        except:
            cpu_count = joblib.cpu_count()
    # Memory:
    if memory is None:
        try:
            memory=int(os.environ['SLURM_MEM_PER_NODE']*1e6)
        except:
            memory=cpu_count*int(2*1e9)

    if tmp_dir is None:
        path = os.getcwd()
        tmp_dir = os.path.join(path, 'ray')

        number_of_bytes = len(tmp_dir.encode('utf-8')) + 61

        if number_of_bytes > 107:
            tmp_dir = os.path.expanduser('~') + '/tmp/ray'
            print('USING USER ROOT FOLDER FOR RAY TEMP BECAUSE GIVEN OR DEFAULT GENERATED PATH IS TOO MANY BYTES.')
            print(f'Path: {tmp_dir}')
    
    ray.init(_memory=memory, object_store_memory=int(memory/4),
            num_cpus=cpu_count, ignore_reinit_error=True, _temp_dir=tmp_dir, 
            include_dashboard=False)
    print(ray.cluster_resources())

ray_kwarg_keys = ('tmp_dir', 'memory', 'cpu_count')

class RayBaseClass:

    def __init__(self, tmp_dir=None, memory=None, cpu_count=None):
        self.tmp_dir = tmp_dir
        self.memory = memory
        self.cpu_count = cpu_count

    def ray_startup(self):
        """
        Start a Ray instance on Slurm.

        Parameters
        ----------
        cpu_count : int, optional
            Number of CPU cores to use, by default None in which case defaults are 
            read from SLURM.
        memory : int, optional
            Amount of memeory to use in bits, by default None in which case environment
            variables are used to guess a suitable amount. 
        """
        cpu_count = self.cpu_count
        memory = self.memory 
        tmp_dir = self.tmp_dir

        # CPU Count:
        if cpu_count is None:
            try:
                cpu_count = int(os.environ['SLURM_NTASKS'])
            except:
                cpu_count = joblib.cpu_count()
        # Memory:
        if memory is None:
            try:
                memory=int(os.environ['SLURM_MEM_PER_NODE']*1e6)
            except:
                memory=cpu_count*int(2*1e9)

        if tmp_dir is None:
            path = os.getcwd()
            tmp_dir = os.path.join(path, 'ray')

            number_of_bytes = len(tmp_dir.encode('utf-8')) + 61

            if number_of_bytes > 107:
                tmp_dir = os.path.expanduser('~') + '/tmp/ray'
                print('USING USER ROOT FOLDER FOR RAY TEMP BECAUSE GIVEN OR DEFAULT GENERATED PATH IS TOO MANY BYTES.')
                print(f'Path: {tmp_dir}')
        
        if not ray.is_initialized():
            ray.init(_memory=memory, object_store_memory=int(memory/4),
                     num_cpus=cpu_count, ignore_reinit_error=True, _temp_dir=tmp_dir)
            print(ray.cluster_resources())

"""
The Ray Actor implementation has a three Class hierachy. 
- Actor: Simple implementation of a Ray actor, but rather general in its ability 
        to execute functions remotely. You should never need to interact directly 
        with an actor. 

- Pool: This is analagous to scheduler running in the background when using Ray 
        workers. It is given tasks to the Actors.

- RayPoolUser: Class that modules that make use of a pool instance should inherit 
        from as it implements a range of convenient functions. 

The fundamental limitation is that the same set of Actors must be used by all 
modules that wish to parallelize in some way. In this implementation all modules 
that used a Pool must therefore use the same pool. 
"""

@ray.remote
class Actor:
    
    def __init__(self):
        self.modules = {}

    def execute_function(self, fn, module_keys, *args, **kwargs):
        kwargs = self.set_seed(**kwargs)
        return fn(*[self.modules[key] for key in module_keys], *args, **kwargs)

    def set_seed(self, **kwargs):
        seed = kwargs.pop('pool_internal_seed', None)
        if seed is not None:
            np.random.seed(seed)
        return kwargs

    def add_module(self, module, key):
        self.modules[key] = module

    def remove_module(self, key):
        del self.modules[key]

from agox.observer import Observer
from agox.writer import Writer, agox_writer
from agox.module import Module

class Pool(Observer, Writer, Module):

    name = 'ParallelPool'

    def __init__(self, num_actors, **kwargs):
        Observer.__init__(self)
        Writer.__init__(self)
        Module.__init__(self)
        self.modules = {}
        self.idle_actors = [Actor.remote() for _ in range(num_actors)]
        self.number_of_actors = len(self.idle_actors)
        self.future_to_actor = {}
        self.pending_submits = []
        self.next_task_index = 0

        self.add_observer_method(self.update_pool_actors, sets={}, gets={}, order=100, handler_identifier='AGOX')

        self.attached_handlers = []

    ##########################################################################################################
    # Using the pool - mapping out jobs. 
    ##########################################################################################################

    def map(self, fn, module_keys, args_fn, kwargs_fn):
        """

        Performs the given function 'fn' in parallel over the pool of actors.

        Parameters
        ----------
        fn : function
            Function to execute on the actors. 
        module_keys : list
            Keys indicating which module on the actor is passed to the function.
        args_fn : list of lists. 
            Positional arguments passed to 'fn'. 
        kwargs_fn : list of dicts
            Keyword-arguments passed to 'fn'. 

        Returns
        -------
        list
            List of outputs such that fn(module, args[0], kwargs[0]) is the first 
            element of the list. 
        """

        for args, kwargs, module_key, in zip(args_fn, kwargs_fn, module_keys):
            kwargs['pool_internal_seed'] = np.random.randint(0, 10e6, size=1)
            self.submit(fn, module_key, args, kwargs)

        # Futures:
        done = False
        all_results = [None for _ in range(len(args_fn))]

        while not done:
            # Use Ray wait to get a result. 
            future_ready, _ = ray.wait(list(self.future_to_actor), num_returns=1)

            # Find the actor that was working on that future: 
            index, actor = self.future_to_actor.pop(future_ready[0])

            # Return the actor to the idle pool: 
            self.idle_actors.append(actor)

            # Now theres a free actor we re-submit:
            if len(self.pending_submits) > 0:
                fn, module_key, args, kwargs = self.pending_submits.pop(0) # Pop 0 preserves the order.
                self.submit(fn, module_key, args, kwargs)

            # Because the result is ready this is 'instant' and the next job is 
            # already queued.
            all_results[index] = ray.get(future_ready[0])

            if len(self.idle_actors) == self.number_of_actors:
                done = True

        # Reset the task index counter. 
        self.next_task_index = 0

        return all_results

    def submit(self, fn, module_keys, args, kwargs):
        """
        Submit tasks to the worker-pool. 

        Parameters
        ----------
        fn : function-handle
            Function to execute.
        module_keys : valid-key type (probably str is the best choice).
            Used to identify modules on the actor. 
        args : list
            Arguments for fn
        kwargs : dict
            Key-word arguments for fn.
        """

        if len(self.idle_actors) > 0: # Look to see if an idle actor exists.
            actor = self.idle_actors.pop() # Take an idle actor
            future = actor.execute_function.remote(fn, module_keys, *args, **kwargs) # Execute. 
            future_key = tuple(future) if isinstance(future, list) else future # Future as key
            self.future_to_actor[future_key] = (self.next_task_index, actor) # Save 
            self.next_task_index += 1 # Increment
        else: # If no idle actors are available we save the job for later. 
            # I am wondering if it would be helpful to put the args/kwargs now
            # Such that they are ready when an actor is available again. 
            self.pending_submits.append((fn, module_keys, args, kwargs))

    def execute_on_actors(self, fn, module_key, args, kwargs):
        """
        Execute the function _once_ per actor with the same data. 

        This could e.g. be used to set or get parameters. 

        Parameters
        ----------
        fn : function
            Function to execute once pr. actor. 
        module_key : list
            Modules to sue. 
        args : list
            Arguments to pass to fn.
        kwargs : dict
            Key-word argument to pass to fn.

        Returns
        -------
        list
            The results of the evaluating the fn. 
        """
        assert len(self.idle_actors) == self.number_of_actors        
        futures = []
        for actor in self.idle_actors:
            futures += [actor.execute_function.remote(fn, module_key, *args, **kwargs)]
        return ray.get(futures)

    ##########################################################################################################
    # Managing the pool.
    ##########################################################################################################

    def get_key(self, module):
        """_summary_

        Parameters
        ----------
        module : AGOX Module
            Module to get key for.

        Returns
        -------
        int
            The hash of the module
        """
        return hash(module)

    def add_module(self, module):
        """
        Parameters
        ----------
        module : AGOX module
            Module to add to the pool and its actors.            

        Returns
        -------
        int
           The key that has been generateed for the module. 
        """
        key = self.get_key(module)
        assert len(self.idle_actors) == self.number_of_actors
        futures = [actor.add_module.remote(module, key) for actor in self.idle_actors]
        ray.get(futures) # Block to make sure all the actors do this!
        self.modules[key] = module
        return key

    def remove_module(self, module):
        """
        Parameters
        ----------
        module : AGOX module
            Module to remove from the pool and its actors.            

        """

        key = self.get_key(module)
        assert len(self.idle_actors) == self.number_of_actors
        futures = [actor.remove_module.remote(key) for actor in self.idle_actors]
        ray.get(futures) # Block to make sure all the actors do this!
        del self.modules[key]

    @agox_writer
    @Observer.observer_method
    def update_pool_actors(self, state):
        """

        Responsible for 
        1. Updating module attributes on the pool and actors. 
        2. Interconnecting modules on the pool and its actors.

        The updating is handled by looking for which attributes of the supplied modules 
        are marked as dynamic_attributes. E.g. for a generic AGOX Module:

            from agox.module import Module
            class GenericObserverClass(Module):

                dynamic_attributes = ['parameters']

                def __init__(self):
                    self.parameters = [1, 2, 3]
                    self.static_parameters = [2, 2, 2]

                def update_parameters(self):
                    self.parameters[0] += 1

        In this case only the 'parameters' attribute will be updated as 'static_parameters' 
        are not referenced in dynamic_attributes. 

        We want to internal modules on the Actors to link to each other if they use
        each other.  

        If a module on the pool references another module on the pool, then they 
        should also be connected on the Actors. 
    
        Parameters
        ----------
        state : AGOX State object
            An instance of an AGOX State object. 
        """
        self.update_modules()

    def update_modules(self, writer=None):
        """
        Update dynamic attributes of modules on the actors. 
        """
        if writer is None:
            writer = self.writer

        tab = '   '
        for i, module in enumerate(self.modules.values()):
            attribute_dict = module.get_dynamic_attributes()
            t0 = dt()
            if len(attribute_dict) > 0:
                self.set_module_attributes(module, attribute_dict.keys(), attribute_dict.values())
            t1 = dt()
            writer(tab + f'{i}: {module.name} -> {len(attribute_dict)} updates in {t1-t0:04.2f} s')
            for j, attribute_name in enumerate(attribute_dict.keys()):
                writer(2 * tab + f'{j}: {attribute_name}')

    def update_module_interconnections(self):
        """
        Update module interconnections on actors. 

        This is done like so for each module in self.modules:
        1. Find all dynamic submodules recursively. 
        2. Update the connection on the Actor. 
        """
        pretty_print('Making module interconnections on actors')

        def interconnect_module(module, reference_module, setting_key):
            module.set_for_submodule(setting_key, reference_module)

        count = 0
        t0 = dt()
        tab = '   '
        for module_key, module in self.modules.items():
            submodules = module.find_submodules(only_dynamic=True)
            for setting_key, submodule in submodules.items():
                sub_module_key = self.get_key(submodule)

                modules = [module_key, sub_module_key]
                args = [setting_key]

                pretty_print(tab + f'{count}: Connected {module.name} with {submodule.name}')
                att_name = '.'.join(setting_key)
                pretty_print(2*tab + f'  Attribute name: {att_name}')
                self.execute_on_actors(interconnect_module, modules, args, {})
                count += 1
        if count == 0:
            pretty_print('No module interconnections found!')
        t1 = dt()
        pretty_print(f'Interconnecting time: {t1-t0:04.2f}')

    def print_modules(self, writer=None):
        if writer is None:
            writer = self.writer
        
        tab = '   '
        pretty_print('Modules in pool')
        for i, (key, module) in enumerate(self.modules.items()):
            num_dynamic = len(module.dynamic_attributes)
            report_str = tab + f'{i}: ' + module.name + f' - Attrs. = {num_dynamic}'
            writer(report_str)

    def get_module_attributes(self, module, attributes):
        """
        Get one or more attributes of a module on the Actors of the pool. 

        Parameters
        ----------
        module : AGOX module
            Module to get attributes from. 
        attributes : list of str
            Names of the attributes to retrieve. 

        Returns
        -------
        list of dicts
            A list containing the dicts that hold the requested attributes. 
            The list has length equal to the number of actors and the dicts 
            have length equal to the number of requested attributes. 
        """
        def get_attributes(module, attributes):
            return {attribute:module.__dict__.get(attribute, None) for attribute in attributes}

        return self.execute_on_actors(get_attributes, [self.get_key(module)], [attributes], {})

    def set_module_attributes(self, module, attributes, values):
        """
        Set attributes of a module on the actors of the pool. 

        Parameters
        ----------
        module : AGOX module
            Module to get attributes from. 
        attributes : list of str
            Names of the attributes to set. 
        values : list
            Values to set for each attribute. 
        """
        def set_attributes(module, attributes, values):
            for value, attribute in zip(values, attributes):
                module.__dict__[attribute] = value
        
        self.execute_on_actors(set_attributes, [self.get_key(module)], [attributes, values], {})
      
class RayPoolUser(Observer):

    kwargs = ['pool', 'tmp_dir', 'memory', 'cpu_count']

    def __init__(self, pool=None, tmp_dir=None, memory=None, cpu_count=None):
        ray_startup(cpu_count, memory, tmp_dir)
        self.pool = get_ray_pool() if pool is None else pool

    def pool_add_module(self, module, include_submodules=True):
        """
        Parameters
        ----------
        module : AGOX module 
            The module being added to the pools modules. 

        Returns
        -------
        int
            The key the pool uses for the module. 
        """

        if include_submodules:
            submodules = module.find_submodules(only_dynamic=True)
            for submodule in submodules.values():
                self.pool_add_module(submodule)

        return self.pool.add_module(module)

    def pool_remove_module(self, module):
        """
        Parameters
        ----------
        module : AGOX module 
            The module being added to the pools modules. 
        """
        self.pool.remove_module(module)

    def pool_get_key(self, module):
        """_summary_

        Parameters
        ----------
        module : AGOX module.
            The module for which to retrieve the key. 

        Returns
        -------
        int
            Key used for indexing to the module in the pool and on the actors.  
        """
        return self.pool.get_key(module)

    def pool_map(self, fn, modules, args, kwargs):
        """
        This is method that does the parallelization. 

        Parameters
        ----------
        fn : function
            The function to execute in parallel. 
        modules : list
            Keys for modules used by fn. 
        args : list
            List of positional arguments for each call of the function. 
        kwargs : list
            List of keyword arguments for the functions. 

        The function 'fn' is executed once pr. set of modules, args and kwargs.
        So these lists must have the same length, and that length is equal to 
        the number of parallel tasks - which does not need to equal the number 
        of actors in the pool.

        As an example: 

            def fn(calculator, atoms):
                atoms.set_calculator(calculator)
                return E = atoms.get_potential_energy()
            
            modules = [[model_key], [model_key]]
            args = [[atoms1], [atoms2]]
            kwargs = [{}, {}]

            E = pool_map(fn, modules, args, kwargs)

        Calculates the energy of two atoms objects in parallel with the calculator 
        that is assumed to be a module on the Actors. 

        The args and kwargs may also contain ObjectRefs obtained from ray.put.

        Returns
        -------
        list
            List of the output obtained by applying the function fn to each set 
            of modules, args, kwargs. 
        """
        return self.pool.map(fn, modules, args, kwargs)

    def pool_get_module_attributes(self, module, attributes):
        """

        Get one or more attributes of a module on the Actors of the pool. 

        Parameters
        ----------
        module : AGOX module
            Module to get attributes from. 
        attributes : list of str
            Names of the attributes to retrieve. 

        Returns
        -------
        list of dicts
            A list containing the dicts that hold the requested attributes. 
            The list has length equal to the number of actors and the dicts 
            have length equal to the number of requested attributes. 
        """
        return self.pool.get_module_attributes(module, attributes)
    
    def pool_set_module_attributes(self, module, attributes, values):
        """
        Set attributes of a module on the actors of the pool. 

        Parameters
        ----------
        module : AGOX module
            Module to get attributes from. 
        attributes : list of str
            Names of the attributes to set. 
        values : list
            Values to set for each attribute. 
        """
        self.pool.set_module_attributes(module, attributes, values)

    def attach(self, handler):
        """
        This method helps getting the pool attached to AGOX without needing 
        to do so explicitly in a runscript. 
    
        Parameters
        ----------
        handler : AGOX ObserverHandler instance
            The observer handler to attach to. 
        """
        super().attach(handler)
        if hash(handler) not in self.pool.attached_handlers:
            self.pool.attach(handler)
            header_print('Parallel Pool Initialization')
            self.pool.print_modules(writer=pretty_print)
            self.pool.update_module_interconnections()
            self.pool.update_modules(writer=pretty_print)
            self.pool.attached_handlers.append(hash(handler))

    def get_pool(self):
        return self.pool

global ray_pool_instance
ray_pool_instance = None

def make_ray_pool():
    ray_pool_instance = Pool(num_actors=int(ray.cluster_resources()['CPU']))
    return ray_pool_instance

def get_ray_pool():
    global ray_pool_instance
    if ray_pool_instance is None:
        ray_pool_instance = make_ray_pool()
    return ray_pool_instance