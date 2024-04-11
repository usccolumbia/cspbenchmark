import numpy as np
from abc import ABC, abstractmethod

from agox.candidates import StandardCandidate
from agox.writer import header_print, pretty_print
from agox.module import Module
import functools
from copy import copy

global A
A = 0
def get_next_key():
    """
    Generates a unique always increasing key for observer methods. 

    Returns
    --------
    int: 
        Unique key. 
    """
    global A
    A += 1
    return A

class ObserverHandler:
    """
    Base-class for classes that can have attached observers. 
    """

    def __init__(self, handler_identifier, dispatch_method):
        self.observers = {}
        self.execution_sort_idx = []
        self.handler_identifier = handler_identifier
        self.set_dispatch_method(dispatch_method)

    ####################################################################################################################
    # General handling methods.
    ####################################################################################################################

    def attach_observer(self, observer_method):
        """
        Attaches an observer, saving it to the internal dict and re-evaluating 
        the execution order. 

        Parameters
        -----------
        observer_method: object
            An instance of the ObserverMethod class that will be attached. 
        """
        self.observers[observer_method.key] = observer_method
        self.evaluate_execution_order()

    def delete_observer(self, observer_method):
        """
        Removes an observer, deleting it from the internal dict and re-evaluating 
        the execution order. 

        Parameters
        -----------
        observer_method: object
            An instance of the ObserverMethod class that will be attached. 
        """
        del self.observers[observer_method.key]
        self.evaluate_execution_order()

    def evaluate_execution_order(self):
        """
        Evaluates the execution order by sorting the orders of stored 
        observers. 
        """
        keys = self.observers.keys()
        orders = [self.observers[key]['order'] for key in keys]
        self.execution_sort_idx = np.argsort(orders)

    def get_observers_in_execution_order(self):
        """
        Returns
        --------
        List
            Observers sorted in order of execution.
        """
        observers = [obs for obs in self.observers.values()]
        sort_idx = self.execution_sort_idx
        return [observers[p]['method'] for p in sort_idx]

    def dispatch_to_observers(self, *args, **kwargs):
        """
        Dispatch to observers. 

        Only rely on the order of execution if you have specified the 'order' argument for each observer. 
        """
        for observer_method in self.get_observers_in_execution_order():
            observer_method(*args, **kwargs)

    def set_dispatch_method(self, method):
        self.dispatch_method = self.name + '.' + method.__name__

    ####################################################################################################################
    # Printing / Reporting 
    ####################################################################################################################

    def print_observers(self, include_observer=False, verbose=0, hide_log=True):
        """
        Parameters
        -----------
        include_observer: bool (default: False)
            Turns printing of observer name on. Might be useful to debug if 
            something is not working as expected. 
        verbose: int (default: 0)
            If verbose > 1 then more information is printed. 
        hide_log: bool (default: True)
            If True then the LogEntry observers are not printed. Keeps the 
            report more clean. 
        """
        order_indexs = self.execution_sort_idx
        keys = [key for key in self.observers.keys()]
        names = [obs['name'] for obs in self.observers.values()]
        methods = [obs['method'] for obs in self.observers.values()]
        orders = [obs['order'] for obs in self.observers.values()]
        
        base_string = '{}: order = {} - name = {} - method - {}'
        if include_observer:
            base_string += ' - method: {}'

        #start_string = '|'+'=' * 33 + ' Observers ' + '='*33+'|'
        header_print('Observers')
        for idx in order_indexs:
            
            if hide_log and 'LogEntry.' in names[idx]:
                continue

            pretty_print('  Order {} - Name: {}'.format(orders[idx], names[idx]))
            if verbose > 1:
                pretty_print('  Key: {}'.format(keys[idx]))
                pretty_print('  Method: {}'.format(methods[idx]))
                pretty_print('_'*50)

    def observer_reports(self, report_key=False, hide_log=True):
        """
        Generate observer report, which checks if the data flow is valid. 

        Parameters
        -----------
        report_key: bool
            Whether or not to print the keys used to get or set from the cache. 
        hide_log: bool
            Whether or not to print the LogEntry observers. 
        """
        dicts_out_of_order = [value for value in  self.observers.values()]

        
        header_print('Observers set/get reports')

        base_offset = '  '
        extra_offset = base_offset + '    '        
        for i in self.execution_sort_idx:

            observer_method = dicts_out_of_order[i]
            if 'LogEntry' in observer_method.name and hide_log:
                continue

            pretty_print(base_offset + observer_method.name)
            report = observer_method.report(offset=extra_offset, report_key=report_key, print_report=False, return_report=True)
            for string in report:
                pretty_print(string)
            else:
                #print(f"{dicts_out_of_order['name']} cannot report on its behavior!")
                pass

        get_set, set_set = self.get_set_match()
        pretty_print(base_offset)
        pretty_print(base_offset+'Overall:')
        pretty_print(base_offset+f'Get keys: {get_set}')
        pretty_print(base_offset+f'Set keys: {set_set}')
        pretty_print(base_offset+f'Key match: {get_set==set_set}')
        if not get_set==set_set:
            pretty_print(base_offset+'Sets do not match, this can be problematic!')
            if len(get_set) > len(set_set):
                pretty_print(base_offset+'Automatic check shows observers will attempt to get un-set item!')
                pretty_print(base_offset+'Program likely to crash!')
            if len(set_set) > len(get_set):
                pretty_print(base_offset+'Automatic check shows observers set value that is unused!')
                pretty_print(base_offset+'May cause unintended behaviour!')

            unmatched_keys = list(get_set.difference(set_set))+list(set_set.difference(get_set))
            pretty_print(base_offset+f'Umatched keys {unmatched_keys}')

    def get_set_match(self):
        """
        Check if gets and sets match.
        """
        dicts_out_of_order = [value for value in  self.observers.values()]
        all_sets = []
        all_gets = []

        for observer_method in dicts_out_of_order:
            all_sets += observer_method.sets.values()
            all_gets += observer_method.gets.values()

        all_sets = set(all_sets)
        all_gets = set(all_gets)

        return all_gets, all_sets
            
class FinalizationHandler:
    """
    Just stores information about functions to be called when finalizaing a run. 
    """

    def __init__(self):
        self.finalization_methods = {}
        self.names = {}

    def attach_finalization(self, name, method):
        """
        Attaches finalization method.

        Parameters
        -----------
        name: str
            Human readable name of the attached method.
        method: method
            A method or function to attach. 
        """
        key = get_next_key()
        self.finalization_methods[key] = method
        self.names[key] = name
    
    def get_finalization_methods(self):
        """
        Returns
        --------
        List
            List of finalization methods.
        """
        return self.finalization_methods.values()

    def print_finalization(self, include_observer=False, verbose=0):
        """
        include_observer flag might be useful to debug if something is not working as expected. 
        """

        names = [self.names[key] for key in self.finalization_methods.keys()]
        
        base_string = '{}: order = {} - name = {} - method - {}'
        if include_observer:
            base_string += ' - method: {}'

        print('=' * 24 + ' Finalization ' + '='*24)
        for name in names:
            print('Name: {}'.format(name))
        print('='*len('=' * 25 + ' Observers ' + '='*25))
    
class Observer(Module):

    def __init__(self, gets=[dict()], sets=[dict()], order=[0], surname='', **kwargs):
        """
        Base-class for classes that act as observers. 

        Parameters
        -----------
        gets: dict
        Dict where the keys will be set as the name of attributes with the 
        value being the value of the attribute. Used to get something from the 
        iteration_cache during a run. 

        sets: dict
        Dict where the keys will be set as the name of attributes with the 
        value being the value of the attribute. Used to set something from the 
        iteration_cache during a run. 

        order: int/float
        Specifies the (relative) order of when the observer will be executed, 
        lower numbers are executed first. 

        sur_name: str
            An additional name added to classes name, can be used to distinguish 
            between instances of the same class. 
        """
        Module.__init__(self, surname=surname)

        if type(gets) == dict:
            gets = [gets]
        if type(sets) == dict:
            sets = [sets]
        if type(order) == int or type(order) == float:
            order = [order]

        combined = dict()
        for tuple_of_dicts in [gets, sets]:
            for dict_ in tuple_of_dicts:
                for key, item in dict_.items():
                    combined[key] = item
        #self.__dict__ = combined
        for key, value in combined.items():
            if key not in self.__dict__.keys():
                self.__dict__[key] = value
            else:
                raise BadOptionError('{}-set or get key has the same name as an attribute, this is not allowed'.format(key))

        self.set_keys = sum([list(set_dict.keys()) for set_dict in sets], [])
        self.set_values = sum([list(set_dict.values()) for set_dict in sets], [])
        self.get_keys = sum([list(get_dict.keys()) for get_dict in gets], [])
        self.get_values = sum([list(get_dict.values()) for get_dict in gets], [])
        self.gets = gets
        self.sets = sets

        self.order = order

        self.observer_methods = {}

        if len(kwargs) > 0:
            print('Unused key word arguments supplied to {}'.format(self.name))
            for key, value in kwargs.items():
                print(key, value)

        self.iteration_counter = None
        self.candidate_instanstiator = StandardCandidate
        
    def get_from_cache(self, key):
        """
        Attempts to get from the cache, but it is checked that the class instance 
        that tries to get has registered that it will get something with the 
        given key. 

        Parameters:
        -----------
        key: str
            The key used to index into the cache. 
        """
        assert key in self.get_values # Makes sure the module has said it wants to get with this key. 
        return self.main_get_from_cache(key)

    def add_to_cache(self, key, data, mode):
        """
        Attempts to add to the cache, but it is checked that the class instance 
        that tries to add has registered that it will add something with the 
        given key. 

        Parameters:
        -----------
        key: str
            The key used to index into the cache. 
        """
        assert key in self.set_values # Makes sure the module has said it wants to set with this key. 
        return self.main_add_to_cache(key, data, mode)

    def add_observer_method(self, method, sets, gets, order, handler_identifier='any'):
        """
        Adds an observer method that will later be attached with 'self.attach'. 

        Parameters
        -----------
        method: method
            The function/method that will be called by the observer-loop.
        sets: dict
            Dict containing the keys that the method will set in the cache. 
        gets: dict
            Dict containing the keys that the method will get from the cache. 
        """
        observer_method = ObserverMethod(self.__name__, method.__name__, method, gets, sets, order, handler_identifier)
        self.observer_methods[observer_method.key] = observer_method

    def remove_observer_method(self, observer_method):
        """
        Remove an observer_method from the internal dict of methods, if this is 
        done before calling 'attach' it will not be added as an observer. 

        Parameters
        -----------
        observer_method: object
            An instance of ObserverMethod.
        """
        key = observer_method.key
        if key in self.observer_methods.keys():
            del self.observer_methods[key]

    def update_order(self, observer_method, order):
        """
        Change the order of a method. Not really tested in practice. 

        Parameters
        -----------
        observer_method: object
            An instance of ObserverMethod.

        order: float
            New order to set.         
        """
        key = observer_method.key
        assert key in self.observer_methods.keys()
        self.observer_methods[key].order = order

    def attach(self, handler):
        """
        Attaches all ObserverMethod's as to observer-loop of the ObserverHandler 
        object 'main'
        
        Parameters
        ----------
        handler: object
            An instance of an ObserverHandler. 
        """
        for observer_method in self.observer_methods.values():
            if observer_method.handler_identifier == 'any' or observer_method.handler_identifier == handler.handler_identifier:
                handler.attach_observer(observer_method)

    def reset_observer(self):
        """
        Reset the observer_methods dict, removing all observer methods.
        """
        self.observer_methods = {}

    ############################################################################
    # State
    ############################################################################

    def observer_method(func):
        @functools.wraps(func)
        def wrapper(self, state, *args, **kwargs):
            self.state_update(state)
            return func(self, state, *args, **kwargs)
        return wrapper
            
    def state_update(self, state):
        self.set_iteration_counter(state.get_iteration_counter())
    
    ############################################################################
    # Iteration counter methods:
    ############################################################################

    def get_iteration_counter(self):
        return self.iteration_counter
    
    def set_iteration_counter(self, iteration_count):
        self.iteration_counter = iteration_count

    def check_iteration_counter(self, count):
        if self.iteration_counter is None:
            return True
        return self.iteration_counter >= count

    ############################################################################
    # Checker
    ############################################################################

    def do_check(self, **kwargs):
        """
        Check if the method will run or just do nothing this iteration. 

        Returns
        -------
        bool
            True if function will run, False otherwise.w
        """
        return True

    ############################################################################
    # Candidate instanstiator
    ############################################################################

    def set_candidate_instanstiator(self, candidate_instanstiator):
        self.candidate_instanstiator = copy(candidate_instanstiator)

class ObserverMethod:

    def __init__(self, class_name, method_name, method, gets, sets, order, handler_identifier):
        """
        ObserverMethod class. Holds all information about the method that an 
        observer class attaches to the observer-loop. 

        Parameters
        ----------
        name: str
            Name of the class
        method_name: str
            Name of the method.
        method: method
            The method that is called by the observer-loop.
        sets: dict
            Dict containing the keys that the method will set in the cache. 
        gets: dict
            Dict containing the keys that the method will get from the cache. 
        order: float
            Execution order
        """
        self.class_name = class_name
        self.method_name = method_name
        self.method = method
        self.gets = gets
        self.sets = sets
        self.order = order

        self.name = self.class_name + '.' + self.method_name
        self.class_reference = method.__self__
        #self.key = method.__hash__() # Old
        self.key = get_next_key()
        self.handler_identifier = handler_identifier

    def __getitem__(self, key):
        return self.__dict__[key]

    def report(self, offset='', report_key=False, return_report=False, print_report=True):
        """
        Generate report. Used by ObserverHandler.observer_reports

        Parameters
        ----------
        offset: str
        report_key: bool
            Print the key or not. 
        return_report: bool
            Whether to return the report or not.
        print_report: bool
            Whether to print hte print or not. 

        Returns
        -------
        str or None:
            If report_key = True then it returns a string otherwise None is r
            eturned. 
        """
        report = []

        for key in self.gets.keys():
            value = self.gets[key]
            out = offset + f"Gets '{value}'" 
            if report_key:
                out += f" using key attribute '{key}'"
            report.append(out)

        for key in self.sets.keys():
            value = self.sets[key]
            out = offset + f"Sets '{value}'"
            if report_key:
                out += f" using key attribute '{key}'"
            report.append(out)
        
        if len(self.gets) == 0 and len(self.sets) == 0:
            out = offset+'Doesnt set/get anything'
            report.append(out)

        if print_report:
            for string in report:
                print(string)
        if return_report:
            return report

if __name__ == '__main__':

    # class TestObserverObject:

    #     def __init__(self, name):
    #         self.name = name

    #     def test_observer_method(self):
    #         #print('Name {} - I am method {} on {}'.format(id(self.test_observer_method), id(self)))
    #         print('My name is {}'.format(self.name))

    # handler = ObserverHandler()

    # A = TestObserverObject('A')
    # B = TestObserverObject('B')

    # handler.add_observer(A.test_observer_method,'A', order=0)
    # handler.add_observer(B.test_observer_method,'B', order=0)
    # #handler.add_observer(B.test_observer_method,'B', order=-1)

    # for method in handler.get_observers_in_execution_order():
    #     method()

    #for key in handler.observers.keys():
    #    print(key, handler.observers[key])        

    # print(dir(B.test_observer_method))
    # print(B.test_observer_method.__self__)

    observer = Observer(get_key='candidates', set_key='candidates', bad_key='bad')
    
    print(observer.__dict__)

    observer.report()


        
    