import numpy as np
from agox.observer import Observer, ObserverHandler
from timeit import default_timer as dt
import pickle
from agox.writer import agox_writer, Writer
from agox.module import Module

"""
File contains three class
- Log
- LogEntry
- Logger

The relation is the 'Logger' is the class that is attached to AGOX (from main.py)
it holds a 'Log' and the 'Log' uses 'LogEntry' observers to time other observers. 

Logger (1) --> Log (1) --> LogEntry (Many).

The Log is saved to disk during a search. 
"""
class Log(Writer):

    name = 'Log'

    def __init__(self):
        """
        Log object. 

        Takes no arguments. 
        """        
        Writer.__init__(self, verbose=True, use_counter=False)
        self.entries = {}

    def add_entry(self, observer_method):
        """
        Adds an instance of the LogEntry class to self.entries.

        Parameters:
        -----------
        observer_method
            An instance of the ObserverMethod found in agox/observer.py.
        """
        self.entries[observer_method.key] = LogEntry(observer_method)

    def __getitem__(self, item):
         return self.entries[item]

    def log_report(self):
        """
        Method that prints the log by calling all LogEntries. 

        Parameters:
        ------------
        None

        Returns
        --------
        None
        """
        total_time = np.sum([entry.get_current_timing() for entry in self.entries.values()])
        self.writer('Total time {:05.2f} s '.format(total_time))

        for entry in self.entries.values():
            report = entry.get_iteration_report()
            for line in report:
                self.writer(line)

    def save_logs(self):
        """
        Save log to pickle, by saving all entries.
        """
        with open('log.pckl', 'wb') as f:
            pickle.dump(self.entries, f)
    
    def restore_logs(self, pickle_path):
        """
        Restore the log by loading pickle file. 

        Parameters
        -----------
        pickle_path: str
            Path to pickle file. 
        """
        with open(pickle_path, 'rb') as f:
            self.entries = pickle.load(f)

    def plot_log(self, ax):
        """
        Plot the log timings as a function of iterations. 

        Parameters
        -----------
        ax: matplotlib axis object. 
            Axis to plot in. 

        Returns: 
        ---------
        ax: matplotlib axis object.
        """
        for key, entry in self.entries.items():
            ax.plot(entry.timings, label=entry.name[10:])

        ax.set_xlim([0, len(entry.timings)])
        ax.legend()
        ax.set_xlabel('iteration [#]', fontsize=12)
        ax.set_ylabel('Timing [s]', fontsize=12)
        return ax

    def get_total_iteration_time(self):
        """
        Get the total time iteration time as

        Parameters
        -----------
        None

        Returns
        --------
        np.array
            Shape (?) # Should find out what the shape of this is! 
        """
        timings = []
        for entry in self.entries.values():
            timings.append(entry.timings)

        return np.mean(timings, axis=0)

class LogEntry(Observer, Writer, Module):

    name = 'LogEntry'

    def __init__(self, observer_method):
        """
        LogEntry class. 
        
        Given an ObserverMethod the LogEntry attaches observers around that method 
        in the observer-loop to time the ObserverMethod. 

        If the Observer that ObserverMethod comes from is an instance of an 
        ObserverHandler it rescursively attaches other LogEntry instances to 
        the observers of that ObserverHandler to time those too. 

        Parameters
        -----------
        observer_method: ObserverMethod object. 
            An instance of a ObserverMethod (agox/observer.py) to attach around. 
        """

        Observer.__init__(self, order=observer_method.order)
        Writer.__init__(self, verbose=True, use_counter=False)
        self.timings = []
        self.name = 'LogEntry.' + observer_method.name
        self.observer_name = observer_method.name

        # Time sub-entries:
        self.sub_entries = {}
        self.recursive_attach(observer_method)

    def attach(self, main):
        """
        Attachs class methods to the observer loop of main. 

        Parameters
        -----------
        main: AGOX object.
            The main AGOX class from agox/main.py
        """

        self.add_observer_method(self.start_timer, sets=self.sets[0], gets=self.gets[0], order=self.order[0]-0.01)
        self.add_observer_method(self.end_timer, sets=self.sets[0], gets=self.gets[0], order=self.order[0]+0.01)
        super().attach(main)

    def start_timer(self, state, *args, **kwargs):
        """
        Method attached as an observer to start the timing. 
        """
        self.timings.append(-dt())
        

    def end_timer(self, state, *args, **kwargs):
        """
        Method attached as an observer to end the timing.
        """
        if len(self.timings):
            self.timings[-1] += dt()
        

    def get_current_timing(self):
        """
        Get most recent timing. 

        Returns
        --------
        float
            Timing of most recent iteration.
        """

        if len(self.timings):
            return self.timings[-1]
        else:
            print(f'{self.name}: Somehow the log failed - Havent figure out why this happens sometimes - {len(self.timings)}')
            return 0

    def get_sub_timings(self):
        """
        Get timing of sub entries, relevant when the object self is attached around 
        is also an ObserverHandler that has other Observers, such as a Database 
        that may have a model listening to it. 

        Returns
        --------
        List or float
            If sub_entries are present then returns a list of those subtimings 
            otherwise returns a throguh 'get_current_timing'.
        """
        if len(self.sub_entries):
            sub_timings = []
            for sub_entry in self.sub_entries.values():
                sub_timings.append(sub_entry.get_sub_timings())
            return sub_timings
        else:
            return self.get_current_timing()

    def recursive_attach(self, observer_method):
        """
        Attach recursively when the class that ObserverMethod originates from 
        is an instance of ObserverHandler.

        Parameters
        -----------
        observer_method: ObserverMethod object. 
            An instance of a ObserverMethod (agox/observer.py) self is attached 
            around. 
        """
        if issubclass(observer_method.class_reference.__class__, ObserverHandler):

            if not observer_method.name == observer_method.class_reference.dispatch_method:
                return
            
            # Want to get all other observers: 
            class_reference = observer_method.class_reference # The class the inbound observer method comes from.
            observer_dicts = observer_method.class_reference.observers # Dict of observer methods.


            # May sometimes find objects that already have LogEntry observers 
            # attached, so we remove those. 
            methods_to_delete = []
            for key, observer_method in observer_dicts.items():
                if 'LogEntry' in observer_method.name:
                    methods_to_delete.append(observer_method)

            for method in methods_to_delete:
                class_reference.delete_observer(method)                    

            # Understand their order of execution:
            keys = []; orders = []
            for key, observer_dict in observer_dicts.items():
                keys.append(key) 
                orders.append(observer_dict['order'])
            sort_index = np.argsort(orders)
            sorted_keys = [keys[index] for index in sort_index]
            sorted_observer_dicts = [observer_dicts[key] for key in sorted_keys]

            # Attach log entry observers to time them:
            for observer_method, key in zip(sorted_observer_dicts, sorted_keys):
                self.add_sub_entry(observer_method)
                self.sub_entries[observer_method.key].attach(class_reference)

    def add_sub_entry(self, observer_method):
        """
        Add a sub entry which is an other instance of LogEntry. 

        Parameters
        -----------
        observer_method: ObserverMethod object. 
            An instance of a ObserverMethod (agox/observer.py) from an Observer 
            that is observing the Observer self is attached around. 
            E.g. if self is attached around a Database then a Model may be 
            observing that database with an ObserverMethod.  
        """
        self.sub_entries[observer_method.key] = LogEntry(observer_method)

    def get_iteration_report(self, report=None, offset=''):
        """
        Recursively generate iteration report. 

        Parameter
        ----------
        report: None or a list. 
            If None an empty list is created. Report strings are appended to that
            list.
        offset: str, 
            Indentation, reports from subentries are offset by 4 spaces. 

        Returns
        --------
        list
            The 'report' list of strings is returned. 
        """
        if report is None:
            report = [] # List of strings:

        report.append(offset + ' {} - {:05.2f} s'.format(self.observer_name, self.get_current_timing()))

        for sub_entry in self.sub_entries.values():
            report = sub_entry.get_iteration_report(report=report, offset=offset+' ' * 4)

        return report

class Logger(Observer, Writer):

    name = 'Logger'

    def __init__(self, save_frequency=100, **kwargs):
        """
        Logger instantiation. 

        The 'Logger' is the observer that is attached to main and prints 
        the log report. 

        Parameters:
        save_frequency: int
            How often to save to save the log to disk.
        """

        Observer.__init__(self, **kwargs)
        Writer.__init__(self, verbose=True, use_counter=False)
        self.log = Log()
        self.ordered_keys = []
        self.save_frequency = save_frequency

    def attach(self, main):
        """
        Attaches to main. 

        Three things happen:
        1.  LogEntry's are created for each observer, which may also recursively 
            create LogEntry's for observers of observers and so on.
        2.  The 'Logger' attaches an 'report_logs' as an Observer to main, so 
            that the log is printed. 
        3.  A finalization method is added so that the log is saved when AGOX 
            finishes. 

        Parameters
        -----------
        main: AGOX object.
            The main AGOX class from agox/main.py
        """
        # Want to get all other observers: 
        observers = main.observers

        # Understand their order of execution:
        keys = []; orders = []
        for observer_method in observers.values():
            keys.append(observer_method.key) 
            orders.append(observer_method.order)
        sort_index = np.argsort(orders)
        sorted_keys = [keys[index] for index in sort_index]
        sorted_observers = [observers[key] for key in sorted_keys]

        # Attach log obserers to time them:
        for observer_method, key in zip(sorted_observers, sorted_keys):
            self.log.add_entry(observer_method)
            self.log[key].attach(main)
    
        # Also attach a reporting:
        self.add_observer_method(self.report_logs, sets={}, gets={}, order=np.max(orders)+1)
        
        # Attach final dumping of logs:
        main.attach_finalization('Logger.dump', self.log.save_logs)

        super().attach(main)

    @agox_writer
    @Observer.observer_method
    def report_logs(self, state):
        """
        Calls the 'log_report' of the log object.

        Saves the log to disk depending on self.save_frequency. 
        """
        self.log.log_report()
        
        if self.get_iteration_counter() % self.save_frequency == 0:
            self.log.save_logs()
        

