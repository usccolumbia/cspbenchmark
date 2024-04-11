from abc import ABC, abstractmethod
from agox.observer import Observer
from agox.writer import Writer, agox_writer


class EvaluatorBaseClass(ABC, Observer, Writer):

    def __init__(self, number_to_evaluate=1, gets={'get_key':'prioritized_candidates'}, sets={'set_key':'evaluated_candidates'}, 
        order=5, verbose=True, use_counter=True, prefix='', surname=''):
        Observer.__init__(self, gets=gets, sets=sets, order=order, surname=surname)
        Writer.__init__(self, verbose=verbose, use_counter=use_counter, prefix=prefix)
        self.number_to_evaluate = number_to_evaluate

        self.add_observer_method(self.evaluate,
                                 sets=self.sets[0], gets=self.gets[0], order=self.order[0],
                                 handler_identifier='AGOX')
    
    def __call__(self, candidate):
        return self.evaluate_candidate(candidate)
        
    @abstractmethod
    def evaluate_candidate(self, candidate):
        """
        Evaluates the given candidate.

        This function MUST append the candidates it wishes to pass to the AGOX 
        State cache to the list self.evaluated_candidates. 

        Parameters
        ----------
        candidate : AGOX candidate object.
            The candidate object to evaluate.

        Returns
        -------
        bool
            Whether the evaluation was successful. 
        """
        return successful

    @property
    @abstractmethod
    def name(self):
        pass

    @agox_writer
    @Observer.observer_method
    def evaluate(self, state):

        candidates = state.get_from_cache(self, self.get_key)
        done = False

        self.evaluated_candidates = []
        passed_evaluation_count = 0
        if self.do_check():
            while candidates and not done:
                self.writer(f'Trying candidate - remaining {len(candidates)}')
                candidate = candidates.pop(0)

                if candidate is None:
                    self.writer('Candidate was None - are your other modules working as intended?')
                    continue

                internal_state = self.evaluate_candidate(candidate)

                if internal_state:
                    self.writer('Succesful calculation of candidate.')
                    passed_evaluation_count += 1
                    self.evaluated_candidates[-1].add_meta_information('final', True)

                    if passed_evaluation_count == self.number_to_evaluate:
                        self.writer('Calculated required number of candidates.')
                        done = True
        
        state.add_to_cache(self, self.set_key, self.evaluated_candidates, mode='a')
        
    def __add__(self, other):
        return EvaluatorCollection(evaluators=[self, other])

class EvaluatorCollection(EvaluatorBaseClass): 

    name = 'EvaluatorCollection'

    def __init__(self, evaluators):
        super().__init__()
        self.evaluators = evaluators

    def evaluate_candidate(self, candidate):
        state = self.apply_evaluators(candidate)
        

    def add_evaluator(self, evaluator):
        self.evaluators.append(evaluator)

    def list_evaluators(self):
        for i, evaluator in enumerate(self.evaluators):
            print('Evaluator {}: {} - {}'.format(i, evaluator.name, evaluator))

    def apply_evaluators(self, candidate):
        for evaluator in self.evaluators:
            evaluator_state = evaluator(candidate)
            if not evaluator_state:
                return False
        return True
    
    def __add__(self, other):
        self.evaluators.append(other)
        return self
