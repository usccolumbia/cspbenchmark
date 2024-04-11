GOFEE 
==============

The GOFEE algorithm is a Bayesian search algorithm first presented by Bisbo & Hammer

In GOFEE *N* candidates are generated each episode and are all locally optimzied 
in a Gaussian process regression (GPR) potential - or in fact in the so-called 
lower-confidence-bound expression given by

.. math::
    E(\mathbf{x}) = \hat{E}(\mathbf{x}) - \kappa \sigma(\mathbf{x})

Where :math:`\hat{E}` and :math:`\sigma` are the predicted energy and uncertainty of the 
GPR model for the structure represented by :math:`\mathbf{x}`. Following that the 
most promising candidate(s) are chosen for evaluation by an acquisitor, that also uses 
the LCB expression - such that those candidates that have low energy and high uncertainty 
are preferentially evaluated. 

Overall the algorithm has the following flow: 

1. Generate *N* candidates. 
2. Locally optimize those candidates in the LCB. 
3. Use the acquisitor to pick candidates for evaluation. 
4. Evaluate a small (usually just 1) number of candidates with only a few (usually 1) gradient-step 
in the target potential. 
5. Store in the evaluated candidate(s) in the database. 
6. Update the GPR model with the new data. 

Both step 1 and 2 happen in parallel using `Ray <https://www.ray.io/>`_.

.. literalinclude:: ../../../agox/test/run_tests/script_gofee.py


