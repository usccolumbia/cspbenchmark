Using GPAW 
===========

The GPAW ASE calculator is kind of special in two ways 
* It is not an IO calculator
* It can, and generally should, be used with MPI. 

However AGOX, debliberatly, does not use MPI we cannot launch an AGOX script 
with mpirun and thus 'naively' using a GPAW calculator will lead to trouble. 
Therefore we have implemented a GPAW IO calculator, that can be used like so

.. code-block:: 

    from agox.helpers.gpaw_io import GPAW_IO

    calc = GPAW_IO(mode='lcao',
        xc='PBE',
        basis='dzp',
        maxiter='200',
        kpts ='(1, 1, 1)',
        poissonsolver='PoissonSolver(eps=1e-7)',
        mixer='Mixer(0.05,5,100)',
        convergence="""{'energy':0.005, 'density':1.0e-3, 
        'eigenstates':1.0e-3, 'bands':'occupied'}", occupations="FermiDirac(0.1)""",
        gpts = "h2gpts(0.2, t.get_cell(), idiv = 8)",
        nbands='110%',
        txt='dft_log_lcao.txt', 
        modules=['from gpaw.utilities import h2gpts',
                'from gpaw import FermiDirac',
                'from gpaw.poisson import PoissonSolver',
                'from gpaw import Mixer'])

Which is almost like initializing a normal GPAW calculator, but rather than 
giving objects as argument everything is given as strings! This is important, 
do **not** import anything from GPAW in your AGOX script - it breaks things! 
The calculator spawns another process, where MPI is enabled, and initalizes 
a GPAW calculator translating the string arguments to Python objects. If as in 
this case more than the standard modules are required, they can be imported 
with the :code:`modules` argument. 