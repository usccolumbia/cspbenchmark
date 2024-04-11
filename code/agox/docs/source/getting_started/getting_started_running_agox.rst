Getting started: First AGOX run
================================

Having installed AGOX we are ready to try performing an actual AGOX search. 

An example of a script is given below.

.. literalinclude:: rss_script_single.py

This script runs a random-structure-search run for a easy system consisting of just 5 gold atoms and a single nickel atom 
described by the EMT potential included in ASE. 

Copy the script and save it, then run the script

.. code-block:: console

   python rss_script_single.py

This will run in about a minute on a modern machine. After completion the script has produced a single file 'db1.db'.
This file contains the information gathered by the search run. We can analyze the run using the a tool installed together 
with AGOX. 

Having pip installed agox the CLI tool :code:`agox-analysis` is available, use the tool like so 

.. code-block:: console

   agox-analysis -d . -e -hg -c

This produces a figure like this one: 

.. figure:: rss_simple_single.png

Which shows the structure with the lowest energy found on the left and the energy as a function of the number of single-point 
calculations (SPCs) on the right.
Here the deeper blue is the best energy found until and including that number of SPCs and the lighter blue is the energy of the actual structure 
found at that iteration. 

