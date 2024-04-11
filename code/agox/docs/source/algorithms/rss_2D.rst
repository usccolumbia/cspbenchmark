Random structure search (2D)
==============================

AGOX can run searches in three, two and one dimensions. The following script 
shows how to setup a 2D run, in this case for RSS but the changes apply to any 
algorithm. 

This involves: 

1. Setting the third vector of the confinement cell to zero in all entries. 
2. Adding an additional FixedPlane constraint. 

Similarly for a 1D problem, the second and third dimensions of confinement cell 
must be set to zero and a FixedLine constraint should be added. 

.. literalinclude:: ../../../agox/test/run_tests/script_rss_2d.py


