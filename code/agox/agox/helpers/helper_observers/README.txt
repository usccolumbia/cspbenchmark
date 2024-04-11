These classes are meant to help add functionality to a run without changing other classes. 
Also helps with understanding how the observer-patterns work.
The current ones are just simple examples of what can be done. 

Example usage: 

agox = AGOX(environment=environment, database=database, db=database, model_calculator=model_calculator, 
                 collector=collector, sampler=sampler, acquisitor=acquisitor)

timer = iterationTimer()
timer.easy_attach(agox)

agox.run(N_iterations=NUM_iterationS)

Will attach the timing observers and print the total run time for each iteration. 

Much more complicated things are possible using the same general concept, e.g. instead of 
writing a new version of a module to dynamically change a hyper-parameter one can write a 
helper observer that does so for a quick & dirty test. 
This also means that this concept can be used to turn things on and off. 
