new env

Everything is happening in one "room".

1. The room is initialized with N objects. With 50% of chance, an object will be located at a common-sense spot.
1. Every time step, the agent observes an object and its location. The observation is uniform-random, but not the same object and location.
1. Every T (\< N) time steps, the locations of the objects change.
1. Every 2T time steps, the envrionment asks the object locations.
