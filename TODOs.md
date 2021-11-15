# TODOs

These are some TODOs for the first paper draft.

## Stochastic environment

At the moment, the environment is deterministic. 
This might lead to an agent learning wrong correlations from the trajectories.
One thing I can do is to make observations come from a certain distribution.

## States

States are the memory system at time $t$. 
Instead of using word embeddings, the simplest way of converting them to numbers is to simply replace them with symbol ids.

This is an example:

```
Tae's: 0                                                                                 
Vincent's: 1                                                                             
Michael's: 2                                                                             
                                                                                         
laptop: 10                                                                               
desk: 20                                                                                 
kitchen: 21                                                                              
AtLocation: 30 
```

