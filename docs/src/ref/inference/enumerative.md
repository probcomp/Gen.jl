# Enumerative Inference

Enumerative inference can be used to compute the exact posterior distribution for a generative model
with a finite number of discrete random choices, to compute a grid approximation of a continuous 
posterior density, or to perform stratified sampling by enumerating over discrete random choices and sampling 
the continuous random choices. This functionality is provided by [`enumerative_inference`](@ref).

```@docs
enumerative_inference
```

To construct a rectangular grid of [choice maps](../core/choice_maps.md) and their associated log-volumes to iterate over, use the [`choice_vol_grid`](@ref) function.

```@docs
choice_vol_grid
```

When the space of possible choice maps is not rectangular (e.g. some addresses only exist depending on the values of other addresses), iterators over choice maps and log-volumes can be also be manually constructed.
