# Standard Inference Library

## Importance Sampling
```@docs
importance_sampling
importance_resampling
```

## Markov Chain Monte Carlo
The following inference library methods take a trace and return a new trace.
```@docs
metropolis_hastings
mh
mala
hmc
elliptical_slice
```

## Optimization over Random Choices
```@docs
map_optimize
```

## Particle Filtering
```@docs
initialize_particle_filter
particle_filter_step!
maybe_resample!
log_ml_estimate
get_traces
get_log_weights
sample_unweighted_traces
```

## Supervised Training
```@docs
train!
```

## Variational Inference
```@docs
black_box_vi!
```
