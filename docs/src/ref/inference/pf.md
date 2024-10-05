# [Particle Filtering and Sequential Monte Carlo](@id particle_filtering)

Gen.jl provides support for Sequential Monte Carlo (SMC) inference in the form of particle filtering.
The state of a particle filter is a represented as a `ParticleFilterState` object.

```@docs
Gen.ParticleFilterState
```

## Particle Filtering Steps

The basic steps of particle filtering are *initialization* (via [`initialize_particle_filter`](@ref)), *updating* (via [`particle_filter_step!`](@ref)), and *resampling* (via [`maybe_resample!`](@ref)). The latter two operations are applied to a [`ParticleFilterState`](@ref), modifying it in place.

```@docs
initialize_particle_filter
particle_filter_step!
maybe_resample!
```

## Accessors

The following accessor functions can be used to return information about a [`ParticleFilterState`](@ref), or to sample traces from the distribution that the particle filter approximates.

```@docs
log_ml_estimate
get_traces
get_log_weights
sample_unweighted_traces
```

## [Advanced Particle Filtering](@id advanced-particle-filtering)

For a richer set of particle filtering techniques, including support for stratified sampling, multiple resampling methods, MCMC rejuvenation moves, particle filter resizing, users are recommended to use the [GenParticleFilters.jl](https://github.com/probcomp/GenParticleFilters.jl) extension library.

To use the generalization of standard SMC known as Sequential Monte Carlo with Probabilistic Program Proposals (SMCP³), use the API provided by [GenSMCP3.jl](https://github.com/probcomp/GenSMCP3.jl), or implement an [`UpdatingTraceTranslator`](https://probcomp.github.io/GenParticleFilters.jl/stable/translate/#GenParticleFilters.UpdatingTraceTranslator) in GenParticleFilters.jl.

Even more advanced SMC techniques (such as divide-and-conquer SMC) are not currently supported by Gen.
