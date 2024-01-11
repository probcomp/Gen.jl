---
permalink: /ecosystem/
title: "Gen Ecosystem"
layout: splash
---

<br>
# Gen Ecosystem

Gen is a platform for probabilistic modeling and inference.
Gen includes a set of built-in languages for defining probabilistic models and a standard library for defining probabilistic inference algorithms, but is designed to be extended with an open-ended set of more specialized modeling languages and inference libraries.

<br>
<hr> 
## Core packages
<br>

#### [Gen](https://github.com/probcomp/Gen.jl)
The main Gen package.
Contains the core abstract data types for models and traces.
Also includes general-purpose modeling languages and a standard inference library.

#### [GenPyTorch](https://github.com/probcomp/GenPyTorch.jl)
Gen modeling language that wraps [PyTorch](https://pytorch.org) computation graphs.

#### [GenTF](https://github.com/probcomp/GenTF)
Gen modeling language that wraps [TensorFlow](https://www.tensorflow.org) computation graphs.

#### [GenFluxOptimizers](https://github.com/probcomp/GenFluxOptimizers.jl)
Enables the use of any of [Flux](https://github.com/FluxML/Flux.jl)'s optimizers for parameter learning in generative functions from Gen's static or dynamic modeling languages.

#### [GenParticleFilters](https://github.com/probcomp/GenParticleFilters.jl)
Building blocks for basic and advanced particle filtering.

<br>
<hr> 

## Contributed packages
<br>

#### [GenPseudoMarginal](https://github.com/probcomp/GenPseudoMarginal.jl)
Building blocks for [modular probabilistic inference](https://arxiv.org/abs/1612.04759) using pseudo-marginal Monte Carlo algorithms.

#### [GenVariableElimination](https://github.com/probcomp/GenVariableElimination.jl)
Compile portions of traces into factor graphs and use variable elimination on them.

#### [Gen2DAgentMotion](https://github.com/probcomp/Gen2DAgentMotion.jl)
Components for building generative models of the motion of an agent moving around a 2D environment.

#### [GenDirectionalStats](https://github.com/probcomp/GenDirectionalStats.jl)
Probability distributions and involutive MCMC kernels on orientations and rotations.

#### [GenRedner](https://github.com/probcomp/GenRedner.jl)
Wrapper for employing the [Redner](https://github.com/BachiLi/redner) differentiable renderer in Gen generative models.

#### [GenTraceKernelDSL](https://github.com/probcomp/GenTraceKernelDSL.jl)
An alternative interface to defining [trace translators](https://www.gen.dev/docs/dev/ref/trace_translators/).
