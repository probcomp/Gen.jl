# Gen.jl

A general-purpose probabilistic programming system with programmable inference, embedded in Julia.

## Features

- Multi-paradigm Bayesian inference via [Sequential Monte Carlo](ref/inference/pf.md), [variational inference](ref/inference/vi.md), [MCMC](ref/inference/mcmc.md), and more.
- Gradient-based training of generative models via [parameter optimization](ref/inference/parameter_optimization.md), [wake-sleep learning](ref/inference/wake_sleep.md), etc.
- An expressive and intuitive [modeling language](ref/modeling/dml.md) for writing and composing probabilistic programs.
- Inference algorithms are *programmable*: Write [custom proposals](https://www.gen.dev/tutorials/data-driven-proposals/tutorial), [variational families](tutorials/vi.md), [MCMC kernels](ref/inference/mcmc.md) or [SMC updates](ref/inference/trace_translators.md) without worrying about the math.
- Support for Bayesian structure learning via [involutive MCMC](@ref involutive_mcmc) and [SMCP³](@ref advanced-particle-filtering).
- [Specialized modeling constructs](tutorials/scaling_with_sml.md) that speed-up inference by supporting incremental computation.
- Well-defined APIs for implementing [custom generative models](how_to/custom_gen_fns.md), [distributions](how_to/custom_distributions.md), [gradients](how_to/custom_gradients.md), etc.

## Installation

The Gen package can be installed with the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and then run:

```
add Gen
```

To install the latest development version, you may instead run:

```
add https://github.com/probcomp/Gen.jl.git
```

Gen can now be used in the Julia REPL, or at the top of a script:

```julia
using Gen
```

To test the installation locally, you can run the tests with:

```julia
using Pkg; Pkg.test("Gen")
```

## Tutorials

Learn how to use Gen by following these tutorials:

```@contents
Pages = [
    "tutorials/getting_started.md",
    "tutorials/modeling_in_gen.md",
    "tutorials/smc.md",
    "tutorials/scaling_with_sml.md",
    "tutorials/learning_gen_fns.md"
]
Depth = 1
```

More tutorials [can also be found on our website](https://www.gen.dev/tutorials/).
Additional examples of Gen usage can be found in [GenExamples.jl](https://github.com/probcomp/GenExamples.jl).

## Questions and Contributions

If you have questions about using Gen.jl, feel free to open a [discussion on GitHub](https://github.com/probcomp/Gen.jl/discussions). If you encounter a bug, please [open an issue](https://github.com/probcomp/Gen.jl/issues). We also welcome bug fixes and feature additions as [pull requests](https://github.com/probcomp/Gen.jl/pulls).

## Supporting and Citing
 
Gen.jl is part of ongoing research at the [MIT Probabilistic Computing Project](http://probcomp.csail.mit.edu). If you use Gen for your work, please consider citing us:

> *Gen: A General-Purpose Probabilistic Programming System with Programmable Inference.* Cusumano-Towner, M. F.; Saad, F. A.; Lew, A.; and Mansinghka, V. K. In Proceedings of the 40th ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI ‘19). ([pdf](https://dl.acm.org/doi/10.1145/3314221.3314642)) ([bibtex](https://www.gen.dev/assets/gen-pldi.txt))
