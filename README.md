# Gen.jl

[![Build Status](https://img.shields.io/github/actions/workflow/status/probcomp/Gen.jl/ContinuousIntegration.yml?branch=master)](https://github.com/probcomp/Gen.jl/actions)
[![Documentation (Stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://probcomp.github.io/Gen.jl/docs/stable)
[![Documentation (Dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://probcomp.github.io/Gen.jl/docs/dev)
![GitHub Release](https://img.shields.io/github/v/release/probcomp/Gen.jl?color=white)

A general-purpose probabilistic programming system with programmable inference, embedded in Julia.

See [https://gen.dev](https://gen.dev/) for introduction, documentation, and tutorials.

## Features

*   Multi-paradigm Bayesian inference via [Sequential Monte Carlo](https://www.gen.dev/docs/stable/ref/inference/pf/), [variational inference](https://www.gen.dev/docs/stable/ref/inference/vi/), [MCMC](https://www.gen.dev/docs/stable/ref/inference/mcmc/), and more.
*   Gradient-based training of generative models via [parameter optimization](https://www.gen.dev/docs/stable/ref/inference/parameter_optimization/), [wake-sleep learning](https://www.gen.dev/docs/stable/ref/inference/wake_sleep/), etc.
*   An expressive and intuitive [modeling language](https://www.gen.dev/docs/stable/ref/modeling/dml/) for writing and composing probabilistic programs.    
*   Inference algorithms are _programmable_: Write [custom proposals](https://www.gen.dev/tutorials/data-driven-proposals/tutorial), [variational families](https://www.gen.dev/docs/stable/tutorials/vi/), [MCMC kernels](https://www.gen.dev/docs/stable/ref/inference/mcmc/) or [SMC updates](https://www.gen.dev/docs/stable/ref/inference/trace_translators/) without worrying about the math.
*   Support for Bayesian structure learning via [involutive MCMC](https://www.gen.dev/docs/stable/ref/inference/mcmc/#involutive_mcmc) and [SMCP³](https://www.gen.dev/docs/stable/ref/inference/pf/#advanced-particle-filtering).
*   [Specialized modeling constructs](https://www.gen.dev/docs/stable/tutorials/scaling_with_sml/) that speed-up inference by supporting incremental computation.
*   Well-defined APIs for implementing [custom generative models](https://www.gen.dev/docs/stable/how_to/custom_gen_fns/), [distributions](https://www.gen.dev/docs/stable/how_to/custom_distributions/), [gradients](https://www.gen.dev/docs/stable/how_to/custom_gradients/), etc.

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

## Questions and Contributions

If you have questions about using Gen.jl, feel free to open a [discussion on GitHub](https://github.com/probcomp/Gen.jl/discussions). If you encounter a bug, please [open an issue](https://github.com/probcomp/Gen.jl/issues). We also welcome bug fixes and feature additions as [pull requests](https://github.com/probcomp/Gen.jl/pulls). Please refer to our [contribution guidelines](https://github.com/probcomp/Gen.jl/blob/master/CONTRIBUTING.md) for more details.

## Supporting and Citing
 
Gen.jl is part of ongoing research at the [MIT Probabilistic Computing Project](http://probcomp.csail.mit.edu). To get in contact, please email gen-contact@mit.edu.

If you use Gen in your research, please cite our 2019 PLDI paper:

> *Gen: A General-Purpose Probabilistic Programming System with Programmable Inference.* Cusumano-Towner, M. F.; Saad, F. A.; Lew, A.; and Mansinghka, V. K. In Proceedings of the 40th ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI ‘19). ([pdf](https://dl.acm.org/doi/10.1145/3314221.3314642)) ([bibtex](https://www.gen.dev/assets/gen-pldi.txt))
