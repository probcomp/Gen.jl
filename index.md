---
---

# Introduction
Probabilistic modeling and inference are core tools in diverse fields including statistics, machine learning, computer vision, cognitive science, robotics, natural language processing, and artificial intelligence.
To meet the functional requirements of applications, practitioners use a broad range of modeling techniques and approximate inference algorithms.
However, implementing inference algorithms is often difficult and error prone.
Gen simplifies the use of probabilistic modeling and inference, by providing *modeling languages* in which users express models, and high-level programming constructs that automate aspects of inference.

Like some probabilistic programming research languages, Gen includes *universal* modeling languages that can represent any model, including models with stochastic structure, discrete and continuous random variables, and simulators.
However, Gen is distinguished by the flexibility that it affords to users for customizing their inference algorithm.
It is possible to use built-in algorithms that require only a couple lines of code, as well as develop custom algorithms that are more able to meet scalability and efficiency requirements.

Gen's flexible modeling and inference programming capabilities unify symbolic, neural, probabilistic, and simulation-based approaches to modeling and inference, including causal modeling, symbolic programming, deep learning, hierarchical Bayesiam modeling, graphics and physics engines, and planningand reinforcement leraning.


# Getting Started
First, [download Julia 1.0 or later](https://julialang.org/downloads/).

The, install the Gen package with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and then run:
```
pkg> add https://github.com/probcomp/Gen
```

# Benchmarks

Code for benchmarks, presented at PLDI 2019, are available [here](https://github.com/probcomp/pldi2019-gen-experiments).

# Docker Image with Tutorials

A docker image, with tutorial Jupyter notebooks are available [here](https://github.com/probcomp/gen-quickstart). 
