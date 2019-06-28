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

Gen's flexible modeling and inference programming capabilities unify symbolic, neural, probabilistic, and simulation-based approaches to modeling and inference, including causal modeling, symbolic programming, deep learning, hierarchical Bayesiam modeling, graphics and physics engines, and planning and reinforcement learning.

# Getting Started

## Using Julia package manager
First, [download Julia 1.0 or later](https://julialang.org/downloads/).

The, install the Gen package with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and then run:
```
pkg> add https://github.com/probcomp/Gen
```
## Docker

A docker image containing an installation of Gen, with tutorial Jupyter notebooks, is available [here](https://github.com/probcomp/gen-quickstart). 

# Benchmarks

Code for benchmarks, presented at PLDI 2019, are available [here](https://github.com/probcomp/pldi2019-gen-experiments).

# Publications

Gen: A General-Purpose Probabilistic Programming System with Programmable Inference. Cusumano-Towner, M. F.; Saad, F. A.; Lew, A.; and Mansinghka, V. K. In Proceedings of the 40th ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI '19). [URL](https://dl.acm.org/citation.cfm?id=3314642)

Incremental inference for probabilistic programs. Cusumano-Towner, M. F.; Bichsel, B.; Gehr, T.; Vechev, M.; and Mansinghka, V. K. In Proceedings of the 39th ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI), pages 571–585. ACM, 2018. [URL](https://dl.acm.org/citation.cfm?id=3192399).

A design proposal for Gen: Probabilistic programming with fast custom inference via code generation. Cusumano-Towner, M. F.; and Mansinghka, V. K. In Workshop on Machine Learning and Programming Languages (MAPL, co-located with PLDI), pages 52–57. 2018. [URL](https://dl.acm.org/citation.cfm?id=3211350).

Using probabilistic programs as proposals. Cusumano-Towner, M. F.; and Mansinghka, V. K. In Workshop on Probabilistic Programming Languages, Semantics, and Systems (PPS, co-located with POPL). 2018. [URL](https://arxiv.org/pdf/1801.03612.pdf).

Encapsulating models and approximate inference programs in probabilistic modules. Cusumano-Towner, M. F.; and Mansinghka, V. K. In Workshop on Probabilistic Programming Semantics (PPS, co-located with POPL). 2017. [URL](https://arxiv.org/pdf/1612.04759.pdf).
