---
layout: index
title: "Gen"
---


<br>
# Why Gen
{: class="homepage"}

### Gen automates the implementation details of probabilistic inference algorithms
{: class="homepage"}
Gen's inference library gives users building blocks for writing efficient probabilistic inference algorithms that are tailored to their models, while automating the tricky math and the low-level implementation details.
Gen helps users write hybrid algorithms that combine neural networks, variational inference, sequential Monte Carlo samplers, and Markov chain Monte Carlo.

### Gen allows users to flexibly navigate performance trade-offs
{: class="homepage"}
Gen features an easy-to-use modeling language for writing down generative models, inference models, variational families, and proposal distributions using ordinary code. 
But it also lets users migrate parts of their model or inference algorithm to specialized modeling languages for which it can generate especially fast code.
Users can also hand-code parts of their models that demand better performance.

### Gen supports custom hybrid inference algorithms
{: class="homepage"}
Neural network inference is fast, but can be inaccurate on out-of-distribution data, and requires expensive training. Model-based inference is more computationally expensive, but does not require retraining, and can be more accurate. Gen supports custom hybrid inference algorithms that benefit from the strengths of both approaches.

### Users write custom inference algorithms without extending the compiler
{: class="homepage"}
Instead of an *inference engine* that tightly couples inference algorithms with language compiler details, Gen gives users a *flexible API* for implementing an open-ended set of inference and learning algorithms.
This API includes automatic differentiation (AD), but goes far beyond AD and includes many other operations that are needed for model-based inference algorithms.

### Efficient inference in models with stochastic structure
{: class="homepage"}
Generative models and inference models in Gen can have dynamic computation graphs.
Gen's unique support for custom reversible jump and [involutive MCMC](https://arxiv.org/abs/2007.09871) algorithms allows for more efficient inference in generative models with stochastic structure.


<br>
# Installing Gen
{: class="homepage"}

<br>
We maintain a Julia implementation of the Gen architecture, and we are currently working on porting Gen to other languages.
To install the Julia implementation of Gen, [download Julia](https://julialang.org/downloads/).
Then, install the Gen package with the Julia package manager:

From the Julia REPL, type `]` to enter the Pkg REPL mode and then run:

```
pkg> add Gen
```

<br>
# Institutions using Gen
{: class="homepage"}

<div class="logo-table">
<table>
<tr>
<td> <img src="assets/images/mit-logo.png" width="300" /> </td>
<td> <img src="assets/images/berkeley-logo.jpg" width="200" /> </td>
<td> <img src="assets/images/yale-logo.jpeg" width="300" /> </td>
</tr>
<tr>
<td> <img src="assets/images/uw-madison-logo.png" width="300" /> </td>
<td> <img src="assets/images/intel-logo.png" width="150" /> </td>
<td> <img src="assets/images/ibm-logo.png" width="150" /> </td>
</tr>
<tr>
<td> <img src="assets/images/umass-amherst-logo.png" width="200" /> </td>
</tr>
</table>
</div>

<br>
# The Gen.jl team
{: class="homepage"}

Gen.jl was created by [Marco Cusumano-Towner](https://www.mct.dev) the [MIT Probabilistic Computing Project](http://probcomp.csail.mit.edu/), which is led by [Vikash Mansinghka](http://probcomp.csail.mit.edu/principal-investigator/).
Gen.jl has grown and is maintained through the help of a core research and engineering team that includes [Alex Lew](http://alexlew.net/), [Tan Zhi-Xuan](https://github.com/ztangent/), [George Matheos](https://www.linkedin.com/in/george-matheos-429982160/), [McCoy Becker](https://femtomc.github.io/), and [Feras Saad](http://fsaad.mit.edu), as well as a number of open-source [contributors](https://github.com/probcomp/Gen.jl/graphs/contributors).
The Gen architecture is described in Marco's [PhD thesis](https://www.mct.dev/assets/mct-thesis.pdf).

If you use Gen in your research, please cite our PLDI paper:

Gen: A General-Purpose Probabilistic Programming System with Programmable Inference. Cusumano-Towner, M. F.; Saad, F. A.; Lew, A.; and Mansinghka, V. K. In Proceedings of the 40th ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI ‘19). ([pdf](https://dl.acm.org/doi/10.1145/3314221.3314642)) ([bibtex](assets/gen-pldi.txt))

<br>
