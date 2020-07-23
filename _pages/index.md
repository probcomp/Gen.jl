---
permalink: /
title: "Gen"
layout: splash

header:
  overlay_color: "#000"
  overlay_filter: "0.5"
  overlay_image: assets/images/cropped_waves.jpg
excerpt: "Gen is a general-purpose probabilistic programming system, embedded in Julia."

intro: 
  - excerpt: 'Nullam suscipit et nam, tellus velit pellentesque at malesuada, enim eaque. Quis nulla, netus tempor in diam gravida tincidunt, *proin faucibus* voluptate felis id sollicitudin. Centered with `type="center"`'

logo_row1:
  - image_path: assets/images/mit-logo.png
    alt: "placeholder image 1"
  - image_path: /assets/images/berkeley-logo.jpg
    alt: "placeholder image 2"
  - image_path: /assets/images/yale-logo.jpeg
    alt: "placeholder image 2"
  - image_path: /assets/images/uw-madison-logo.png
    alt: "placeholder image 2"

feature_row2:
    - title: "Gen supports custom hybrid inference algorithms"
    - excerpt: 'Neural network inference is fast, but can be inaccurate on out-of-distribution data, and requires expensive training. Model-based inference is more computationally expensive, but does not require retraining, and can be more accurate. Gen supports custom hybrid inference algorithms that benefit from the strengths of both approaches.'

feature_row3:
  - image_path: /assets/images/trace.png
    alt: "placeholder image 2"
    title: "Gen users implement custom algorithms without extending the language implementation"
    excerpt: 'Instead of an inference engine, Gen offers a flexible "Trace" data type for implementing an open-ended set of inference and learning algorithms without touching the modeling-language implementation. Gen trace operations includes automatic differentiation (AD), but goes beyond AD and includes other operations needed for model-based reasoning.'

feature_row4:
  - image_path: /assets/images/open-universe.png
    alt: "placeholder image 2"
    title: "Gen supports inference in open-universe models"
    excerpt: 'Generative and inference models in Gen can have stochastic control flow (dimension). Support for custom reversible jump MCMC allows for much more efficient inference over model structure.'

feature_row5:
  - image_path: /assets/images/extensible.png
    alt: "placeholder image 2"
    title: "Gen is extensible"
    excerpt: 'Gen is designed for extensibility and to support incremental performance optimization as the inference application is scaled from a small-scale proof of concept to a prototype. Users can optimize bottlenecks in their model by hand-optimized code that inter-operates with Gens inference library code.'

---

<br>
# Why Gen
{: style="text-align: center"}

### Implement sophisticated probabilistic inference algorithms without the math
Gen's inference library gives you building blocks for writing efficient probabilistic inference algorithms that are tailored to your model, while automating the tricky math and the low-level implementation details.
Gen helps you write hybrid algorithms that combine neural networks, variational inference, sequential Monte Carlo samplers, and Markov chain Monte Carlo.

### Flexibly navigate performance trade-offs
Gen features an easy-to-use modeling language for writing down generative models, inference models, variational families, and proposal distributions using ordinary Julia code. 
But it also lets you migrate parts of your model or inference algorithm to specialized modeling languages for which it can generate especially fast code.
You can also hand-code parts of your models that demand better performance.

### Gen supports custom hybrid inference algorithms
Neural network inference is fast, but can be inaccurate on out-of-distribution data, and requires expensive training. Model-based inference is more computationally expensive, but does not require retraining, and can be more accurate. Gen supports custom hybrid inference algorithms that benefit from the strengths of both approaches.

### Implement custom inference algorithms without modifying the compiler
Instead of an *inference engine* that tightly couples inference algorithms with language compiler details, Gen gives you a *flexible API* for implementing an open-ended set of inference and learning algorithms.
This API includes automatic differentiation (AD), but goes far beyond AD and includes many other operations that are needed for model-based inference algorithms.

### Efficient inference in models with stochastic structure
Generative models and inference models in Gen can have dynamic computation graphs.
Gen's unique support for custom reversible jump and [involutive MCMC](https://arxiv.org/abs/2007.09871) algorithms allows for more efficient inference in generative models with stochastic structure.


<br>
# Installing Gen
{: style="text-align: center"}
<br>
First, [download Julia](https://julialang.org/downloads/).
Then, install the Gen package with the Julia package manager:

From the Julia REPL, type `]` to enter the Pkg REPL mode and then run:
```
pkg> add Gen
```

<br>
# Gen is an open-source academic project
If you use Gen in your research, please cite our [paper](https://dl.acm.org/doi/10.1145/3314221.3314642):

Gen: A General-Purpose Probabilistic Programming System with Programmable Inference. Cusumano-Towner, M. F.; Saad, F. A.; Lew, A.; and Mansinghka, V. K. In Proceedings of the 40th ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI ‘19).
```
@inproceedings{Cusumano-Towner:2019:GGP:3314221.3314642,
 author = {Cusumano-Towner, Marco F. and Saad, Feras A. and Lew, Alexander K. and Mansinghka, Vikash K.},
 title = {Gen: A General-purpose Probabilistic Programming System with Programmable Inference},
 booktitle = {Proceedings of the 40th ACM SIGPLAN Conference on Programming Language Design and Implementation},
 series = {PLDI 2019},
 year = {2019},
 isbn = {978-1-4503-6712-7},
 location = {Phoenix, AZ, USA},
 pages = {221--236},
 numpages = {16},
 url = {http://doi.acm.org/10.1145/3314221.3314642},
 doi = {10.1145/3314221.3314642},
 acmid = {3314642},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {Markov chain Monte Carlo, Probabilistic programming, sequential Monte Carlo, variational inference},
} 
```


<br>
# Institutions using Gen
{: style="text-align: center"}
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
# The Gen team
{: style="text-align: center"}
Gen was originally created by [Marco Cusumano-Towner](https://www.mct.dev) at the [MIT Probabilistic Computing Project](http://probcomp.csail.mit.edu/), which is led by [Vikash Mansinghka](http://probcomp.csail.mit.edu/principal-investigator/).
Gen continues to actively evolve with the help of a core team that includes Ben Zinberg, [Alex Lew](http://alexlew.net/), [Tan Zhi-Xuan](https://github.com/ztangent/), and [George Matheos](https://www.linkedin.com/in/george-matheos-429982160/); and a number of open-source [contributors](https://github.com/probcomp/Gen.jl/graphs/contributors).
