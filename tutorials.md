---
permalink: /tutorials/
title: "Gen Tutorials"
layout: splash
---

<br>
# Gen Tutorials

The tutorials below were generated from Jupyter notebooks, which are available in the [Gen Quickstart](https://github.com/probcomp/gen-quickstart) repository.

<br>
### [Introduction to Modeling in Gen](intro-to-modeling/tutorial)
This tutorial introduces the basics of modeling in Gen. It shows how to perform inference using generic inference algorithms. It does not explore custom inference programming.

<br>
### [Basics of Iterative Inference in Gen](iterative-inference/tutorial)
This tutorial introduces the basics of inference programming in Gen using iterative inference programs, which include Markov chain Monte Carlo algorithms.

<br>
### [Data-Driven Proposals in Gen](data-driven-proposals/tutorial)
Data-driven proposals use information in the observed data set to choose the proposal distibution for latent variables in a generative model. 
This tutorial shows you how to use custom data-driven proposals to accelerate Monte Carlo inference. 
It also demonstrates how 'black-box' code, like algorithms and simulators written in Julia, can be included
in probabilistic models that are expressed as generative functions.

<br>
### [Scaling with Combinators and the Static Modeling Language](scaling-with-combinators-new/tutorial)
This tutorial shows how generative function combinators and the static modeling language are used to achieve good asymptotic scaling time of inference algorithms.

<br>
### [Particle Filtering in Gen](particle-filtering/tutorial)
This tutorial shows how to implement a particle filter for tracking the location of an object from measurements of its relative bearing.

<br>
### [Reversible-Jump MCMC in Gen](rj/tutorial)
This tutorial shows how to use Gen's [automated involutive MCMC features](https://arxiv.org/abs/2007.09871) to implement reversible-jump proposals, for models that 
have unknown structure (and not just unknown parameters).

<br>
### [Modeling with TensorFlow code](tf-mnist/tutorial)
This tutorial shows how to write a generative function that invokes TensorFlow code, and how to perform basic supervised training of a generative function.

<br>
### [Proposals with PyTorch](pytorch-proposals/tutorial)
This tutorial extends our tutorial on [data-driven inference](data-driven-proposals/tutorial) by using PyTorch to build a neural network for amortized inference.

<br>
### [A Bottom-up Introduction to Gen](bottom-up-intro/tutorial)
This tutorial describes the reasoning behind some of the basic concepts in Gen.

<br>
### [Reasoning about Regenerate](regenerate/tutorial)
This tutorial explains some of the mathematical details of MCMC in Gen.
