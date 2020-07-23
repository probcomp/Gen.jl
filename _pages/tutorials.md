---
permalink: /tutorials/
title: "Gen Tutorials"
layout: splash
---

<br>
# Gen Tutorials

The tutorials below were generated from Jupyter notebooks, which are available as part of a Docker container, at [Gen Quickstart](https://github.com/probcomp/gen-quickstart).
Some tutorials have not yet been rendered into HTML.

### [Introduction to Modeling in Gen](intro-to-modeling/Introduction to Modeling in Gen)
This tutorial introduces the basics of modeling in Gen. It shows how to perform inference using generic inference algorithms. It does not explore custom inference programming.

### Modeling with Black-box Julia code
This tutorial shows how 'black-box' code like algorithms and simulators can be included in probabilistic models that are expressed as generative functions.

### Modeling with TensorFlow code
This tutorial shows how to write a generative function that invokes TensorFlow code, and how to perform basic supervised training of a generative function.

### Basics of Iterative Inference in Gen
This tutorial introduces the basics of inference programming in Gen using iterative inference programs, which include Markov chain Monte Carlo algorithms.

### Data-Driven Proposals in Gen
Data-driven proposals use information in the observed data set to choose the proposal distibution for latent variables in a generative model. 
This tutorial shows you how to use custom data-driven proposals to accelerate Monte Carlo inference. 

### [Scaling with Combinators and the Static Modeling Language](scaling-with-combinators/Scaling with Combinators and the Static Modeling Language)
This tutorial shows how generative function combinators and the static modeling language are used to achieve good asymptotic scaling time of inference algorithms.

### [Particle Filtering in Gen](particle-filtering-in-gen/Particle Filtering in Gen)
This tutorial shows how to implement a particle filter for tracking the location of an object from measurements of its relative bearing.

### [A Bottom-up Introduction to Gen](bottom-up/A Bottom-Up Introduction to Gen)
This tutorial describes the reasoning behind some of the basic concepts in Gen.

### [Reasoning about Regenerate](Reasoning+About+Regenerate)
This tutorial explains some of the mathematical details of MCMC in Gen.
