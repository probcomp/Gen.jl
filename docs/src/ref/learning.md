# Learning Generative Functions

Learning and inference are closely related concepts, and the distinction between the two is not always clear.
Often, **learning** refers to inferring long-lived unobserved quantities that will be reused across many problem instances (like a dynamics model for an entity that we are trying to track), whereas **inference** refers to inferring shorter-lived quantities (like a specific trajectory of a specific entity at a specific time).
Specifically, we treat learning as the way to use data to automatically generate **models** of the world, or to automatically fill in unknown parameters in hand-coded models.

There are many different types of learning---we could aim to learn the discrete structure of the model, just a few numerical parameters, or weights in a neural network.
Also, we could do Bayesian learning in which we seek a probability distribution on possible models, or we could seek just the best model, as measured by e.g. **maximum likelihood**.
This section focuses on learning the **trainable parameters** of a generative function, which are numerical quantities with respect to which generative functions are able to report gradients of their (log) probability density functio of their density function
See [`Trainable Parameters`] for more information on trainable parameters.

There are two settings we could be in, when trying to learn the parameters of a generative function.
If our observed data contains values for all of the random choices made by the generative function, this is called **learning from complete data**, and is a relatively straightforward task.
If our observed data is missing values for some random choices (either because the value happened to be missing, or because it was too expensive to acquire it, or because it is an inherently unmeasurable quantity), this is called **learning from incomplete data**, and is a substantially harder task.
Gen provides programming primitives and design patterns for both tasks.

## Learning from Complete Data

If we are given a collection of traces of a generative function, it is relatively straightforward to update the parameters of the function to make the trace more likely.
This is how we represent **maximum likelihood** learning in Gen.
The design pattern is very simple.
We obtain 


## Learning from Incomplete Data
