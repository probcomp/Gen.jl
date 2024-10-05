# [Extending Gen](@id extending_gen_howto)

Gen is designed for extensibility. To implement behaviors that are not directly supported by the existing modeling languages, users can implement `black-box' generative functions that directly implement the [generative function interface](@ref gfi). These generative functions can then be invoked by generative functions defined using the built-in modeling language.

The following how-tos describe various ways of extending Gen:

- [Adding custom distributions](@ref custom_distributions_howto)
- [Custom incremental computation of return values](@ref custom_incremental_computation_howto)
- [Custom gradient computations](@ref custom_gradients_howto)
- [Implementing custom generative functions from scratch](@ref custom_gen_fns_howto)
 
Gen can also be extended with entirely new modeling languages by implementing new generative function types, and constructors for these types that take models as input. This typically requires implementing the entire generative function interface, and is advanced usage of Gen.
