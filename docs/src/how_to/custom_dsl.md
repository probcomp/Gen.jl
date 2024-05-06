# How to Write a Custom Modeling Languages

Gen can be extended with new modeling languages by implementing new generative function types, and constructors for these types that take models as input.
This typically requires implementing the entire generative function interface, and is advanced usage of Gen.


Gen is designed for extensibility.
To implement behaviors that are not directly supported by the existing modeling languages, users can implement `black-box' generative functions directly, without using built-in modeling language.
These generative functions can then be invoked by generative functions defined using the built-in modeling language.
This includes several special cases:

- Extending Gen with custom gradient computations

- Extending Gen with custom incremental computation of return values

- Extending Gen with new modeling languages.