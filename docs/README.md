# Gen.jl Documentation

- `pages.jl` to find skeleton of the Gen.jl documentation.
- `make.jl` to build the documentation website index.

The documentation is divided into three sections:
- Getting Started + Tutorials
- How-to Guides
- Reference Guides

# Developing

To build the docs, run `julia --make.jl`. Alternatively, start the Julia REPL, activate the `Project.toml` in this directory, then include `make.jl`. For debugging, consider setting `draft=true` in the `makedocs` function found in `make.jl`. This will avoid running the `@example` blocks when generating the tutorials.

Currently you must write the tutorial directly in the docs rather than in a source file. See `tutorials` for examples.

Code snippets must use the triple backtick with a label to run. The environment carries over so long as the labels match. Example:

```@example tutorial_1
x = rand()
```

```@example tutorial_1
print(x)
```
