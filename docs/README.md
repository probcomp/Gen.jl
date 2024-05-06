# Website Docs
- `pages.jl` to find skeleton of website.
- `make.jl` to build the website index.

The docs are divided in roughly four sections:
- Getting Started + Tutorials
- How-to Guides
- API = Modeling API + Inference API
- Explanations + Internals


# Developing
To build the docs, run `julia --make.jl` or alternatively startup the Julia REPL and include `make.jl`. For debugging, consider setting `draft=true` in the `makedocs` function found in `make.jl`.
Currently you must write the tutorial directly in the docs rather than a source file (e.g. Quarto). See `getting_started` or `tutorials` for examples.

Code snippets must use the triple backtick with a label to run. The environment carries over so long as the labels match. Example:

```@example tutorial_1
x = rand()
```

```@example tutorial_1
print(x)
```