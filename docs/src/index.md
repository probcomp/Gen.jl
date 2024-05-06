# Gen.jl

*A general-purpose probabilistic programming system with programmable inference, embedded in Julia*

- What does Gen provide.
!!! note
    `Gen.jl` is still under active development. If you find a a bug or wish to share ideas for improvement, feel free to visit the Github site or contact at us here.

## Installation

The Gen package can be installed with the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and then run:
```
pkg> add Gen
```
!!! note
    Alternatively,

    ```julia
    julia> import Pkg; Pkg.add("Gen")
    ```

To test the installation locally, you can run the tests with:
```julia
using Pkg; Pkg.test("Gen")
```

## Getting Started
!!! warning "Tutorial Redirect"
    We are in the process of moving tutorial docs around. For more stable tutorial docs, see [these tutorials](https://www.gen.dev/tutorials/) instead. See more examples [here](https://github.com/probcomp/GenExamples.jl).

To see a overview of the package, check out the [examples](getting_started/linear_regression.md). For a deep-dive on how to do inference with Gen.jl, check out the [tutorials](@ref introduction_to_modeling_in_gen). 

```@contents
Pages = [
    "getting_started/linear_regression.md",
    ]
Depth = 2
```

## Contributing
See the [Developer's Guide](https://gen.dev) on how to contribute to the Gen ecosystem.

## Supporting and Citing
This repo is part of ongoing research at [ProbComp](http://probcomp.csail.mit.edu) and may later include new experimental  (for the better)! If you use Gen for your work, please consider citing us:

```bibtex
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