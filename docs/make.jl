using Documenter, Gen

makedocs(
    format = :html,
    sitename = "Gen",
    modules = [Gen],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Tutorials" => "tutorials.md",
        "Modeling Languages and APIs" => [
            "Generative Functions" => "ref/gfi.md",
            "Probability Distributions" => "ref/distributions.md",
            "Built-in Modeling Language" => "ref/modeling.md",
            "Generative Function Combinators" => "ref/combinators.md",
            "Choice Maps" => "ref/choice_maps.md",
            "Selections" => "ref/selections.md",
            "Optimizing Trainable Parameters" => "ref/parameter_optimization.md",
            "Extending Gen" => "ref/extending.md"
         ],
        "Standard Inference Library" => [
            "Importance Sampling" => "ref/importance.md",
            "Markov chain Monte Carlo" => "ref/mcmc.md",
            "MAP Optimization" => "ref/map.md",
            "Particle Filtering" => "ref/pf.md",
            "Variational Inference" => "ref/vi.md",
            "Learning Generative Functions" => "ref/learning.md"
        ],
        "Internals" => [
            "Optimizing Trainable Parameters" => "ref/internals/parameter_optimization.md",
            "Modeling Language Implementation" => "ref/internals/language_implementation.md"
         ]
    ]
)

deploydocs(
    repo = "github.com/probcomp/Gen.jl.git",
    target = "build"
)
