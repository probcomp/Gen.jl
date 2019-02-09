using Documenter, Gen

makedocs(
    format = :html,
    sitename = "Gen",
    modules = [Gen],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Tutorials" => "tutorials.md",
        "Language and API Reference" => [
            "Generative Functions" => "ref/gfi.md",
            "Simple Generative Functions" => "ref/distributions.md",
            "Built-in Modeling Language" => "ref/modeling.md",
            "Generative Function Combinators" => "ref/combinators.md",
            "Choice Maps" => "ref/choice_maps.md",
            "Selections" => "ref/selections.md",
            "Optimizing Trainable Parameters" => "ref/parameter_optimization.md",
            "Inference Library" => "ref/inference.md",
         ],
        "Internals" => [
            "Optimizing Trainable Parameters" => "ref/internals/parameter_optimization.md"
         ]
    ]
)

deploydocs(
    repo = "github.com/probcomp/Gen.git",
    target = "build"
)
