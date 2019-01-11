using Documenter, Gen

makedocs(
    format = :html,
    sitename = "Gen",
    modules = [Gen],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Tutorials" => "tutorials.md",
        "Guide" => "guide.md",
        "Reference" => [
            "Built-in Modeling Language" => "ref/modeling.md",
            "Generative Function Combinators" => "ref/combinators.md",
            "Assignments" => "ref/assignments.md",
            "Selections" => "ref/selections.md",
            "Inference Library" => "ref/inference.md",
            "Generative Function Interface" => "ref/gfi.md",
            "Probability Distributions" => "ref/distributions.md"
         ]
    ]
)

deploydocs(
    repo = "github.com/probcomp/Gen.git",
    target = "build"
)
