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
        "Reference" => "ref.md"
    ]
)

deploydocs(
    repo = "github.com/probcomp/Gen.git",
    target = "build"
)
