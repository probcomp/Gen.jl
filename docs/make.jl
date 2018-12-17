using Documenter, Gen

makedocs(
    format = :html,
    sitename = "Gen",
    modules = [Gen],
    pages = [
        "index.md",
        "documentation.md"
    ]
)

deploydocs(
    repo = "github.com/probcomp/Gen.git",
    target = "build"
)
