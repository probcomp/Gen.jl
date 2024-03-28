# Run: julia --project make.jl
using Documenter, Gen

include("pages.jl")
makedocs(
    modules = [Gen],
    # linkcheck = true,
    doctest = false,
    clean = true,
    warnonly = true,
    # warnonly = [:missing_docs],
    format = Documenter.HTML(;
        assets = String["assets/header.js", "assets/header.css", "assets/theme.css"]
    ),
    sitename = "Gen.jl",
    pages = pages,
)

deploydocs(
    repo = "github.com/probcomp/Gen.jl.git",
    target = "build",
    dirname = "docs",
)
