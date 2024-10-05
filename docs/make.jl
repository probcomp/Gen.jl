# Run: julia --project make.jl
using Documenter, Gen

include("pages.jl")
makedocs(
    modules = [Gen],
    doctest = false,
    clean = true,
    warnonly = true,
    format = Documenter.HTML(;
        collapselevel=1,
    ),
    sitename = "Gen.jl",
    pages = pages,
    checkdocs=:exports,
    pagesonly=true,
)

deploydocs(
    repo = "github.com/probcomp/Gen.jl.git",
    target = "build",
    dirname = "docs",
    push_preview = true,
)
