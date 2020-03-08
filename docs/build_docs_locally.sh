#!/bin/sh

# run this script from the Gen/ directory, it will generate HTML documentation under docs/build

julia --project=docs/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs/ docs/make.jl
