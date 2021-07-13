FROM            julia
MAINTAINER      MIT Probabilistic Computing Project

ADD             . /probcomp

ENV             JULIA_PKG_DEVDIR=/probcomp

RUN             julia -e 'using Pkg; Pkg.develop("Gen"); Pkg.test("Gen")'
