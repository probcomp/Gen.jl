# [Saving Traces between REPL Sessions](@id how_to_serialize)

When saving a trace to disk, it is likely to run into issues using `Serialization.jl` since traces internally track function pointers (recall `Trace`s may contain function pointers). Instead, you can try the experimental [`GenSerialization.jl`](https://github.com/probcomp/GenSerialization.jl) which can discard function call data. This is most useful for check-pointing work (e.g. inference) between different REPL sessions on the same machine. 

Since `Serialization.jl` is used underneath, similar restrictions apply (see [`serialize`](https://docs.julialang.org/en/v1/stdlib/Serialization/#Serialization.serialize)). Please note we do not guarantee portability between different machines and have yet to extensively test this.

!!! note
    See the repository for warnings and limitations.


An example:

```julia
using Gen
using GenSerialization

@gen function model(p) 
    x ~ bernoulli(p)
end

trace = simulate(model, (0.2))
serialize("coin_flip.gen", trace)
```

This stores the trace in a file. Now to deserialize, run
```julia
saved_trace = deserialize("coin_flip.gen", model)
```
!!! note
    The same generative function used to save out the trace must be defined at runtime before deserialization.

