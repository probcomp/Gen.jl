"""
    (traces, log_norm_weights, lml_est) = enumerative_inference(
        model::GenerativeFunction, model_args::Tuple,
        observations::ChoiceMap, choice_vol_iter
    )

Run enumerative inference over a `model`, given `observations` and an iterator over 
choice maps and their associated log-volumes (`choice_vol_iter`), specifying the
choices to be iterated over. An iterator over a grid of choice maps and log-volumes
can be constructed with [`choice_vol_grid`](@ref).

Return an array of traces and associated log-weights with the same shape as 
`choice_vol_iter`. The log-weight of each trace is normalized, and corresponds
to the log probability of the volume of sample space that the trace represents.
Also return an estimate of the log marginal likelihood of the observations (`lml_est`).

All addresses in the `observations` choice map must be sampled by the model when
given the model arguments. The same constraint applies to choice maps enumerated
over by `choice_vol_iter`, which must also avoid sharing addresses with the 
`observations`.
"""
function enumerative_inference(
    model::GenerativeFunction{T,U}, model_args::Tuple,
    observations::ChoiceMap, choice_vol_iter::I
) where {T,U,I}
    if Base.IteratorSize(I) isa Base.HasShape
        traces = Array{U}(undef, size(choice_vol_iter))
        log_weights = Array{Float64}(undef, size(choice_vol_iter))
    elseif Base.IteratorSize(I) isa Base.HasLength
        traces = Vector{U}(undef, length(choice_vol_iter))
        log_weights = Vector{Float64}(undef, length(choice_vol_iter))
    else
        choice_vol_iter = collect(choice_vol_iter)
        traces = Vector{U}(undef, length(choice_vol_iter))
        log_weights = Vector{Float64}(undef, length(choice_vol_iter))
    end
    for (i, (choices, log_vol)) in enumerate(choice_vol_iter)
        constraints = merge(observations, choices)
        (traces[i], log_weight) = generate(model, model_args, constraints)
        log_weights[i] = log_weight + log_vol
    end
    log_total_weight = logsumexp(log_weights)
    log_normalized_weights = log_weights .- log_total_weight
    return (traces, log_normalized_weights, log_total_weight)
end

"""
    choice_vol_grid((addr, vals, [support, dims])::Tuple...; anchor=:midpoint)

Given tuples of the form `(addr, vals, [support, dims])`, construct an iterator
over tuples of the form `(choices::ChoiceMap, log_vol::Real)` via grid enumeration. 

Each `addr` is an address of a random choice, and `vals` are the corresponding 
values or intervals to enumerate over. The (optional) `support` denotes whether
each random choice is `:discrete` (default) or `:continuous`. This controls how
the grid is constructed:
- `support = :discrete`: The grid iterates over each value in `vals`.
- `support = :continuous` and `dims == Val(1)`: The grid iterates over the
  anchors of 1D intervals whose endpoints are given by `vals`.
- `support = :continuous` and `dims == Val(N)` where `N` > 1:  The grid iterates
  over the anchors of multi-dimensional regions defined `vals`, which is a tuple
  of interval endpoints for each dimension. 
Continuous choices are assumed to have `dims = Val(1)` dimensions by default.
The `anchor` keyword argument controls which point in each interval is used as
the anchor (`:left`, `:right`, or `:midpoint`).

The log-volume `log_vol` associated with each set of `choices` in the grid is given
by the log-product of the volumes of each continuous region used to construct those
choices. If all addresses enumerated over are `:discrete`, then `log_vol = 0.0`.
"""
function choice_vol_grid(grid_specs::Tuple...; anchor::Symbol=:midpoint)
    val_iter = (expand_grid_spec_to_values(spec...; anchor=anchor) 
                for spec in grid_specs)
    val_iter = Iterators.product(val_iter...)
    vol_iter = (expand_grid_spec_to_volumes(spec...) for spec in grid_specs)
    vol_iter = Iterators.product(vol_iter...)
    choice_vol_iter = Iterators.map(zip(val_iter, vol_iter)) do (vals, vols)
        return (choicemap(vals...), sum(vols))
    end
    return choice_vol_iter
end

function expand_grid_spec_to_values(
    addr, vals, support::Symbol = :discrete, dims::Val{N} = Val(1);
    anchor::Symbol = :midpoint
) where {N}
    if support == :discrete
        return ((addr, v) for v in vals) 
    elseif support == :continuous && N == 1
        if anchor == :left
            vals = @view(vals[begin:end-1])
        elseif anchor == :right
            vals = @view(vals[begin+1:end])
        else
            vals = @view(vals[begin:end-1]) .+ (diff(vals) ./ 2)
        end
        return ((addr, v) for v in vals)
    elseif support == :continuous && N > 1
        @assert length(vals) == N "Dimension mismatch between `vals` and `dims`"
        vals = map(vals) do vs
            if anchor == :left
                vs = @view(vs[begin:end-1])
            elseif anchor == :right
                vs = @view(vs[begin+1:end])
            else
                vs = @view(vs[begin:end-1]) .+ (diff(vs) ./ 2)
            end
            return vs
        end
        return ((addr, v) for v in Iterators.product(vals...))
    else
        error("Support must be :discrete or :continuous")
    end
end

function expand_grid_spec_to_volumes(
    addr, vals, support::Symbol = :discrete, dims::Val{N} = Val(1)
) where {N}
    if support == :discrete
        return zeros(length(vals))
    elseif support == :continuous && N == 1
        return log.(diff(vals))
    elseif support == :continuous && N > 1
        @assert length(vals) == N "Dimension mismatch between `vals` and `dims`"
        diffs = Iterators.product((log.(diff(vs)) for vs in vals)...)
        return (sum(ds) for ds in diffs)
    else
        error("Support must be :discrete or :continuous")
    end
end

export enumerative_inference, choice_vol_grid