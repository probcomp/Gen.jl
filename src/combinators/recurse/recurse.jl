using FunctionalCollections: assoc, dissoc, PersistentHashMap
using DataStructures: PriorityQueue, dequeue!, enqueue!

#################
# recurse trace #
#################

struct RecurseTrace{S,T,U,V,W,X,Y,DV,DU,DW}
    gen_fn::GenerativeFunction
    production_traces::PersistentHashMap{Int,S}
    aggregation_traces::PersistentHashMap{Int,T}
    max_branch::Int
    score::Float64
    root_idx::Int
    num_has_choices::Int
end

get_gen_fn(trace::RecurseTrace) = trace.gen_fn

function get_args(trace::RecurseTrace)
    root_arg = get_args(trace.production_traces[trace.root_idx])[1]
    (root_arg, trace.root_idx)
end

function get_retval(trace::RecurseTrace)
    get_retval(trace.aggregation_traces[trace.root_idx])
end

get_score(trace::RecurseTrace) = trace.score

# TODO assumes there is no untraced randomness
project(trace::RecurseTrace, ::EmptyAddressSet) = 0.

##############################
# recurse assignment wrapper #
##############################

struct RecurseTraceAssignment <: Assignment
    trace::RecurseTrace
end

get_assmt(trace::RecurseTrace) = RecurseTraceAssignment(trace)

function Base.isempty(assmt::RecurseTraceAssignment)
    assmt.trace.num_has_choices == 0
end

get_address_schema(::Type{RecurseTraceAssignment}) = DynamicAddressSchema()

function get_subassmt(assmt::RecurseTraceAssignment,
                      addr::Tuple{Int,Val{:production}})
    idx = addr[1]
    if !haskey(assmt.trace.aggregation_traces, idx)
        EmptyAssignment()
    else
        get_assmt(assmt.trace.production_traces[idx])
    end
end

function get_subassmt(assmt::RecurseTraceAssignment,
                      addr::Tuple{Int,Val{:aggregation}})
    idx = addr[1]
    if !haskey(assmt.trace.aggregation_traces, idx)
        EmptyAssignment()
    else
        get_assmt(assmt.trace.aggregation_traces[idx])
    end
end

function get_subassmt(assmt::RecurseTraceAssignment, addr::Pair)
    _get_subassmt(assmt, addr)
end

function has_value(assmt::RecurseTraceAssignment, addr::Pair)
    _has_value(assmt, addr)
end

function get_value(assmt::RecurseTraceAssignment, addr::Pair)
    _get_value(assmt, addr)
end

get_values_shallow(assmt::RecurseTraceAssignment) = ()

function get_subassmts_shallow(assmt::RecurseTraceAssignment)
    production_iter = (((idx, Val(:production)), get_assmt(subtrace))
        for (idx, subtrace) in assmt.trace.production_traces)
    aggregation_iter = (((idx, Val(:aggregation)), get_assmt(subtrace))
        for (idx, subtrace) in assmt.trace.aggregation_traces)
    Iterators.flatten((production_iter, aggregation_iter))
end

# TODO when lightweight Gen functions properly declare their argument and return types, use:
# production_kern::GenerativeFunction{Tuple{V,Vector{U}},S}
# aggregation_kern::GenerativeFunction{W,T}

""""
    Recurse(production_kernel, aggregation_kernel, max_branch,
         ::Type{U}, ::Type{V}, ::Type{W}, ::Type{DV}, ::Type{DU}, ::Type{DW})

Constructor for recurse production and aggregation function.

Type parameters of `Recurse`: `S, T, U, V, W, DU, DV, DW, X, Y`
- DV must implement `isnodiff`
- DW must implement `isnodiff`
- Tuple{V, Vector{U}} <: X
- W <: Y

production kernel is a `GenerativeFunction{X,S}`
- input type: `U`
- argdiff type: `Union{NoArgDiff,DU}`
- trace type: `S`
- return type: `Tuple{V,Vector{U}}`
- retdiff type: `RecurseProductionRetDiff{DV,DU}`

aggregation kernel is a `GenerativeFunction{Y,T}`
- input type: `Tuple{V,Vector{W}}`
- argdiff: `RecurseAggregationRetDiff{Union{NoArgDiff,DV},DW}`
- trace type: `T`
- return type: `W`
- retdiff type: `DW`

recurse argdiff type: `Union{NoArgDiff,DU}`
recurse retdiff type: `Union{RecurseRetNoDiff,DW}`
"""
struct Recurse{S,T,U,V,W,X,Y,DV,DU,DW} <: GenerativeFunction{W,RecurseTrace{S,T,U,V,W,X,Y,DV,DU,DW}}
    production_kern::GenerativeFunction{X,S}
    aggregation_kern::GenerativeFunction{Y,T}
    max_branch::Int
end

function Recurse(production_kernel::GenerativeFunction{X,S},
                 aggregation_kernel::GenerativeFunction{Y,T},
                 max_branch::Int, ::Type{U}, ::Type{V}, ::Type{W},
                 ::Type{DV}, ::Type{DU}, ::Type{DW}) where {S,T,U,V,W,X,Y,DV,DU,DW}
    Recurse{S,T,U,V,W,X,Y,DV,DU,DW}(production_kernel, aggregation_kernel, max_branch)
end

# TODO
accepts_output_grad(::Recurse) = false

function get_child(parent::Int, child_num::Int, max_branch::Int)
    @assert child_num >= 1 && child_num <= max_branch
    (parent - 1) * max_branch + child_num + 1
end

function get_child_num(child::Int, max_branch::Int)
    @assert max_branch > 0
    (child - 2) % max_branch + 1
end

function get_parent(child::Int, max_branch::Int)
    @assert max_branch > 0
    child_num = get_child_num(child, max_branch)
    div(child - 1 - child_num, max_branch) + 1
end


"""
    RecurseProductionRetDiff{DV,DU}(dv::DV, dus::Dict{Int,DU})

Return value difference for production kernels used with `Recurse`.
If the number of children changed, there are only fields for retained children.
If a field does not appear for a given child in dus, then the u value being passed to that child has not changed.
`DV` must have method `isnodiff()`.
"""
struct RecurseProductionRetDiff{DV,DU}
    dv::DV
    dus::Dict{Int,DU} # map from child_num to retdiff
end

"""
    RecurseAggregationArgDiff{DV,DW}(dv::DV, dws::Dict{Int,DW})

Argument difference for aggregation kernels used with `Recurse`.
If the number of children changed, there are only fields for retained children.
If a field does not appear for a given child in dws, then the w value returned from that child has not changed.
`DV` must have method `isnodiff()`.
`DW` must have method `isnodiff()`.
"""
struct RecurseAggregationArgDiff{DV,DW}
    dv::Union{NoArgDiff,DV}
    dws::Dict{Int,DW} # map from child_num to argdiff

    function RecurseAggregationArgDiff{DV,DW}(dv::DV, dws::Dict{Int,DW}) where {DV,DW}
        @assert !isnodiff(dv) # otherwise, pass NoArgDiff
        new{DV,DW}(dv, dws)
    end

    function RecurseAggregationArgDiff{DV,DW}(dv::NoArgDiff, dws::Dict{Int,DW}) where {DV,DW}
        new{DV,DW}(dv, dws)
    end
end


"""
Indicates that the return value of the Recurse has not changed.
"""
struct RecurseRetNoDiff end
isnodiff(::RecurseRetNoDiff) = true

function get_num_children(production_trace)
    length(get_retval(production_trace)[2])
end

function get_production_input(gen_fn::Recurse{S,T,U,V,W}, cur::Int,
                              production_traces::AbstractDict{Int,S},
                              root_idx::Int, root_production_input::U) where {S,T,U,V,W}
    if cur == root_idx
        return root_production_input
    else
        parent = get_parent(cur, gen_fn.max_branch)
        child_num = get_child_num(cur, gen_fn.max_branch)
        # return type of parent is Tuple{V,Vector{U}}
        @assert haskey(production_traces, parent)
        return get_retval(production_traces[parent])[2][child_num]::U
    end
end

function get_aggregation_input(gen_fn::Recurse{S,T,U,V,W}, cur::Int,
                               production_traces::AbstractDict{Int,S},
                               aggregation_traces::AbstractDict{Int,T}) where {S,T,U,V,W}
    # requires that the corresponding production trace exists (for the v)
    # also requires that the child aggregation traces exist (for the w's)
    # does not require that this aggregation trace already exists
    vinput::V = get_retval(production_traces[cur])[1]
    num_children = get_num_children(production_traces[cur])
    winputs::Vector{W} = [
        get_retval(aggregation_traces[get_child(cur, i, gen_fn.max_branch)])
        for i=1:num_children]
    return (vinput, winputs)
end


function get_production_constraints(constraints::Assignment, cur::Int)
    get_subassmt(constraints, (cur, Val(:production)))
end

function get_aggregation_constraints(constraints::Assignment, cur::Int)
    get_subassmt(constraints, (cur, Val(:aggregation)))
end

##############
# generate #
##############

function generate(gen_fn::Recurse{S,T,U,V,W,X,Y,DV,DU,DW}, args::Tuple{U,Int},
                    constraints::Assignment) where {S,T,U,V,W,X,Y,DV,DU,DW}
    (root_production_input::U, root_idx::Int) = args
    production_traces = PersistentHashMap{Int,S}()
    aggregation_traces = PersistentHashMap{Int,T}()
    weight = 0.
    score = 0.
    num_has_choices = 0
    
    # production phase
    # does not matter in which order we visit (since children are inserted after parents)
    prod_to_visit = Set{Int}([root_idx]) 
    while !isempty(prod_to_visit)
        local subtrace::S
        local input::U
        cur = first(prod_to_visit)
        delete!(prod_to_visit, cur)
        input = get_production_input(gen_fn, cur, production_traces, root_idx, root_production_input)
        subconstraints = get_production_constraints(constraints, cur)
        (subtrace, subweight) = generate(gen_fn.production_kern, (input,), subconstraints)
        score += get_score(subtrace)
        production_traces = assoc(production_traces, cur, subtrace)
        weight += subweight
        children_inputs::Vector{U} = get_retval(subtrace)[2]
        for child_num in 1:length(children_inputs)
            push!(prod_to_visit, get_child(cur, child_num, gen_fn.max_branch))
        end
        if !isempty(get_assmt(subtrace))
            num_has_choices += 1
        end
    end

    # aggregation phase
    # visit children first
    agg_to_visit = sort(collect(keys(production_traces)), rev=true)
    for cur in agg_to_visit
        local subtrace::T
        local input::Tuple{V,Vector{W}}
        input = get_aggregation_input(gen_fn, cur, production_traces, aggregation_traces)
        subconstraints = get_aggregation_constraints(constraints, cur)
        (subtrace, subweight) = generate(gen_fn.aggregation_kern, input, subconstraints)
        score += get_score(subtrace)
        aggregation_traces = assoc(aggregation_traces, cur, subtrace)
        weight += subweight
        if !isempty(get_assmt(subtrace))
            num_has_choices += 1
        end
    end

    trace = RecurseTrace{S,T,U,V,W,X,Y,DV,DU,DW}(gen_fn,
                production_traces, aggregation_traces, gen_fn.max_branch,
                score, root_idx, num_has_choices)
    return (trace, weight)
end


################
# update #
################

struct NodeQueue{T}
    ordering::T
    pq::PriorityQueue{Int,Int}
end

function NodeQueue(ordering)
    pq = PriorityQueue{Int,Int}(ordering)
    NodeQueue(ordering, pq)
end

function Base.push!(queue::NodeQueue, node::Int)
    if !haskey(queue.pq, node)
        enqueue!(queue.pq, node, node)
    end
end
Base.pop!(queue::NodeQueue) = dequeue!(queue.pq)
Base.in(node::Int, queue::NodeQueue) = haskey(queue.pq, node)
Base.isempty(queue::NodeQueue) = isempty(queue.pq)


function recurse_unpack_constraints(constraints::Assignment)
    production_constraints = Dict{Int, Any}()
    aggregation_constraints = Dict{Int, Any}()
    for (addr, node) in get_subassmts_shallow(constraints)
        idx::Int = addr[1]
        if addr[2] == Val(:production)
            production_constraints[idx] = node
        elseif addr[2] == Val(:aggregation)
            aggregation_constraints[idx] = node
        else
            error("Unknown address: $addr")
        end
    end
    if length(get_values_shallow(constraints)) > 0
        error("Unknown address: $(first(get_values_shallow(constraints))[1])")
    end
    return (production_constraints, aggregation_constraints)
end

function dissoc_subtree!(discard::DynamicAssignment,
						 production_traces::PersistentHashMap{Int,S},
                         aggregation_traces::PersistentHashMap{Int,T},
                         root::Int, max_branch::Int) where {S,T}
    num_has_choices = 0
    production_subtrace = production_traces[root]
    aggregation_subtrace = aggregation_traces[root]
	set_subassmt!(discard, (root, Val(:production)), get_assmt(production_subtrace))
	set_subassmt!(discard, (root, Val(:aggregation)), get_assmt(aggregation_subtrace))
    if !isempty(get_assmt(production_subtrace))
        num_has_choices += 1
    end
    if !isempty(get_assmt(aggregation_subtrace))
        num_has_choices += 1
    end
    num_children = get_num_children(production_subtrace)
    removed_score = get_score(production_subtrace) + get_score(aggregation_subtrace)
    for child_num=1:num_children
        child = get_child(root, child_num, max_branch)
        (production_traces, aggregation_traces, child_removed_score, child_num_has_choices) = dissoc_subtree!(
            discard, production_traces, aggregation_traces, child, max_branch)
        removed_score += child_removed_score
        num_has_choices += child_num_has_choices
    end
    production_traces = dissoc(production_traces, root)
    aggregation_traces = dissoc(aggregation_traces, root)
    return (production_traces, aggregation_traces, removed_score, num_has_choices)
end

function get_production_argdiff(production_retdiffs::Dict{Int,RecurseProductionRetDiff{DV,DU}},
                                root_idx::Int, root_argdiff::Union{NoArgDiff,DU}, cur::Int,
                                max_branch::Int) where {DV,DU}
    if cur == root_idx
        return root_argdiff
    else
        parent = get_parent(cur, max_branch)
        if !haskey(production_retdiffs, parent)
            return noargdiff
        else
            @assert haskey(production_retdiffs, parent)
            child_num = get_child_num(cur, max_branch)
            dus = production_retdiffs[parent].dus
            if haskey(dus, child_num)
                return dus[child_num]::DU
            else
                return noargdiff
            end
        end
    end
end

function get_aggregation_argdiff(production_retdiffs::Dict{Int,RecurseProductionRetDiff{DV,DU}},
                                 aggregation_retdiffs::Dict{Int,DW},
                                 idx_to_prev_num_children::Dict{Int,Int},
                                 production_traces, cur::Int) where {DU,DV,DW}
    if haskey(production_retdiffs, cur)
        production_retdiff = production_retdiffs[cur]
        if isnodiff(production_retdiff.dv)
            dv = noargdiff
        else
            dv = production_retdiff.dv::DV
        end
    else
        dv = noargdiff
    end
    new_num_children = get_num_children(production_traces[cur])
    if haskey(idx_to_prev_num_children, cur)
        prev_num_children = idx_to_prev_num_children[cur]
    else
        prev_num_children = new_num_children
    end
    dws = Dict{Int,DW}() # values have type DW
    for child_num=1:min(prev_num_children,new_num_children)
        if haskey(aggregation_retdiffs, child_num)
            dws[child_num] = aggregation_retdiffs[child_num]::DW
        end
    end
    RecurseAggregationArgDiff{DV,DW}(dv, dws)
end

function update(trace::RecurseTrace{S,T,U,V,W,X,Y,DV,DU,DW},
                new_args::Tuple{U,Int},
                root_argdiff::Union{NoArgDiff,DU}, 
                constraints::Assignment) where {S,T,U,V,W,X,Y,DV,DU,DW}
    gen_fn = get_gen_fn(trace)
    (root_production_input::U, root_idx::Int) = new_args
    if root_idx != get_args(trace)[2]
        error("Cannot change root_idx argument") 
    end

    production_traces = trace.production_traces
    aggregation_traces = trace.aggregation_traces
    (production_constraints, aggregation_constraints) = recurse_unpack_constraints(constraints)
    discard = DynamicAssignment()

    # initial score from previous trace
    score = trace.score
    weight = 0.
    num_has_choices = trace.num_has_choices

    # initialize set of production nodes to visit
    # visit nodes in forward order (starting with root)
    prod_to_visit = NodeQueue(Base.Order.Forward)
    for node in keys(production_constraints)
        push!(prod_to_visit, node)
    end

    # if the root argdiff is not noargdiff then we visit the root production node
    if root_argdiff != noargdiff && !(root_idx in prod_to_visit)
        push!(prod_to_visit, root_idx)
    end

    # initialize set of aggregation nodes to visit
    # visit nodes in reverse order (starting with leaves)
    agg_to_visit = NodeQueue(Base.Order.Reverse)
    for node in keys(aggregation_constraints)
        push!(agg_to_visit, node)
    end

    # only store for nodes that are retained
    # if a value is not present for a retained node, then the number of children of that node did not change
    idx_to_prev_num_children = Dict{Int,Int}()
    
    # production phase
    production_retdiffs = Dict{Int,RecurseProductionRetDiff{DV,DU}}() # elements with no difference are ommitted
    while !isempty(prod_to_visit)
        local subtrace::S

        cur = pop!(prod_to_visit)
        subconstraints = get_production_constraints(constraints, cur)
        input = (get_production_input(gen_fn, cur, production_traces, root_idx, root_production_input)::U,)

        if haskey(production_traces, cur)
            # the node exists already
            local subargdiff::Union{NoArgDiff,DU}
            local subretdiff::RecurseProductionRetDiff{DV,DU}

            # get argdiff for this production node
            subargdiff = get_production_argdiff(production_retdiffs, root_idx,
                                                root_argdiff, cur, gen_fn.max_branch)

            # call update on production kernel
            prev_subtrace = production_traces[cur]
            (subtrace, subweight, subretdiff, subdiscard) = update(
                prev_subtrace, input, subargdiff, subconstraints)
            prev_num_children = get_num_children(production_traces[cur])
            new_num_children = length(get_retval(subtrace)[2])
            idx_to_prev_num_children[cur] = prev_num_children
            set_subassmt!(discard, (cur, Val(:production)), subdiscard)
            production_retdiffs[cur] = subretdiff

            # update trace, weight, and score
            production_traces = assoc(production_traces, cur, subtrace)
            weight += subweight
            score += subweight

            # update num_has_choices
            if !isempty(get_assmt(prev_subtrace)) && isempty(get_assmt(subtrace))
                num_has_choices -= 1
            elseif isempty(get_assmt(prev_subtrace)) && !isempty(get_assmt(subtrace))
                num_has_choices += 1
            end

            # delete children (and their descendants), both production and aggregation nodes
            for child_num=new_num_children+1:prev_num_children
                child = get_child(cur, child_num, gen_fn.max_branch)
                set_subassmt!(discard, (child, Val(:production)), get_assmt(production_traces[child]))
                set_subassmt!(discard, (child, Val(:aggregation)), get_assmt(aggregation_traces[child]))
                (production_traces, aggregation_traces, removed_score, removed_num_has_choices) = dissoc_subtree!(
                    discard, production_traces, aggregation_traces, child, gen_fn.max_branch)
                score -= removed_score
                num_has_choices -= removed_num_has_choices
                weight -= removed_score
            end

            # mark new children for processing
            for child_num=prev_num_children+1:new_num_children
                child = get_child(cur, child_num, gen_fn.max_branch)
                push!(prod_to_visit, child)
            end

            # maybe mark existing children for processing, if they have a custom retdiff
            # (otherwise they do not change)
            for child_num in keys(subretdiff.dus)
                @assert child_num <= min(prev_num_children,new_num_children)
                child = get_child(cur, child_num, gen_fn.max_branch)
                push!(prod_to_visit, child)
            end

            # mark corresponding aggregation node for processing if v has
            # changed, or if the number of children has changed
            if !isnodiff(subretdiff.dv) || prev_num_children != new_num_children
                push!(agg_to_visit, cur)
            end

        else
            # the node does not exist already (and none of its children exist either)
            (subtrace, ) = generate(gen_fn.production_kern, input, subconstraints)

            # update trace, weight, and score
            production_traces = assoc(production_traces, cur, subtrace)
            weight += get_score(subtrace)
            score += get_score(subtrace)

            # update num_has_choices
            if !isempty(get_assmt(subtrace))
                num_has_choices += 1
            end

            # mark corresponding aggregation node for processsing
            push!(agg_to_visit, cur)

            # mark children (which are all new) for processing
            for child_num=1:get_num_children(subtrace)
                child = get_child(cur, child_num, gen_fn.max_branch)
                push!(prod_to_visit, child)
            end
        end
    end

    # aggregation phase

    # reasons to visit an aggregation node:
    # - it is new
    # - it is directly constrained
    # - its corresponding production node has !isnodiff(dv)
    # - one or more of of its children has a !isnodiff(dw)
    # note: we already deleted aggregation nodes that are to be deleted above

    aggregation_retdiffs = Dict{Int,DW}() # isnodiff(dw)'s are ommitted
    local subtrace::T
    local subargdiff::RecurseAggregationArgDiff{DV,DW}
    local subretdiff::DW
    local retdiff::Union{RecurseRetNoDiff,DW}
    retdiff = RecurseRetNoDiff()
    while !isempty(agg_to_visit)

        cur = pop!(agg_to_visit)
        subconstraints = get_aggregation_constraints(constraints, cur)
        input::Tuple{V,Vector{W}} = get_aggregation_input(
            gen_fn, cur, production_traces, aggregation_traces)

        # if the node exists already
        if haskey(aggregation_traces, cur)

            # get argdiff for this aggregation node
            subargdiff = get_aggregation_argdiff(production_retdiffs, aggregation_retdiffs,
                                                 idx_to_prev_num_children, production_traces, cur)

            # call update on aggregation kernel
            prev_subtrace = aggregation_traces[cur]
            (subtrace, subweight, subretdiff, subdiscard) = update(
                prev_subtrace, input, subargdiff, subconstraints)

            # update trace, weight, and score, and discard
            aggregation_traces = assoc(aggregation_traces, cur, subtrace)
            weight += subweight
            score += subweight
            set_subassmt!(discard, (cur, Val(:aggregation)), subdiscard)

            # update num_has_choices
            if !isempty(get_assmt(prev_subtrace)) && isempty(get_assmt(subtrace))
                num_has_choices -= 1
            elseif isempty(get_assmt(prev_subtrace)) && !isempty(get_assmt(subtrace))
                num_has_choices += 1
            end
 
            # take action based on our subretdiff
            if cur == root_idx
                retdiff = subretdiff
            else
                if !isnodiff(subretdiff)
                    aggregation_retdiffs[cur] = subretdiff
                    push!(agg_to_visit, get_parent(cur, gen_fn.max_branch))
                end
            end

        # if the node does not exist (but its children do, since we created them already)
        else
            (subtrace, _) = generate(gen_fn.aggregation_kern, input, subconstraints)

            # update trace, weight, and score
            aggregation_traces = assoc(aggregation_traces, cur, subtrace)
            weight += get_score(subtrace)
            score += get_score(subtrace)

            # update num_has_choices
            if !isempty(get_assmt(subtrace))
                num_has_choices += 1
            end

            # the parent should have been marked during production phase
            @assert cur != 1
            @assert get_parent(cur, gen_fn.max_branch) in agg_to_visit
        end
        
    end

    new_trace = RecurseTrace{S,T,U,V,W,X,Y,DV,DU,DW}(gen_fn,
        production_traces, aggregation_traces, gen_fn.max_branch,
        score, root_idx, num_has_choices)
    
    return (new_trace, weight, retdiff, discard)
end

export Recurse
export RecurseAggregationArgDiff, RecurseProductionRetDiff
