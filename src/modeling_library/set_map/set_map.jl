include("multiset.jl")

export SetMap

struct SetTrace{ArgType, RetType, TraceType} <: Trace
    gen_fn::GenerativeFunction
    subtraces::PersistentHashMap{ArgType, TraceType}
    args::Tuple
    score::Float64
    noise::Float64
end
function get_choices(trace::SetTrace)
    # TODO: specialized choicemap type
    c = choicemap()
    for (arg, tr) in trace.subtraces
        set_subtree!(c, arg, get_choices(tr))
    end
    c
end
get_retval(trace::SetTrace) = setmap(((_, tr),) -> get_retval(tr), trace.subtraces)
get_args(trace::SetTrace) = trace.args
get_score(trace::SetTrace) = trace.score
get_gen_fn(trace::SetTrace) = trace.gen_fn
project(trace::SetTrace, ::EmptyAddressTree) = trace.noise
Base.getindex(tr::SetTrace, address) = tr.subtraces[address][]
Base.getindex(tr::SetTrace, address::Pair) = tr.subtraces[address.first][address]

struct SetMap{RetType, TraceType} <: GenerativeFunction{MultiSet{RetType}, SetTrace{<:Any, RetType, TraceType}}
    kernel::GenerativeFunction{RetType, T} where {T >: TraceType}
    function SetMap{RetType, TraceType}(kernel::GenerativeFunction{RetType, T} where {T >: TraceType}) where {RetType, TraceType}
        new{RetType, TraceType}(kernel)
    end
end
function SetMap(kernel::GenerativeFunction{RetType, TraceType}) where {RetType, TraceType}
    SetMap{RetType, get_trace_type(kernel)}(kernel)
end

has_argument_grads(gf::SetMap) = has_argument_grads(gf.kernel)
accepts_output_grad(gf::SetMap) = accepts_output_grad(gf.kernel)

function simulate(sm::SetMap{RetType, TraceType}, (set,)::Tuple{<:AbstractSet{ArgType}}) where {RetType, TraceType, ArgType}
    subtraces = PersistentHashMap{ArgType, TraceType}()
    score = 0.
    noise = 0.
    for item in set
        subtr = simulate(sm.kernel, (item,))
        subtraces = assoc(subtraces, item, subtr)
        score += get_score(subtr)
        noise += project(subtr, EmptyAddressTree())
    end
    return SetTrace{ArgType, RetType, TraceType}(sm, subtraces, (set,), score, noise)
end

function generate(sm::SetMap{RetType, TraceType}, (set,)::Tuple{<:AbstractSet{ArgType}}, constraints::ChoiceMap) where {ArgType, RetType, TraceType}
    subtraces = PersistentHashMap{ArgType, TraceType}()
    score = 0.
    weight = 0.
    noise = 0.
    for item in set
        constraint = get_subtree(constraints, item)
        subtr, wt = generate(sm.kernel, (item,), constraint)
        weight += wt
        noise += project(subtr, EmptyAddressTree())
        subtraces = assoc(subtraces, item, subtr)
        score += get_score(subtr)
    end
    return (SetTrace{ArgType, RetType, TraceType}(sm, subtraces, (set,), score), weight, noise)
end

# TODO: handle argdiffs
function update(tr::SetTrace{ArgType, RetType, TraceType}, (set,)::Tuple, ::Tuple{<:Diff}, spec::UpdateSpec, ext_const_addrs::Selection) where {ArgType, RetType, TraceType}
    new_subtraces = PersistentHashMap{ArgType, TraceType}()
    discard = choicemap()
    weight = 0.
    score = 0.
    noise = 0.
    for item in set
        if item in keys(tr.subtraces)
            (new_tr, wt, retdiff, this_discard) = update(
                tr.subtraces[item], (item,),
                (UnknownChange(),),
                get_subtree(spec, item),
                get_subtree(ext_const_addrs, item)
            )
            new_subtraces = assoc(new_subtraces, item, new_tr)
            score += get_score(new_tr)
            noise += project(new_tr, EmptyAddressTree())
            weight += wt
            set_subtree!(discard, item, this_discard)
        else
            tr, weight = generate(tr.gen_fn.kernel, (item,), get_subspec(spec, item))
            score += get_score(tr)
            noise += project(tr, EmptyAddressTree())
            new_subtraces = assoc(new_subtraces, item, tr)
        end
    end
    for (item, tr) in tr.subtraces
        if !(item in set)
            ext_const = get_subtree(ext_const_addrs, item)
            weight -= project(tr, addrs(get_selected(get_choices(tr), ext_const)))
            set_subtree!(discard, item, get_choices(tr))
        end
    end
    tr = SetTrace{ArgType, RetType, TraceType}(tr.gen_fn, new_subtraces, (set,), score, noise)
    return (tr, weight, UnknownChange(), discard)
end