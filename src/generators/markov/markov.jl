using FunctionalCollections: PersistentVector, push, assoc

####################
# markov generator # 
####################

struct Markov{T,U} <: Generator{PersistentVector{T},VectorTrace{T,U}}
    kernel::Generator{T,U}
end

function markov(kernel::Generator{T,U}) where {T,U}
    kernel_arg_types = get_static_argument_types(kernel)
    println(kernel_arg_types)
    if length(kernel_arg_types) != 3 || kernel_arg_types[1] != Int
        error("markov requires a kernel with arguments (t::Int, state, params)")
    end
    Markov{T,U}(kernel)
end

function get_static_argument_types(markov::Markov)
    kernel_arg_types = get_static_argument_types(markov.kernel)
    state_type = kernel_arg_types[2]
    params_type = kernel_arg_types[3]
    # 1 total number of time steps
    # 2 initial state (this must also be the return type of the kernel, not currently checked)
    # 3 parameters (shared across all time steps)
    [Int, state_type, params_type]
end

function generate(gen::Markov{T,U}, args, constraints) where {T,U}
    # NOTE: could be strict and check there are no extra constraints
    # probably we want to have this be an option that can be turned on or off?
    (len, init, params) = args
    states = Vector{T}(undef, len)
    subtraces = Vector{U}(undef, len)
    weight = 0.
    score = 0.
    state::T = init
    is_empty = false
    for key=1:len
        if has_internal_node(constraints, key)
            node = get_internal_node(constraints, key)
        else
            node = EmptyChoiceTrie()
        end
        kernel_args = (key, state, params)
        (subtrace::U, w) = generate(gen.kernel, kernel_args, node)
        subtraces[key] = subtrace
        weight += w
        call = get_call_record(subtrace)
        states[key] = call.retval
        score += call.score
        is_empty = is_empty && !has_choices(subtrace)
    end
    call = CallRecord(score, PersistentVector{T}(states), args)
    trace = VectorTrace{T,U}(PersistentVector{U}(subtraces), call, is_empty)
    (trace, weight)
end

function simulate(gen::Markov{T,U}, args) where {T,U}
    (trace, weight) = generate(gen, args, EmptyChoiceTrie())
    trace
end

struct MarkovChange
    len_changed::Bool
    init_changed::Bool
    params_changed::Bool
end

function extend(gen::Markov{T,U}, args, change::MarkovChange, trace::VectorTrace{T,U},
                constraints) where {T,U}
    (len, init_state, params) = args
    prev_call = get_call_record(trace)
    prev_args = prev_call.args
    prev_len = prev_args[1]
    if len < prev_len
        error("Extend cannot remove addresses or namespaces")
    end
    if change.params_changed
        to_visit = Set{Int}(1:len)
    else
        to_visit = Set{Int}(prev_len+1:len)
    end
    if change.init_changed
        push!(to_visit, 1)
    end
    for (key::Int, _) in get_internal_nodes(constraints)
        push!(to_visit, key)
    end
    subtraces::PersistentVector{U} = trace.subtraces
    states::PersistentVector{T} = trace.call.retval
    weight = 0.
    score = prev_call.score
    is_empty = !has_choices(trace)
    for key in sort(collect(to_visit))
        state = key > 1 ? states[key-1] : init_state
        kernel_args = (key, state, params)
        if has_internal_node(constraints, key)
            node = get_internal_node(constraints, key)
        else
            node = EmptyChoiceTrie()
        end
        local call::CallRecord{T}
        if key > prev_len
            (subtrace::U, w) = generate(gen.kernel, kernel_args, node)
            call = get_call_record(subtrace)
            score += call.score
            states = push(states, call.retval)
            @assert length(states) == key
            subtraces = push(subtraces, subtrace)
            @assert length(subtraces) == key
        else
            prev_subtrace::U = subtraces[key]
            prev_score = get_call_record(prev_subtrace).score
            (subtrace, w, retchange) = extend(gen.kernel, kernel_args, prev_subtrace, node)
            call = get_call_record(subtrace)
            score += call.score - prev_score
            @assert length(states) == key
            subtraces = assoc(subtraces, key, subtrace)
            states = assoc(states, key, call.retval)
        end
        weight += w
        is_empty = is_empty && !has_choices(subtrace)
    end
    call = CallRecord(score, states, args)
    trace = VectorTrace(subtraces, call, is_empty)
    (trace, weight, nothing)
end

export markov
export MarkovChange
