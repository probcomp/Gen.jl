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
    if length(kernel_arg_types) < 3 || kernel_arg_types[1] != Int
        error("markov requires a kernel with arguments (t::Int, state, params...)")
    end
    Markov{T,U}(kernel)
end

function get_static_argument_types(markov::Markov)
    kernel_arg_types = get_static_argument_types(markov.kernel)
    state_type = kernel_arg_types[2]
    params_types = kernel_arg_types[3:end]
    # 1 total number of time steps
    # 2 initial state (this must also be the return type of the kernel, not currently checked)
    # 3 parameters (shared across all time steps)
    [Int, state_type, params_types...]
end

function unpack_args(args::Tuple)
    len = args[1]
    init = args[2]
    params = args[3:end]
    (len, init, params)
end


function check_length(len::Int)
    if len < 1
        error("markov got length of $len < 1")
    end
end

############
# generate #
############

function generate(gen::Markov{T,U}, args, constraints) where {T,U}
    # NOTE: could be strict and check there are no extra constraints
    # probably we want to have this be an option that can be turned on or off?
    (len, init, params) = unpack_args(args)
    check_length(len)
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
        kernel_args = (key, state, params...)
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

##################################
# update, fix_update, and extend #
##################################

# TODO update
# TODO fix_update

struct MarkovChange
    len_changed::Bool
    init_changed::Bool
    params_changed::Bool
end

function extend(gen::Markov{T,U}, args, change::Nothing, trace::VectorTrace{T,U},
                constraints) where {T,U}
    change = MarkovChange(true, true, true)
    extend(gen, args, change, trace, constraints)
end

function extend(gen::Markov{T,U}, args, change::MarkovChange, trace::VectorTrace{T,U},
                constraints) where {T,U}
    (len, init, params) = unpack_args(args)
    check_length(len)
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
        kernel_args = (key, state, params...)
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

function update(gen::Markov{T,U}, args, change::Nothing, trace::VectorTrace{T,U},
                constraints) where {T,U}
    change = MarkovChange(true, true, true)
    update(gen, args, change, trace, constraints)
end

function update(gen::Markov{T,U}, args, change::MarkovChange,
                trace::VectorTrace{T,U}, constraints) where {T,U}
    (len, init, params) = unpack_args(args)
    check_length(len)
    prev_call = get_call_record(trace)
    prev_args = prev_call.args
    prev_len = prev_args[1]

    subtraces::PersistentVector{U} = trace.subtraces
    states::PersistentVector{T} = trace.call.retval

    # discard deleted applications
    discard = DynamicChoiceTrie()
    if prev_len > len
        for key=len+1:prev_len
            set_internal_node!(discard, key, get_choices(get_subtrace(trace, key)))
        end
        n_delete = prev_len - len
        for i=1:n_delete
            subtraces = pop(subtraces)
            states = pop(states)
        end
    end
    @assert length(subtraces) == min(prev_len, len)
    @assert length(states) == min(prev_len, len)

    # which retained (not deleted or new) applications to visit
    if change.params_changed
        to_visit = Set{Int}(1:min(prev_len, len))
    else
        to_visit = Set{Int}()
    end
    if change.init_changed
        push!(to_visit, 1)
    end
    for (key::Int, _) in get_internal_nodes(constraints)
        if key <= min(prev_len, len)
            push!(to_visit, key)
        end
    end

    # handle retained applications
    weight = 0.
    score = prev_call.score
    is_empty = !has_choices(trace)
    for key in sort(collect(to_visit))
        state = key > 1 ? states[key-1] : init
        kernel_args = (key, state, params...)
        if has_internal_node(constraints, key)
            node = get_internal_node(constraints, key)
        else
            node = EmptyChoiceTrie()
        end
        prev_subtrace::U = subtraces[key]
        prev_score = get_call_record(prev_subtrace).score
        args_change = nothing # NOTE we could propagate detailed change information
        (subtrace, w, kern_discard, retchange) = update(
            gen.kernel, kernel_args, args_change, prev_subtrace, node)
        set_internal_node!(discard, key, kern_discard)
        call = get_call_record(subtrace)
        score += call.score - prev_score
        subtraces = assoc(subtraces, key, subtrace)
        states = assoc(states, key, call.retval)
        weight += w
        is_empty = is_empty && !has_choices(subtrace)
    end

    # handle new applications
    for key=prev_len+1:len
        state = states[key-1]
        kernel_args = (key, state, params...)
        if has_internal_node(constraints, key)
            node = get_internal_node(constraints, key)
        else
            node = EmptyChoiceTrie()
        end
        subtrace::U = assess(gen.kernel, kernel_args, node)
        call = get_call_record(subtrace)
        score += call.score
        weight += call.score
        states = push(states, call.retval)
        subtraces = push(subtraces, subtrace)
        @assert length(states) == key
        @assert length(subtraces) == key
        is_empty = is_empty && !has_choices(subtrace)
    end

    call = CallRecord(score, states, args)
    new_trace = VectorTrace(subtraces, call, is_empty)
    retchange = nothing # NOTE we could provide some information
    (new_trace, weight, discard, retchange)
end

##################
# backprop_trace #
##################




export markov
export MarkovChange
