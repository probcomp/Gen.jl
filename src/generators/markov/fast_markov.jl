#########################
# fast markov generator # 
#########################

include("fast_persistent_vector.jl")

struct FastMarkov{T,U} <: Generator{LinkedList{T},FastVectorTrace{T,U}}
    kernel::Generator{T,U}
end

function fast_markov(kernel::Generator{T,U}) where {T,U}
    kernel_arg_types = get_static_argument_types(kernel)
    if length(kernel_arg_types) != 3 || kernel_arg_types[1] != :Int
        error("markov requires a kernel with arguments (t::Int, state, params)")
    end
    FastMarkov{T,U}(kernel)
end

function get_static_argument_types(markov::FastMarkov)
    kernel_arg_types = get_static_argument_types(kernel)
    state_type = kernel_arg_types[2]
    params_type = kernel_arg_types[3]
    # 1 total number of time steps
    # 2 initial state (this must also be the return type of the kernel, not currently checked)
    # 3 parameters (shared across all time steps)
    [:Int, state_type, params_type]
end

function generate(gen::FastMarkov{T,U}, args, constraints) where {T,U}
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
    call = CallRecord{LinkedList{T}}(score, LinkedList{T}(states), args)
    trace = FastVectorTrace{T,U}(LinkedList{U}(subtraces), call, is_empty)
    (trace, weight)
end

# the only change possible is an extension --- cannot revisit existing nodes.
function extend(gen::FastMarkov{T,U}, args, change::MarkovChange, trace::FastVectorTrace{T,U},
                constraints) where {T,U}
    if change.params_changed || change.init_changed
        error("Changing the initial state or params not supported")
    end
    (len, init_state, params) = args
    prev_call = get_call_record(trace)
    prev_args = prev_call.args
    prev_len = prev_args[1]
    if len < prev_len
        error("Extend cannot remove addresses or namespaces")
    end
    subtraces::LinkedList{U} = trace.subtraces
    states::LinkedList{T} = trace.call.retval
    weight = 0.
    score = prev_call.score
    is_empty = !has_choices(trace)
    for key=prev_len+1:len
        state = key > 1 ? states.value : init_state
        kernel_args = (key, state, params)
        if has_internal_node(constraints, key)
            node = get_internal_node(constraints, key)
        else
            node = EmptyChoiceTrie()
        end
        (subtrace::U, w) = generate(gen.kernel, kernel_args, node)
        kernel_call::CallRecord{T} = get_call_record(subtrace)
        states = push(states, kernel_call.retval)
        subtraces = push(subtraces, subtrace)
        score += kernel_call.score
        weight += w
        is_empty = is_empty && !has_choices(subtrace)
    end
    call = CallRecord{LinkedList{T}}(score, states, args)
    trace = FastVectorTrace(subtraces, call, is_empty)
    (trace, weight, nothing)
end

export fast_markov
export FastMarkovChange
