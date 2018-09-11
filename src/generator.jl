###################
# Trace interface #
###################

struct CallRecord{T}
    score::Float64
    retval::T
    args::Tuple
end

"""
    get_call_record(trace)::CallRecord
"""
function get_call_record end

"""
    has_choices(trace)::Bool

If the choice trie returned by get_choices is empty or not.
"""
function has_choices end

"""
get_choices(trace)

Return a value implementing the choice trie interface
"""
function get_choices end

export CallRecord
export get_call_record
export get_choices
export has_choices


#############
# Generator #
#############

"""
Generator with return value type T and trace type U
"""
abstract type Generator{T,U} end

get_return_type(::Generator{T,U}) where {T,U} = T
get_trace_type(::Generator{T,U}) where {T,U} = U
get_change_type(::Generator) = :Any

"""
Return a boolean indicating whether a gradient of the output is accepted.
"""
function accepts_output_grad end

"""
Returns a tuple of booleans indicating whether a gradient is available, for each argument
"""
function has_argument_grads end

"""
Return a vector of Types indicating the statically known concrete argument types
"""
function get_concrete_argument_types end

"""
    (new_trace, weight, discard, retchange) = fix_update(gen::Generator, new_args, args_change, trace, constraints)
"""
function fix_update end

"""
Optional Generator method; special case of `fix_update` whree addresses cannot be deleted.

Exists for performance optimization.

    (new_trace::U, weight, retchange) = extend(g::Generator{T,U}, args, args_change, trace::U, constraints)
"""
function extend(gen::Generator{T,U}, new_args, args_change, trace::U, constraints) where {T,U}
    (new_trace, weight, discard, retchange) = fix_update(gen, new_args, args_change, trace, constraints)
    if !isempty(discard)
        error("Some addresses were deleted within extend on generator $gen")
    end
    (new_trace, weight, retchange)
end

# TODO add retchange as return value from predict
"""
    (new_trace::U, retchange) = predict(g::Generator{T,U}, args, args_change, trace::U)
"""
function predict(g::Generator, args, args_change, trace)
    (new_trace, weight, _) = extend(g, args, args_change, trace, EmptyChoiceTrie())
    @assert weight == 0.
    new_trace
end

"""
    (trace::U, weight) = generate(g::Generator{T,U}, args, constraints)
"""
function generate end

# TODO will be removed
"""
    trace = simulate(g::Generator, args)
"""
function simulate(gen::Generator, args)
    (trace, weight) = generate(gen, args, EmptyChoiceTrie())
    if weight != 0.
        error("Got nonzero weight during simulate.")
    end
    trace
end

"""
    (trace, discard) = project(g::Generator, args, constraints)
"""
function project end

"""
choices must contain all random choices, (and no extra random choices, or else
it is an error

    trace = assess(g::Generator, args, choices)
"""
function assess end

"""
change of args.
may not simulate.
addresses may be added or deleted.
constraints may collide with existing random choices.

    (new_trace, weight, discard, retchange) = update(g::Generator, new_args, args_change, trace, constraints)
"""
function update end

"""
change of args.
may simulate.
addresses may be added or deleted.
constraints may collide with existing random choices, but no new choices may be constrained.

    (new_trace, weight, retchange) = fix_update(g::Generator, new_args, args_change, trace, constraints)
"""
function fix_update end

"""
    (new_trace, weight, retcahnge) = regenerate(
        gen::Generator, new_args, args_change, trace, selection::AddressSet)
"""
function regenerate end

"""
    weight = ungenerate(g::Generator, trace, constraints)
"""
function ungenerate end

"""
    input_grads::Tuple = backprop_params(gen:Generator, trace, retval_grad)
"""
function backprop_params end

"""
    (input_grads::Tuple, values, gradients) = backprop_trace(gen:Generator, trace, selection::AddressSet, retval_grad)
"""
function backprop_trace end


export Generator
export simulate
export extend
export predict
export generate
export project
export assess
export fix_update
export update
export regenerate
export ungenerate
export backprop_params
export backprop_trace


###########################
# incremental computation #
###########################

struct NoChange end
export NoChange
