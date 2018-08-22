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

# TODO make the reads from the read_trace part of the trace. 

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
args should be such that it does not delete, or change the distribution of, any
existing random choice.  constraints impleemnts read-only trie interface, must not collide with any existing random
choices. may simulate new random choices.

    (new_trace::U, weight, retchange) = extend(g::Generator{T,U}, args, args_change, trace::U, constraints, read_trace)
"""
function extend end

# TODO add retchange as return value from predict
"""
    (new_trace::U, retchange) = predict(g::Generator{T,U}, args, args_change, trace::U, read_trace)
"""
function predict(g::Generator, args, args_change, trace, read_trace=nothing)
    (new_trace, weight, _) = extend(g, args, args_change, trace, EmptyChoiceTrie(), read_trace)
    @assert weight == 0.
    new_trace
end

"""
    (trace::U, weight) = generate(g::Generator{T,U}, args, constraints, read_trace)
"""
function generate end

"""
    trace = simulate(g::Generator, args, read_trace)
"""
function simulate end

"""
    (trace, discard) = project(g::Generator, args, constraints, read_trace)
"""
function project end

"""
choices must contain all random choices, (and no extra random choices, or else
it is an error

    trace = assess(g::Generator, args, choices, read_trace)
"""
function assess end

"""
change of args.
may not simulate.
addresses may be added or deleted.
constraints may collide with existing random choices.

    (new_trace, weight, discard, retchange) = update(g::Generator, new_args, args_change, trace, constraints, read_trace)
"""
function update end

"""
change of args.
may simulate.
addresses may be added or deleted.
constraints may collide with existing random choices, but no new choices may be constrained.

    (new_trace, weight, retchange) = resimulation_update(g::Generator, new_args, args_change, trace, constraints, read_trace)
"""
function resimulation_update end

"""
    (new_trace, weight, retcahnge) = regenerate(
        gen::Generator, new_args, args_change, trace, selection::AddressSet, read_trace)
"""
function regenerate end

"""
    weight = ungenerate(g::Generator, trace, constraints, read_trace)
"""
function ungenerate end

export simulate
export extend
export predict
export generate
export project
export assess
export update
export regenerate
export ungenerate


###########################
# incremental computation #
###########################

struct NoChange end
export NoChange
