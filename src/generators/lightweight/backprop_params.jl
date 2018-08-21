using ReverseDiff: SpecialInstruction, istracked, increment_deriv!, deriv, TrackedArray
using ReverseDiff: InstructionTape, TrackedReal, seed!, unseed!, reverse_pass!, record!
import ReverseDiff: special_reverse_exec!, track

# if a value can't be tracked, return the untracked value instead, silently
track(value) = value

mutable struct GFBackpropParamsState
    trace::GFTrace
    read_trace::Nullable{Any}
    score::TrackedReal
    tape::InstructionTape
    visitor::AddressVisitor
    tracked_params::Dict{Symbol,Any}
end

function GFBackpropParamsState(trace::GFTrace, read_trace, tape, gf::GenFunction)
    tracked_params = Dict{Symbol,Any}()
    for (name, value) in gf.params
        tracked_params[name] = track(value, tape)
    end
    score = track(0., tape)
    GFBackpropParamsState(trace, read_trace, score, tape, AddressVisitor(), tracked_params)
end

get_args_change(state::GFBackpropParamsState) = nothing
get_addr_change(state::GFBackpropParamsState, addr) = nothing
set_ret_change!(state::GFBackpropParamsState, value) = begin end



function read_param(state::GFBackpropParamsState, name::Symbol)
    value = state.tracked_params[name]
    value
end

function addr(state::GFBackpropParamsState, dist::Distribution{T}, args, addr) where {T}
    visit!(state.visitor, addr)
    call::CallRecord = get_primitive_call(state.trace, addr)
    retval::T = call.retval
    state.score += logpdf(dist, retval, args...)
    retval
end

struct BackpropParamsRecord
    generator::Generator
    subtrace::Any
    read_trace::Nullable{Any}
end

function addr(state::GFBackpropParamsState, gen::Generator{T}, args, addr) where {T}
    visit!(state.visitor, addr)
    subtrace = get_subtrace(state.trace, addr)
    call::CallRecord = get_call_record(subtrace) # use the return value recorded in the trace
    retval::T = call.retval
    retval_maybe_tracked = track(retval, state.tape)
    if accepts_output_grad(gen) && !istracked(retval_maybe_tracked)
        error("Could not track return value at address $addr on AD tape.")
    end
    # some of the args may be tracked (see special_reverse_exec!)
    # note: we still need to run backprop_params on gen, even if it does not
    # accept an output gradient, because it may make random choices.
    record!(state.tape, SpecialInstruction,
        BackpropParamsRecord(gen, subtrace, state.read_trace), (args...), retval_maybe_tracked)
    retval_maybe_tracked 
end

function splice(state::GFBackpropParamsState, gf::GenFunction, args::Tuple)
    exec(gf, state, args)
end

function maybe_track(arg, has_argument_grad::Bool, tape)
    has_argument_grad ? track(arg, tape) : arg
end

function backprop_params(gf::GenFunction, trace::GFTrace, retval_grad, read_trace=nothing)
    tape = InstructionTape()
    state = GFBackpropParamsState(trace, read_trace, tape, gf)
    call = get_call_record(trace)
    args = call.args
    args_maybe_tracked = (map(maybe_track, args, gf.has_argument_grads, fill(tape, length(args)))...)
    retval_maybe_tracked = exec(gf, state, args_maybe_tracked)
    if gf.accepts_output_grad
        if !istracked(retval_maybe_tracked)
            error("Output of $gf accepts gradient but return value was not tracked on AD tape.")
        end
        if retval_grad != nothing
            # it is not an error if retval_grad is nothing, it means the gradient is zero
            # this is a convenience that 
            deriv!(retval_maybe_tracked, retval_grad)
        end
    end
    seed!(state.score)
    reverse_pass!(tape)

    # increment the gradient accumulators for static parameters
    for (name, tracked) in state.tracked_params
        gf.params_grad[name] += deriv(tracked)
    end

    # return gradients with respect to inputs
    # NOTE: if a value isn't tracked the gradient is nothing (Void)
    input_grads::Tuple = (map((arg, has_grad) -> has_grad ? deriv(arg) : nothing,
                             args_maybe_tracked, gf.has_argument_grads)...)
    input_grads
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{BackpropParamsRecord})
    record = instruction.func
    gen = record.generator
    args_maybe_tracked = instruction.input
    retval_maybe_tracked = instruction.output
    if accepts_output_grad(gen)
        @assert istracked(retval_maybe_tracked)
        retval_grad = deriv(retval_maybe_tracked)
    else
        retval_grad = nothing
    end
    read_trace = isnull(record.read_trace) ? nothing : get(record.read_trace)
    arg_grads = backprop_params(gen, record.subtrace, retval_grad, read_trace)
    for (arg, grad, has_grad) in zip(args_maybe_tracked, arg_grads, has_argument_grads(gen))
        if has_grad && istracked(arg)
            increment_deriv!(arg, grad)
        end
    end
    nothing
end
