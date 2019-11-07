import ReverseDiff

# Gen versions of certain ReverseDiff functions
# this is because the generated code for the static IR is currently loaded in
# the Main module. to avoid requiring the Main module environment to import the
# name ReverseDiff, these aliases allow us to only require that the Main module
# environmment have Gen imported.
new_tape() = ReverseDiff.InstructionTape()
track(value, tape) = ReverseDiff.track(value, tape)
value(maybe_tracked) = ReverseDiff.value(maybe_tracked)
deriv!(maybe_tracked, deriv) = ReverseDiff.deriv!(maybe_tracked, deriv)
reverse_pass!(tape) = ReverseDiff.reverse_pass!(tape)
deriv(tracked) = ReverseDiff.deriv(tracked)
istracked(maybe_tracked) = ReverseDiff.istracked(maybe_tracked)
record!(tape, instruction_type, args...) = ReverseDiff.record!(tape, instruction_type, args...)
increment_deriv!(arg, deriv) = ReverseDiff.increment_deriv!(arg, deriv)
seed!(tracked) = ReverseDiff.seed!(tracked)
unseed!(tracked) = ReverseDiff.unseed!(tracked)

using ReverseDiff: InstructionTape, TrackedReal, SpecialInstruction, TrackedArray

########
# fill #
########

function Base.fill(x::TrackedReal{V}, dims::Integer...) where {V}
    tp = ReverseDiff.tape(x)
    out = ReverseDiff.track(fill(ReverseDiff.value(x), dims...), V, tp)
    ReverseDiff.record!(tp, SpecialInstruction, fill, (x, dims), out)
    return out
end

@noinline function ReverseDiff.special_reverse_exec!(
        instruction::SpecialInstruction{typeof(fill)})
    x, dims = instruction.input
    output = instruction.output
    ReverseDiff.istracked(x) && ReverseDiff.increment_deriv!(x, sum(ReverseDiff.deriv(output)))
    ReverseDiff.unseed!(output) 
    return nothing
end 

@noinline function ReverseDiff.special_forward_exec!(
        instruction::SpecialInstruction{typeof(fill)})
    x, dims = instruction.input
    ReverseDiff.value!(instruction.output, fill(value(x), dims...))
    return nothing
end 
