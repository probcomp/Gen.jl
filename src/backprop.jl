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

using ReverseDiff: InstructionTape, TrackedReal, TrackedArray
