import ReverseDiff
import ReverseDiff: TrackedReal, InstructionTape, seed!, unseed!, reverse_pass!, deriv
using LinearAlgebra: logabsdet

##########################
# injective function DSL #
##########################

struct InjectiveFunction
    julia_function::Function
end

macro inj(ast)
    if ast.head != :function
        error("syntax error at $(ast) in $(ast.head)")
    end
    if length(ast.args) != 2
        error("syntax error at $(ast) in $(ast.args)")
    end
    signature = ast.args[1]
    body = ast.args[2]
    if signature.head != :call
        error("syntax error at $(ast) in $(signature)")
    end
    function_name = signature.args[1]
    args = signature.args[2:end]
    escaped_args = map(esc, args)
    fn_args = [esc(state), escaped_args...]
    Expr(:block,
        Expr(:(=), 
            esc(function_name),
            Expr(:call, :InjectiveFunction,
                Expr(:function, Expr(:tuple, fn_args...), esc(body)))))
end

macro write(value, addr)
    Expr(:call, :write_to_addr, esc(state), esc(value), esc(addr))
end

macro copy(input_addr, output_addr)
    Expr(:call, :copy_addr, esc(state), esc(input_addr), esc(output_addr))
end

macro swap(value, input_addr, output_addr)
    Expr(:call, :swap, esc(state), esc(input_addr), esc(output_addr))
end

macro copyall(input_addr, output_addr)
    Expr(:call, :copyall, esc(state), esc(input_addr), esc(output_addr))
end

macro swapall(input_addr, output_addr)
    Expr(:call, :swapall, esc(state), esc(input_addr), esc(output_addr))
end

mutable struct InjectiveApplyState
    input::Assignment
    output::DynamicAssignment
    tape::InstructionTape
    tracked_reads::Trie{Any,TrackedReal}
    tracked_writes::Trie{Any,TrackedReal}
    copied::AddressSet
    visitor::AddressVisitor
end

function InjectiveApplyState(input)
    output = DynamicAssignment()
    tape = InstructionTape()
    tracked_reads = Trie{Any,TrackedReal}()
    tracked_writes = Trie{Any,TrackedReal}()
    copied = DynamicAddressSet()
    visitor = AddressVisitor()
    InjectiveApplyState(input, output, tape, tracked_reads, tracked_writes, copied, visitor)
end

function exec(fn::InjectiveFunction, state::InjectiveApplyState, args::Tuple)
    fn.julia_function(state, args...)
end

function maybe_track_value!(state::InjectiveApplyState, addr, value)
    value
end

function maybe_track_value!(state::InjectiveApplyState, addr, value::Float64)
    tracked_value = ReverseDiff.track(value, state.tape)
    set_leaf_node!(state.tracked_reads, addr, tracked_value)
    tracked_value
end

function read(state::InjectiveApplyState, addr)
    if has_leaf_node(state.tracked_reads, addr)
        # use the existing tracked read value
        get_leaf_node(state.tracked_reads, addr)::TrackedReal
    else
        value = get_value(state.input, addr)
        maybe_track_value!(state, addr, value)
    end
end

function write_to_addr(state::InjectiveApplyState, value, addr)
    visit!(state.visitor, addr)
    set_value!(state.output, addr, value)
    nothing
end

function write_to_addr(state::InjectiveApplyState, tracked_value::TrackedReal, addr)
    visit!(state.visitor, addr)
    # TODO what if user tries to write same tracked value to two diff addrs?
    set_leaf_node!(state.tracked_writes, addr, tracked_value)
    set_value!(state.output, addr, ReverseDiff.value(tracked_value))
    nothing
end

function copy_addr(state::InjectiveApplyState, input_addr, output_addr)
    visit!(state.visitor, output_addr)
    if !has_value(state.input, input_addr)
        error("Value at $input_addr not found in input ($(state.input))")
    end
    value = get_value(state.input, input_addr)
    set_leaf_node!(state.output, output_addr, value)
    push_leaf_node!(state.copied, input_addr)
    nothing
end

function swap(state::InjectiveApplyState, input_addr, output_addr)
    copy(state, input_addr, output_addr)
    copy(state, output_addr, input_addr)
end

function copyall(state::InjectiveApplyState, input_addr, output_addr)
    visit!(state.visitor, output_addr)
    subassmt = get_subassmt(state.input, input_addr)
    set_subassmt!(state.output, output_addr, subassmt)
    push_leaf_node!(state.copied, input_addr)
    nothing
end

function swapall(state::InjectiveApplyState, input_addr, output_addr)
    copyall(state, input_addr, output_addr)
    copyall(state, output_addr, input_addr)
end

function addr(state::InjectiveApplyState, fn::InjectiveFunction, args, output_addr)
    visit!(state.visitor, output_addr)
    subassmt = DynamicAssignment()
    sub_tracked_writes = Trie{Any,TrackedReal}()
    visitor = AddressVisitor()
    sub_state = InjectiveApplyState(state.input, subassmt,
        state.tape, state.tracked_reads, sub_tracked_writes, state.copied, visitor)
    value = exec(fn, sub_state, args)
    set_subassmt!(state.output, output_addr, subassmt)
    set_internal_node!(state.tracked_writes, output_addr, sub_tracked_writes)
    value
end

function splice(state::InjectiveApplyState, fn::InjectiveFunction, args::Tuple)
    exec(fn, state, args)
end

function apply(fn::InjectiveFunction, args::Tuple, input)
    state = InjectiveApplyState(input)
    val = exec(fn, state, args)

    # addresses that were copied are excluded from inputs
    # they can be ignored when computing Jacobian determinant
    # this is a performance optimization
    delete!(state.tracked_reads, state.copied)
    tracked_inputs = collect(values(state.tracked_reads))
    tracked_outputs = collect(values(state.tracked_writes))
    n = length(tracked_inputs)
    if n != length(tracked_outputs)
        error("input and output dimensions differ")
    end
    jacobian = zeros(n, n)
    for (i, tracked_output) in enumerate(tracked_outputs)
        seed!(tracked_output)
        reverse_pass!(state.tape)
        for (j, tracked_input) in enumerate(tracked_inputs)
            jacobian[i, j] = deriv(tracked_input)
            unseed!(tracked_input)
        end
        unseed!(tracked_output)
    end
    (log_abs_det, det_sign) = logabsdet(jacobian)
    (state.output, log_abs_det, val)
end

export @inj
export @write, @copy, @swap, @copyall, @swapall, apply
