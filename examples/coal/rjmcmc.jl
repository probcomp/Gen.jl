import ForwardDiff
import MacroTools
using Parameters: @with_kw
import LinearAlgebra
using Gen

# TODO do we allow involution functions to call other involution functions (and recursion?)

# TODO what about moving entire subtrees? -- that should be fine, all the addresses underneath will not be involved in Jacobina.

@with_kw mutable struct FwdInvState
    trace
    u
    constraints = choicemap()
    u_back = choicemap()
    arr = Vector{Float64}()
    next_input_index = 1
    next_output_index = 1
    u_key_to_index = Dict()
    t_key_to_index = Dict()
    t_cont_reads = Dict()
    t_move_reads = Set()
    cont_constraints_key_to_index = Dict()
    cont_u_back_key_to_index = Dict()
    marked_as_retained = Set()
end

struct JacInvState{T}
    input_arr::AbstractArray{T}
    output_arr::Array{T}
    t_key_to_index::Dict
    u_key_to_index::Dict
    cont_constraints_key_to_index::Dict
    cont_u_back_key_to_index::Dict
end

function JacInvState(state::FwdInvState, input_arr::AbstractArray{T}) where {T <: Real}
    JacInvState{T}(
        input_arr, Array{T,1}(undef, state.next_output_index-1),
        state.t_key_to_index, state.u_key_to_index,
        state.cont_constraints_key_to_index, state.cont_u_back_key_to_index)
end

const inv_state = gensym("inv_state")

# read from proposal

macro read_from_proposal(addr)
    quote read_from_proposal($(esc(inv_state)), $(esc(addr))) end
end

function read_from_proposal(state::FwdInvState, addr)
    if !haskey(state.u_key_to_index, addr)
        state.u_key_to_index[addr] = state.next_input_index
        state.next_input_index += 1
        push!(state.arr, state.u[addr])
    end
    state.u[addr]
end

function read_from_proposal(state::JacInvState, addr)
    state.input_arr[state.u_key_to_index[addr]]
end

# read from model

macro read_from_model(addr)
    quote read_from_model($(esc(inv_state)), $(esc(addr))) end
end

function read_from_model(state::FwdInvState, addr)
    state.t_cont_reads[addr] = state.trace[addr]
    state.trace[addr]
end

function read_from_model(state::JacInvState, addr)
    state.input_arr[state.t_key_to_index[addr]]
end

# read from model retained (optional, for efficiency)

macro read_from_model_retained(addr)
    quote read_from_model_retained($(esc(inv_state)), $(esc(addr))) end
end

function read_from_model_retained(state::FwdInvState, addr)
    push!(state.marked_as_retained, addr)
    state.trace[addr]
end

function read_from_model_retained(state::JacInvState, addr)
    state.trace[addr] # read directly from the trace, instead of the array
end

# write_to_proposal

macro write_to_proposal(addr, value)
    quote write_to_proposal($(esc(inv_state)), $(esc(addr)), $(esc(value))) end
end

function write_to_proposal(state::FwdInvState, addr, value)
    state.u_back[addr] = value
    state.cont_u_back_key_to_index[addr] = state.next_output_index
    state.next_output_index += 1
    value
end

function write_to_proposal(state::JacInvState, addr, value)
    state.output_arr[state.cont_u_back_key_to_index[addr]] = value
end

# write_to_model

macro write_to_model(addr, value)
    quote write_to_model($(esc(inv_state)), $(esc(addr)), $(esc(value))) end
end

function write_to_model(state::FwdInvState, addr, value)
    state.constraints[addr] = value
    state.cont_constraints_key_to_index[addr] = state.next_output_index
    state.next_output_index += 1
    value
end

function write_to_model(state::JacInvState, addr, value)
    state.output_arr[state.cont_constraints_key_to_index[addr]] = value
end

# move_model

macro move_model(from_addr, to_addr)
    quote move_model($(esc(inv_state)), $(esc(from_addr)), $(esc(to_addr))) end
end

function move_model(state::FwdInvState, from_addr, to_addr)
    state.constraints[to_addr] = state.trace[from_addr]
    push!(state.t_move_reads, from_addr)
    state.trace[from_addr]
end

function move_model(state::JacInvState, from_addr, to_addr)
    nothing
end

macro involution(ex)
    MacroTools.@capture(ex, function f_(args__) body_ end) || error("expected syntax: function f(..) .. end")

    quote

    # mutates state

    function $(esc(f))($(esc(inv_state))::FwdInvState, $(map(esc, args)...))

        $body
    
        # add addresses read from t to arr
        state = $(esc(inv_state))
        for (addr, v) in state.t_cont_reads
            if !(addr in state.t_move_reads) # exclude addresses that were moved to another address
                state.t_key_to_index[addr] = state.next_input_index
                state.next_input_index += 1
                push!(state.arr, v)
            end
        end
    
        function f_array(input_arr::AbstractArray{T}) where {T <: Real}
            $(esc(inv_state)) = JacInvState($(esc(inv_state)), input_arr)
            $body
            $(esc(inv_state)).output_arr
        end
    
        return f_array
    end

    end # quote

end # macro involution()

function rjmcmc(trace, q, proposal_args, f_disc, f_cont!)

    # run proposal
    u, q_fwd_score, = propose(q, (trace, proposal_args...))

    # run discrete involution
    (disc_constraints, disc_u_back, f_disc_retval) = f_disc(trace, u, proposal_args)
    
    # run continuous bijection, and get array function for computing Jacobian
    cont_state = FwdInvState(trace=trace, u=u)
    f_array = f_cont!(cont_state, get_args(trace), proposal_args, f_disc_retval)

    # update model trace
    constraints = merge(disc_constraints, cont_state.constraints)
    (new_trace, model_weight, _, discard) = update(
        trace, get_args(trace), map((_) -> NoChange(), get_args(trace)), constraints)

    # check the user's retained assertions (TODO disable this check in fast mode)
    for addr in cont_state.marked_as_retained
        has_value(discard, addr) && error("addr $addr was marked as retained, but was not")
    end

    # Jacobian of involution
    # columns are inputs, rows are outputs
    J = ForwardDiff.jacobian(f_array, cont_state.arr)
    @assert size(J)[2] == length(cont_state.arr)
    num_outputs = size(J)[1]
    
    # remove columns for inputs from the trace that were retained
    keep = fill(true, length(cont_state.arr))
    for (addr, index) in cont_state.t_key_to_index
        if !has_value(discard, addr)
            keep[index] = false
        end
    end
    @assert sum(keep) == num_outputs # must be square
    J = J[:,keep]

    # compute correction
    correction = LinearAlgebra.logabsdet(J)[1]

    # compute proposal backward score
    u_back = merge(disc_u_back, cont_state.u_back)
    (q_bwd_score, _) = assess(q, (new_trace, proposal_args...), u_back)

    # accept or reject
    alpha = model_weight - q_fwd_score + q_bwd_score - correction # TODO check sign on correction
    if log(rand()) < alpha
        # accept
        return (new_trace, true)
    else
        # reject
        return (trace, false)
    end
end
