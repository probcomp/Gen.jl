import ForwardDiff
import MacroTools
using Parameters: @with_kw
import LinearAlgebra
using Gen

# TODO switch the order of the args so it matches the old involution order?

# TODO do we allow involution functions to call other involution functions (and
# recursion?)

# TODO what about moving entire subtrees? -- that should be fine, all the
# addresses underneath will not be involved in Jacobina.

@with_kw mutable struct FwdInvState
    trace # should be read-only
    u # should be read-only
    constraints = choicemap()
    u_back = choicemap()
    arr = Vector{Float64}()
    next_input_index = 1
    next_output_index = 1
    u_key_to_index = Dict()
    t_key_to_index = Dict()
    t_cont_reads = Dict()
    u_cont_reads = Dict()
    t_move_reads = DynamicSelection()#Set() # TODO make into a Selection
    u_move_reads = DynamicSelection()#Set() # TODO make into a Selection
    cont_constraints_key_to_index = Dict()
    cont_u_back_key_to_index = Dict()
    marked_as_retained = Set()
end

struct JacInvState{T}
    trace
    u
    input_arr::AbstractArray{T}
    output_arr::Array{T}
    t_key_to_index::Dict
    u_key_to_index::Dict
    cont_constraints_key_to_index::Dict
    cont_u_back_key_to_index::Dict
end

function JacInvState(state::FwdInvState, input_arr::AbstractArray{T}) where {T <: Real}
    JacInvState{T}(
        state.trace, state.u,
        input_arr, Array{T,1}(undef, state.next_output_index-1),
        state.t_key_to_index, state.u_key_to_index,
        state.cont_constraints_key_to_index, state.cont_u_back_key_to_index)
end

const inv_state = gensym("inv_state")

# read discrete from model

macro read_discrete_from_model(addr)
    quote read_discrete_from_model($(esc(inv_state)), $(esc(addr))) end
end

function read_discrete_from_model(state::Union{FwdInvState,JacInvState}, addr)
    state.trace[addr]
end

# read discrete from proposal

macro read_discrete_from_proposal(addr)
    quote read_discrete_from_proposal($(esc(inv_state)), $(esc(addr))) end
end

function read_discrete_from_proposal(state::Union{FwdInvState,JacInvState}, addr)
    state.u[addr]
end

# write_discrete_to_proposal

macro write_discrete_to_proposal(addr, value)
    quote write_discrete_to_proposal($(esc(inv_state)), $(esc(addr)), $(esc(value))) end
end

function write_discrete_to_proposal(state::FwdInvState, addr, value)
    state.u_back[addr] = value
    value
end

function write_discrete_to_proposal(state::JacInvState, addr, value)
    value
end

# write_discrete_to_model

macro write_discrete_to_model(addr, value)
    quote write_discrete_to_model($(esc(inv_state)), $(esc(addr)), $(esc(value))) end
end

function write_discrete_to_model(state::FwdInvState, addr, value)
    state.constraints[addr] = value
    value
end

function write_discrete_to_model(state::JacInvState, addr, value)
    value
end

# read continuous from proposal

macro read_continuous_from_proposal(addr)
    quote read_continuous_from_proposal($(esc(inv_state)), $(esc(addr))) end
end

function read_continuous_from_proposal(state::FwdInvState, addr)
    state.u_cont_reads[addr] = state.u[addr]
    state.u[addr]
end

function read_continuous_from_proposal(state::JacInvState, addr)
    state.input_arr[state.u_key_to_index[addr]]
end

# read continuous from model

macro read_continuous_from_model(addr)
    quote read_continuous_from_model($(esc(inv_state)), $(esc(addr))) end
end

function read_continuous_from_model(state::FwdInvState, addr)
    state.t_cont_reads[addr] = state.trace[addr]
    state.trace[addr]
end

function read_continuous_from_model(state::JacInvState, addr)
    state.input_arr[state.t_key_to_index[addr]]
end

# read continuous from model retained (optional, for efficiency)

macro read_continuous_from_model_retained(addr)
    quote read_continuous_from_model_retained($(esc(inv_state)), $(esc(addr))) end
end

function read_continuous_from_model_retained(state::FwdInvState, addr)
    push!(state.marked_as_retained, addr)
    state.trace[addr]
end

function read_continuous_from_model_retained(state::JacInvState, addr)
    state.trace[addr] # read directly from the trace, instead of the array
end

# write_continuous_to_proposal

macro write_continuous_to_proposal(addr, value)
    quote write_continuous_to_proposal($(esc(inv_state)), $(esc(addr)), $(esc(value))) end
end

function write_continuous_to_proposal(state::FwdInvState, addr, value)
    state.u_back[addr] = value
    state.cont_u_back_key_to_index[addr] = state.next_output_index
    state.next_output_index += 1
    value
end

function write_continuous_to_proposal(state::JacInvState, addr, value)
    state.output_arr[state.cont_u_back_key_to_index[addr]] = value
end

# write_continuous_to_model

macro write_continuous_to_model(addr, value)
    quote write_continuous_to_model($(esc(inv_state)), $(esc(addr)), $(esc(value))) end
end

function write_continuous_to_model(state::FwdInvState, addr, value)
    state.constraints[addr] = value
    state.cont_constraints_key_to_index[addr] = state.next_output_index
    state.next_output_index += 1
    value
end

function write_continuous_to_model(state::JacInvState, addr, value)
    state.output_arr[state.cont_constraints_key_to_index[addr]] = value
end

# move_model_to_model

macro move_model_to_model(from_addr, to_addr)
    quote move_model_to_model($(esc(inv_state)), $(esc(from_addr)), $(esc(to_addr))) end
end

function move_model_to_model(state::FwdInvState, from_addr, to_addr)
    trace_choices = get_choices(state.trace)
    push!(state.t_move_reads, from_addr)
    if has_value(trace_choices, from_addr)
        state.constraints[to_addr] = state.trace[from_addr]
    else
        set_submap!(state.constraints, to_addr, get_submap(trace_choices, from_addr))
    end
    nothing
end

function move_model_to_model(state::JacInvState, from_addr, to_addr)
    nothing
end

# move_model_to_proposal

macro move_model_to_proposal(from_addr, to_addr)
    quote move_model_to_proposal($(esc(inv_state)), $(esc(from_addr)), $(esc(to_addr))) end
end

function move_model_to_proposal(state::FwdInvState, from_addr, to_addr)
    trace_choices = get_choices(state.trace)
    push!(state.t_move_reads, from_addr)
    if has_value(trace_choices, from_addr)
        state.u_back[to_addr] = state.trace[from_addr]
    else
        set_submap!(state.u_back, to_addr, get_submap(trace_choices, from_addr))
    end
    nothing
end

function move_model_to_proposal(state::JacInvState, from_addr, to_addr)
    nothing
end

# move_proposal_to_proposal

macro move_proposal_to_proposal(from_addr, to_addr)
    quote move_proposal_to_proposal($(esc(inv_state)), $(esc(from_addr)), $(esc(to_addr))) end
end

function move_proposal_to_proposal(state::FwdInvState, from_addr, to_addr)
    push!(state.u_move_reads, from_addr)
    if has_value(state.u, from_addr)
        state.u_back[to_addr] = state.u[from_addr]
    else
        set_submap!(state.u_back, to_addr, get_submap(state.u, from_addr))
    end
    nothing
end

function move_proposal_to_proposal(state::JacInvState, from_addr, to_addr)
    nothing
end

# move_proposal_to_model

macro move_proposal_to_model(from_addr, to_addr)
    quote move_proposal_to_model($(esc(inv_state)), $(esc(from_addr)), $(esc(to_addr))) end
end

function move_proposal_to_model(state::FwdInvState, from_addr, to_addr)
    push!(state.u_move_reads, from_addr)
    if has_value(state.u, from_addr)
        state.constraints[to_addr] = state.u[from_addr]
    else
        set_submap!(state.constraints, to_addr, get_submap(state.u, from_addr))
    end
    nothing
end

function move_proposal_to_model(state::JacInvState, from_addr, to_addr)
    nothing
end

# call 

macro callinv(ex)
    MacroTools.@capture(ex, f_(args__)) || error("expected syntax: f(..)")
    quote $(esc(f))($(esc(inv_state)), $(map(esc, args)...)) end
end

# apply 

function apply_involution(inv::Function, trace, u, proposal_args, proposal_retval)

    # runs the function, mutates the state
    state = FwdInvState(trace=trace, u=u)
    inv(state, get_args(trace), proposal_args, proposal_retval)

    # add continuous addresses read from model and proposal to arr
    # exclude addresses that were moved to another address
    #state = $(esc(inv_state))
    for (addr, v) in state.u_cont_reads
        if !(addr in state.u_move_reads) 
            state.u_key_to_index[addr] = state.next_input_index
            state.next_input_index += 1
            push!(state.arr, v)
        end
    end
    for (addr, v) in state.t_cont_reads
        if !(addr in state.t_move_reads)
            state.t_key_to_index[addr] = state.next_input_index
            state.next_input_index += 1
            push!(state.arr, v)
        end
    end

    function f_array(input_arr::AbstractArray{T}) where {T <: Real}
        # note: we are closing over the arguments
        state = JacInvState(state, input_arr)
        inv(state, get_args(trace), proposal_args, proposal_retval) # mutate the state
        state.output_arr
    end

    # TODO XXX proposal_retval cannot depend on continuous random choices in model
    # or proposal (we might be abel to relax this later using choice_gradients) XXX
    #f_array = f_cont!(state, get_args(trace), proposal_args, proposal_retval)#, f_disc_retval)

    state, f_array
end


macro involution(ex)
    MacroTools.@capture(ex, function f_(args__) body_ end) || error("expected syntax: function f(..) .. end")

    quote

    # mutates state

    function $(esc(f))($(esc(inv_state)), $(map(esc, args)...))

        $(esc(body))
    
        #function f_array(input_arr::AbstractArray{T}) where {T <: Real}
            ## note: we are closing over the arguments
            #$(esc(inv_state)) = JacInvState($(esc(inv_state)), input_arr)
            #$(esc(body))
            #$(esc(inv_state)).output_arr
        #end
    
        #return f_array
    end

    end # quote

end # macro involution()

function log_abs_det_jacobian(f_array, state, discard)

    # Jacobian of involution
    # columns are inputs, rows are outputs
    J = ForwardDiff.jacobian(f_array, state.arr)
    @assert size(J)[2] == length(state.arr)
    num_outputs = size(J)[1]
    
    # remove columns for inputs from the trace that were retained
    keep = fill(true, length(state.arr))
    for (addr, index) in state.t_key_to_index
        if !has_value(discard, addr)
            keep[index] = false
        end
    end
    @assert sum(keep) == num_outputs # must be square
    J = J[:,keep]

    LinearAlgebra.logabsdet(J)[1]
end

# TODO proposal_retval is a way that continuous data can leak unaccounted for..
# for now, the requirement is that the return value of f_disc cannot depend on
# continuous addresses in the model or proposal although in the future, we
# could do AD through its return value as well using choice_gradients() and add
# these to the Jacobian...?

function involution(f_cont!, trace, proposal_args, u, proposal_retval, check::Bool)
    #f_disc, f_cont!, trace, proposal_args, u, proposal_retval, check::Bool)

    # discrete component of involution
    #(disc_constraints, disc_u_back, f_disc_retval) = f_disc(trace, u, proposal_args, proposal_retval)

    # continuous component of involution
    state, f_array = apply_involution(f_cont!, trace, u, proposal_args, proposal_retval)
    #state = FwdInvState(trace=trace, u=u)
    # TODO XXX proposal_retval cannot depend on continuous random choices in model
    # or proposal (we might be abel to relax this later using choice_gradients) XXX
    #f_array = f_cont!(state, get_args(trace), proposal_args, proposal_retval)#, f_disc_retval)
    #constraints = merge(disc_constraints, state.continuous constraints)
    #u_back = merge(disc_u_back, state.u_back)
    constraints = state.constraints
    u_back = state.u_back

    # update model trace
    (new_trace, model_weight, _, discard) = update(
        trace, get_args(trace), map((_) -> NoChange(), get_args(trace)), constraints)

    # check the user's retained assertions
    if check
        for addr in state.marked_as_retained
            has_value(discard, addr) && error("addr $addr was marked as retained, but was not")
        end
    end

    correction = log_abs_det_jacobian(f_array, state, discard)

    (new_trace, u_back, model_weight + correction)
end

#function rjmcmc(trace, q, proposal_args, f_disc, f_cont!;
function rjmcmc(trace, q, proposal_args, f_cont!;
        check=false, observations=EmptyChoiceMap())

    # run proposal
    u, q_fwd_score, proposal_retval = propose(q, (trace, proposal_args...))

    new_trace, u_back, model_score = involution(
        f_cont!, trace, proposal_args, u, proposal_retval, check)
        #f_disc, f_cont!, trace, proposal_args, u, proposal_retval, check)
    check && Gen.check_observations(get_choices(new_trace), observations)

    # compute proposal backward score
    (q_bwd_score, proposal_retval_back) = assess(q, (new_trace, proposal_args...), u_back)

    # round trip check
    if check
        trace_rt, u_rt, model_score_rt = involution(
            f_cont!, new_trace, proposal_args, u_back, proposal_retval_back, check)
            #f_disc, f_cont!, new_trace, proposal_args, u_back, proposal_retval_back, check)
        if !isapprox(u_rt, u)
            println("u:")
            println(u)
            println("u_rt:")
            println(u_rt)
            error("Involution round trip check failed")
        end
        if !isapprox(get_choices(trace), get_choices(trace_rt))
            println("get_choices(trace):")
            println(get_choices(trace))
            println("get_choices(trace_rt):")
            println(get_choices(trace_rt))
            error("Involution round trip check failed")
        end
        if !isapprox(model_score, -model_score_rt)
            println("model_score: $model_score, -model_score_rt: $(-model_score_rt)")
            error("Involution round trip check failed")
        end
        println("checks done..")

    end

    # accept or reject
    alpha = model_score - q_fwd_score + q_bwd_score
    if log(rand()) < alpha
        # accept
        return (new_trace, true)
    else
        # reject
        return (trace, false)
    end
end
