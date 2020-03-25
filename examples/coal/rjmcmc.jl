import ForwardDiff
import MacroTools
using Parameters: @with_kw
import LinearAlgebra
using Gen

# TODO allow providing a pair of address namespaces, one for the model and one
# for the proposal, when calling another involution function with call

# TODO proposal_retval is a way that continuous data can leak unaccounted for..
# for now, the requirement is that the return value of f_disc cannot depend on
# continuous addresses in the model or proposal although in the future, we
# could do AD through its return value as well using choice_gradients() and add
# these to the Jacobian...?

struct Involution
    fn!::Function
end

struct FirstPassState
    trace
    u::ChoiceMap
    constraints::ChoiceMap
    u_back::ChoiceMap
    t_cont_reads::Dict
    u_cont_reads::Dict
    t_cont_writes::Dict
    u_cont_writes::Dict
    t_copy_reads::DynamicSelection
    u_copy_reads::DynamicSelection
    marked_as_retained::Set
end

function FirstPassState(trace, u::ChoiceMap)
    FirstPassState(
        trace, u, choicemap(), choicemap(),
        Dict(), Dict(), Dict(), Dict(),
        DynamicSelection(), DynamicSelection(),
        Set())
end

struct JacobianPassState{T}
    trace
    u::ChoiceMap
    input_arr::AbstractArray{T}
    output_arr::Array{T}
    t_key_to_index::Dict
    u_key_to_index::Dict
    cont_constraints_key_to_index::Dict
    cont_u_back_key_to_index::Dict
end

const inv_state = gensym("inv_state")

macro involution(ex)
    MacroTools.@capture(ex, function f_(args__) body_ end) || error("expected syntax: function f(..) .. end")

    fn! = gensym("$(esc(f))_fn!")

    quote

    # mutates the state
    function $fn!($(esc(inv_state))::Union{FirstPassState,JacobianPassState}, $(map(esc, args)...))
        $(esc(body))
        nothing
    end

    $(esc(f)) = Involution($fn!)

    end # quote

end # macro involution()

# read discrete from model

macro read_discrete_from_model(addr)
    quote read_discrete_from_model($(esc(inv_state)), $(esc(addr))) end
end

function read_discrete_from_model(state::Union{FirstPassState,JacobianPassState}, addr)
    state.trace[addr]
end

# read discrete from proposal

macro read_discrete_from_proposal(addr)
    quote read_discrete_from_proposal($(esc(inv_state)), $(esc(addr))) end
end

function read_discrete_from_proposal(state::Union{FirstPassState,JacobianPassState}, addr)
    state.u[addr]
end

# write_discrete_to_proposal

macro write_discrete_to_proposal(addr, value)
    quote write_discrete_to_proposal($(esc(inv_state)), $(esc(addr)), $(esc(value))) end
end

function write_discrete_to_proposal(state::FirstPassState, addr, value)
    state.u_back[addr] = value
    value
end

function write_discrete_to_proposal(state::JacobianPassState, addr, value)
    value
end

# write_discrete_to_model

macro write_discrete_to_model(addr, value)
    quote write_discrete_to_model($(esc(inv_state)), $(esc(addr)), $(esc(value))) end
end

function write_discrete_to_model(state::FirstPassState, addr, value)
    state.constraints[addr] = value
    value
end

function write_discrete_to_model(state::JacobianPassState, addr, value)
    value
end

# read continuous from proposal

macro read_continuous_from_proposal(addr)
    quote read_continuous_from_proposal($(esc(inv_state)), $(esc(addr))) end
end

function read_continuous_from_proposal(state::FirstPassState, addr)
    state.u_cont_reads[addr] = state.u[addr]
    state.u[addr]
end

function read_continuous_from_proposal(state::JacobianPassState, addr)
    state.input_arr[state.u_key_to_index[addr]]
end

# read continuous from model

macro read_continuous_from_model(addr)
    quote read_continuous_from_model($(esc(inv_state)), $(esc(addr))) end
end

function read_continuous_from_model(state::FirstPassState, addr)
    state.t_cont_reads[addr] = state.trace[addr]
    state.trace[addr]
end

function read_continuous_from_model(state::JacobianPassState, addr)
    state.input_arr[state.t_key_to_index[addr]]
end

# read continuous from model retained (optional, for efficiency)

macro read_continuous_from_model_retained(addr)
    quote read_continuous_from_model_retained($(esc(inv_state)), $(esc(addr))) end
end

function read_continuous_from_model_retained(state::FirstPassState, addr)
    push!(state.marked_as_retained, addr)
    state.trace[addr]
end

function read_continuous_from_model_retained(state::JacobianPassState, addr)
    state.trace[addr] # read directly from the trace, instead of the array
end

# write_continuous_to_proposal

macro write_continuous_to_proposal(addr, value)
    quote write_continuous_to_proposal($(esc(inv_state)), $(esc(addr)), $(esc(value))) end
end

function write_continuous_to_proposal(state::FirstPassState, addr, value)
    has_value(state.u_back, addr) && error("Proposal address $addr already written to")
    state.u_back[addr] = value
    state.u_cont_writes[addr] = value
    value
end

function write_continuous_to_proposal(state::JacobianPassState, addr, value)
    state.output_arr[state.cont_u_back_key_to_index[addr]] = value
end

# write_continuous_to_model

macro write_continuous_to_model(addr, value)
    quote write_continuous_to_model($(esc(inv_state)), $(esc(addr)), $(esc(value))) end
end

function write_continuous_to_model(state::FirstPassState, addr, value)
    has_value(state.constraints, addr) && error("Model address $addr already written to")
    state.constraints[addr] = value
    state.t_cont_writes[addr] = value
    value
end

function write_continuous_to_model(state::JacobianPassState, addr, value)
    state.output_arr[state.cont_constraints_key_to_index[addr]] = value
end

# copy_model_to_model

macro copy_model_to_model(from_addr, to_addr)
    quote copy_model_to_model($(esc(inv_state)), $(esc(from_addr)), $(esc(to_addr))) end
end

function copy_model_to_model(state::FirstPassState, from_addr, to_addr)
    trace_choices = get_choices(state.trace)
    push!(state.t_copy_reads, from_addr)
    if has_value(trace_choices, from_addr)
        state.constraints[to_addr] = state.trace[from_addr]
    else
        set_submap!(state.constraints, to_addr, get_submap(trace_choices, from_addr))
    end
    nothing
end

function copy_model_to_model(state::JacobianPassState, from_addr, to_addr)
    nothing
end

# copy_model_to_proposal

macro copy_model_to_proposal(from_addr, to_addr)
    quote copy_model_to_proposal($(esc(inv_state)), $(esc(from_addr)), $(esc(to_addr))) end
end

function copy_model_to_proposal(state::FirstPassState, from_addr, to_addr)
    trace_choices = get_choices(state.trace)
    push!(state.t_copy_reads, from_addr)
    if has_value(trace_choices, from_addr)
        state.u_back[to_addr] = state.trace[from_addr]
    else
        set_submap!(state.u_back, to_addr, get_submap(trace_choices, from_addr))
    end
    nothing
end

function copy_model_to_proposal(state::JacobianPassState, from_addr, to_addr)
    nothing
end

# copy_proposal_to_proposal

macro copy_proposal_to_proposal(from_addr, to_addr)
    quote copy_proposal_to_proposal($(esc(inv_state)), $(esc(from_addr)), $(esc(to_addr))) end
end

function copy_proposal_to_proposal(state::FirstPassState, from_addr, to_addr)
    push!(state.u_copy_reads, from_addr)
    if has_value(state.u, from_addr)
        state.u_back[to_addr] = state.u[from_addr]
    else
        set_submap!(state.u_back, to_addr, get_submap(state.u, from_addr))
    end
    nothing
end

function copy_proposal_to_proposal(state::JacobianPassState, from_addr, to_addr)
    nothing
end

# copy_proposal_to_model

macro copy_proposal_to_model(from_addr, to_addr)
    quote copy_proposal_to_model($(esc(inv_state)), $(esc(from_addr)), $(esc(to_addr))) end
end

function copy_proposal_to_model(state::FirstPassState, from_addr, to_addr)
    push!(state.u_copy_reads, from_addr)
    if has_value(state.u, from_addr)
        state.constraints[to_addr] = state.u[from_addr]
    else
        set_submap!(state.constraints, to_addr, get_submap(state.u, from_addr))
    end
    nothing
end

function copy_proposal_to_model(state::JacobianPassState, from_addr, to_addr)
    nothing
end

# call another involution function (NOTE: only the top-level function is actually technically an involution)

macro call(ex)
    MacroTools.@capture(ex, f_(args__)) || error("expected syntax: f(..)")
    quote $(esc(f)).fn!($(esc(inv_state)), $(map(esc, args)...)) end
end

# apply 

function apply_involution(involution::Involution, trace, u, proposal_args, proposal_retval)

    # take the first pass through the involution, mutating first_pass_state
    first_pass_state = FirstPassState(trace, u)
    involution.fn!(first_pass_state, get_args(trace), proposal_args, proposal_retval)

    # create input array and mappings input addresses that are needed for Jacobian
    # exclude addresses that were moved to another address
    input_arr = Vector{Float64}()
    next_input_index = 1

    t_key_to_index = Dict()
    for (addr, v) in first_pass_state.t_cont_reads
        if !(addr in first_pass_state.t_copy_reads)
            t_key_to_index[addr] = next_input_index
            next_input_index += 1
            push!(input_arr, v)
        end
    end

    u_key_to_index = Dict()
    for (addr, v) in first_pass_state.u_cont_reads
        if !(addr in first_pass_state.u_copy_reads) 
            u_key_to_index[addr] = next_input_index
            next_input_index += 1
            push!(input_arr, v)
        end
    end

    # create mappings for output addresses that are needed for Jacobian
    next_output_index = 1

    cont_constraints_key_to_index = Dict()
    for (addr, v) in first_pass_state.t_cont_writes
        cont_constraints_key_to_index[addr] = next_output_index
        next_output_index += 1
    end

    cont_u_back_key_to_index = Dict()
    for (addr, v) in first_pass_state.u_cont_writes
        cont_u_back_key_to_index[addr] = next_output_index
        next_output_index += 1
    end

    function f_array(input_arr::AbstractArray{T}) where {T <: Real}

        # closing over:
        # - trace, u
        # - u_key_to_index, t_key_to_index, cont_constraints_key_to_index, cont_u_back_key_to_index
        # - proposal_args, proposal_retval

        n_output = length(cont_constraints_key_to_index) + length(cont_u_back_key_to_index)
        output_arr = Vector{T}(undef, n_output)

        jacobian_pass_state = JacobianPassState(
            trace, u, input_arr, output_arr, 
            t_key_to_index, u_key_to_index,
            cont_constraints_key_to_index,
            cont_u_back_key_to_index)

        # mutates the state
        involution.fn!(
            jacobian_pass_state, get_args(trace), proposal_args, proposal_retval)

        # return the output array
        output_arr
    end

    # TODO XXX proposal_retval cannot depend on continuous random choices in model
    # or proposal (we might be abel to relax this later using choice_gradients) XXX

    (first_pass_state.constraints, first_pass_state.u_back,
     first_pass_state.marked_as_retained,
     t_key_to_index, input_arr, f_array)
end

function apply_involution_with_corrected_model_weight(
        involution::Involution, trace, proposal_args::Tuple,
        u::ChoiceMap, proposal_retval, check::Bool)

    # apply the involution, and get metadata back about it, including the
    # function f_array from arrays to arrays, that is the restriction of the
    # involution to just a subset of the relevant continuous random choices
    (constraints, u_back, marked_as_retained, t_key_to_index, input_arr, f_array) = apply_involution(
        involution, trace, u, proposal_args, proposal_retval)

    # update model trace
    (new_trace, model_weight, _, discard) = update(
        trace, get_args(trace), map((_) -> NoChange(), get_args(trace)), constraints)

    # check the user's retained assertions
    if check
        for addr in marked_as_retained
            has_value(discard, addr) && error("Address $addr was marked as retained, but was not")
        end
    end

    # compute Jacobian matrix of f_array, where columns are inputs, rows are outputs
    J = ForwardDiff.jacobian(f_array, input_arr)
    @assert size(J)[2] == length(input_arr)
    num_outputs = size(J)[1]

    # remove columns for inputs from the trace that were retained
    # NOTE: these columns did not have to be computed in the first place, if
    # the user had marked the associated random choices as retained
    keep = fill(true, length(input_arr))
    for (addr, index) in t_key_to_index
        if !has_value(discard, addr)
            keep[index] = false
        end
    end
    J = J[:,keep]
    if size(J) != (num_outputs, num_outputs)
        error("Jacobian was not square (size was $(size(J)); the function may not be an involution")
    end

    # log absolute value of Jacobian determinant
    correction = LinearAlgebra.logabsdet(J)[1]

    (new_trace, u_back, model_weight + correction)
end

function rjmcmc(trace, q, proposal_args, involution::Involution;
        check=false, observations=EmptyChoiceMap())

    # run proposal
    u, q_fwd_score, proposal_retval = propose(q, (trace, proposal_args...))

    # apply the involution
    new_trace, u_back, model_score = apply_involution_with_corrected_model_weight(
        involution, trace, proposal_args, u, proposal_retval, check)

    check && Gen.check_observations(get_choices(new_trace), observations)

    # compute proposal backward score
    (q_bwd_score, proposal_retval_back) = assess(q, (new_trace, proposal_args...), u_back)

    # round trip check
    if check
        trace_rt, u_rt, model_score_rt = apply_involution_with_corrected_model_weight(
            involution, new_trace, proposal_args, u_back, proposal_retval_back, check)
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
