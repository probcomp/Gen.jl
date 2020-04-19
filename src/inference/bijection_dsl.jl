import ForwardDiff
import MacroTools
import LinearAlgebra

# See this math writeup for an understanding of how this code works:
# docs/tex/mcmc.pdf

mutable struct BijectionDSLProgram
    fn!::Function
    inverse::Union{Nothing,BijectionDSLProgram}
end

function pair_bijections!(f1::BijectionDSLProgram, f2::BijectionDSLProgram)
    f1.inverse = f2
    f2.inverse = f1
    return nothing
end

function is_involution!(f::BijectionDSLProgram)
    f.inverse = f
    return nothing
end

function inverse(bijection::BijectionDSLProgram)
    if isnothing(bijection.inverse)
        error("inverse bijection was not defined")
    end
    return bijection.inverse
end

const bij_state = gensym("bij_state")

"""
    @bijection function f(...)
        ..
    end

Write a program in the [Involution DSL](@ref).
"""
macro bijection(ex)
    ex = MacroTools.longdef(ex)
    MacroTools.@capture(ex, function f_(args__) body_ end) || error("expected syntax: function f(..) .. end")

    fn! = gensym("$(esc(f))_fn!")

    return quote

        # mutates the state
        function $fn!($(esc(bij_state))::Union{FirstPassState,JacobianPassState}, $(map(esc, args)...))
            $(esc(body))
            return nothing
        end

        Core.@__doc__ $(esc(f)) = BijectionDSLProgram($fn!, nothing)

    end
end

macro bijcall(ex)
    MacroTools.@capture(ex, f_(args__)) || error("expected syntax: f(..)")
    return quote $(esc(f)).fn!($(esc(bij_state)), $(map(esc, args)...)) end
end

macro read_discrete_from_model(addr)
    return quote read_discrete_from_model($(esc(bij_state)), $(esc(addr))) end
end

macro read_discrete_from_proposal(addr)
    return quote read_discrete_from_proposal($(esc(bij_state)), $(esc(addr))) end
end

macro write_discrete_to_proposal(addr, value)
    return quote write_discrete_to_proposal($(esc(bij_state)), $(esc(addr)), $(esc(value))) end
end

macro write_discrete_to_model(addr, value)
    return quote write_discrete_to_model($(esc(bij_state)), $(esc(addr)), $(esc(value))) end
end

macro read_continuous_from_proposal(addr)
    return quote read_continuous_from_proposal($(esc(bij_state)), $(esc(addr))) end
end

macro read_continuous_from_model(addr)
    return quote read_continuous_from_model($(esc(bij_state)), $(esc(addr))) end
end

macro write_continuous_to_proposal(addr, value)
    return quote write_continuous_to_proposal($(esc(bij_state)), $(esc(addr)), $(esc(value))) end
end

macro write_continuous_to_model(addr, value)
    return quote write_continuous_to_model($(esc(bij_state)), $(esc(addr)), $(esc(value))) end
end

macro copy_model_to_model(from_addr, to_addr)
    return quote copy_model_to_model($(esc(bij_state)), $(esc(from_addr)), $(esc(to_addr))) end
end

macro copy_model_to_proposal(from_addr, to_addr)
    return quote copy_model_to_proposal($(esc(bij_state)), $(esc(from_addr)), $(esc(to_addr))) end
end

macro copy_proposal_to_proposal(from_addr, to_addr)
    return quote copy_proposal_to_proposal($(esc(bij_state)), $(esc(from_addr)), $(esc(to_addr))) end
end

macro copy_proposal_to_model(from_addr, to_addr)
    return quote copy_proposal_to_model($(esc(bij_state)), $(esc(from_addr)), $(esc(to_addr))) end
end


################################
# first pass through bijection #
################################

struct FirstPassResults

    "subset of the output model choice map ``t'``"
    constraints::ChoiceMap

    "output proposal choice map ``u'``"
    u_back::ChoiceMap
    
    t_cont_reads::Dict
    u_cont_reads::Dict
    t_cont_writes::Dict
    u_cont_writes::Dict
    t_copy_reads::DynamicSelection
    u_copy_reads::DynamicSelection
end

function FirstPassResults()
    return FirstPassResults(
        choicemap(), choicemap(),
        Dict(), Dict(), Dict(), Dict(),
        DynamicSelection(), DynamicSelection())
end

struct FirstPassState

    "trace containing the input model choice map ``t``"
    trace

    "the input proposal choice map ``u``"
    u::ChoiceMap

    results::FirstPassResults
end

function FirstPassState(trace, u::ChoiceMap)
    return FirstPassState(trace, u, FirstPassResults())
end

function run_first_pass(bijection::BijectionDSLProgram, prev_model_trace, proposal_trace)
    state = FirstPassState(prev_model_trace, get_choices(proposal_trace))
    bijection.fn!(state, get_args(prev_model_trace), get_args(proposal_trace)[2:end], get_retval(proposal_trace))
    return state.results
end

function read_discrete_from_model(state::FirstPassState, addr)
    return state.trace[addr]
end

function read_discrete_from_proposal(state::FirstPassState, addr)
    return state.u[addr]
end

function write_discrete_to_proposal(state::FirstPassState, addr, value)
    state.results.u_back[addr] = value
    return value
end

function write_discrete_to_model(state::FirstPassState, addr, value)
    state.results.constraints[addr] = value
    return value
end

function read_continuous_from_proposal(state::FirstPassState, addr)
    state.results.u_cont_reads[addr] = state.u[addr]
    return state.u[addr]
end

function read_continuous_from_model(state::FirstPassState, addr)
    state.results.t_cont_reads[addr] = state.trace[addr]
    return state.trace[addr]
end

function write_continuous_to_proposal(state::FirstPassState, addr, value)
    has_value(state.results.u_back, addr) && error("Proposal address $addr already written to")
    state.results.u_back[addr] = value
    state.results.u_cont_writes[addr] = value
    return value
end

function write_continuous_to_model(state::FirstPassState, addr, value)
    has_value(state.results.constraints, addr) && error("Model address $addr already written to")
    state.results.constraints[addr] = value
    state.results.t_cont_writes[addr] = value
    return value
end

function copy_model_to_model(state::FirstPassState, from_addr, to_addr)
    trace_choices = get_choices(state.trace)
    push!(state.results.t_copy_reads, from_addr)
    if has_value(trace_choices, from_addr)
        state.results.constraints[to_addr] = state.trace[from_addr]
    else
        set_submap!(state.results.constraints, to_addr, get_submap(trace_choices, from_addr))
    end
    return nothing
end

function copy_model_to_proposal(state::FirstPassState, from_addr, to_addr)
    trace_choices = get_choices(state.trace)
    push!(state.results.t_copy_reads, from_addr)
    if has_value(trace_choices, from_addr)
        state.results.u_back[to_addr] = state.trace[from_addr]
    else
        set_submap!(state.results.u_back, to_addr, get_submap(trace_choices, from_addr))
    end
    return nothing
end

function copy_proposal_to_proposal(state::FirstPassState, from_addr, to_addr)
    push!(state.results.u_copy_reads, from_addr)
    if has_value(state.u, from_addr)
        state.results.u_back[to_addr] = state.u[from_addr]
    else
        set_submap!(state.results.u_back, to_addr, get_submap(state.u, from_addr))
    end
    return nothing
end

function copy_proposal_to_model(state::FirstPassState, from_addr, to_addr)
    push!(state.results.u_copy_reads, from_addr)
    if has_value(state.u, from_addr)
        state.results.constraints[to_addr] = state.u[from_addr]
    else
        set_submap!(state.results.constraints, to_addr, get_submap(state.u, from_addr))
    end
    return nothing
end

#####################################################################
# second pass through bijection (gets automatically differentiated) #
#####################################################################

struct JacobianPassState{T<:Real}
    trace
    u::ChoiceMap
    input_arr::AbstractArray{T}
    output_arr::Array{T}
    t_key_to_index::Dict
    u_key_to_index::Dict
    cont_constraints_key_to_index::Dict
    cont_u_back_key_to_index::Dict
end


function read_discrete_from_model(state::JacobianPassState, addr)
    return state.trace[addr]
end

function read_discrete_from_proposal(state::JacobianPassState, addr)
    return state.u[addr]
end

function write_discrete_to_proposal(state::JacobianPassState, addr, value)
    return value
end

function write_discrete_to_model(state::JacobianPassState, addr, value)
    return value
end

function _read_continuous(input_arr, addr_info::Int)
    return input_arr[addr_info]
end

function _read_continuous(input_arr, addr_info::Tuple{Int,Int})
    # TODO to handle things other than vectors, store shape in addr info and reshape?
    (start_idx, len) = addr_info
    return input_arr[start_idx:start_idx+len]
 end

function read_continuous_from_proposal(state::JacobianPassState, addr)
    if haskey(state.u_key_to_index, addr)
        return _read_continuous(state.input_arr, state.u_key_to_index[addr])
    else
        return state.u[addr]
    end
end

function read_continuous_from_model(state::JacobianPassState, addr)
    if haskey(state.t_key_to_index, addr)
        return _read_continuous(state.input_arr, state.t_key_to_index[addr])
    else
        return state.trace[addr]
    end
end

function _write_continuous(output_arr, addr_info::Int, value)
    return output_arr[addr_info] = value
end

function _write_continuous(output_arr, addr_info::Tuple{Int,Int}, value)
    (start_idx, len) = addr_info
    return output_arr[start_idx:start_idx+len] = value
end

function write_continuous_to_proposal(state::JacobianPassState, addr, value)
    return _write_continuous(state.output_arr, state.cont_u_back_key_to_index[addr], value)
end

function write_continuous_to_model(state::JacobianPassState, addr, value)
    return _write_continuous(state.output_arr, state.cont_constraints_key_to_index[addr], value)
end

function copy_model_to_model(state::JacobianPassState, from_addr, to_addr)
    return nothing
end

function copy_model_to_proposal(state::JacobianPassState, from_addr, to_addr)
    return nothing
end

function copy_proposal_to_proposal(state::JacobianPassState, from_addr, to_addr)
    return nothing
end

function copy_proposal_to_model(state::JacobianPassState, from_addr, to_addr)
    return nothing
end


#################################
# computing jacobian correction #
#################################

discard_skip_read_addr(addr, discard::ChoiceMap) = !has_value(discard, addr)
discard_skip_read_addr(addr, discard::Nothing) = false

function store_addr_info!(dict::Dict, addr, value::Real, next_index::Int)
    dict[addr] = next_index 
    return 1 # number of elements of array
end

function store_addr_info!(dict::Dict, addr, value::AbstractArray{<:Real}, next_index::Int)
    len = length(value)
    dict[addr] = (next_index, len)
    return len # number of elements of array
end

function assemble_input_array_and_maps(
        t_cont_reads, t_copy_reads, u_cont_reads, u_copy_reads, discard::Union{ChoiceMap,Nothing})

    input_arr = Vector{Float64}()
    next_input_index = 1

    t_key_to_index = Dict()
    for (addr, v) in t_cont_reads
        if addr in t_copy_reads
            continue
        end
        if discard_skip_read_addr(addr, discard)
            # note: only happens when the model is unchanged
            continue
        end
        next_input_index += store_addr_info!(t_key_to_index, addr, v, next_input_index)
        append!(input_arr, v)
    end

    u_key_to_index = Dict()
    for (addr, v) in u_cont_reads
        if addr in u_copy_reads
            continue
        end
        next_input_index += store_addr_info!(u_key_to_index, addr, v, next_input_index)
        append!(input_arr, v)
    end

    return (t_key_to_index, u_key_to_index, input_arr)
end

function assemble_output_maps(t_cont_writes, u_cont_writes)
    next_output_index = 1

    cont_constraints_key_to_index = Dict()
    for (addr, v) in t_cont_writes
        next_output_index += store_addr_info!(cont_constraints_key_to_index, addr, v, next_output_index)
    end

    cont_u_back_key_to_index = Dict()
    for (addr, v) in u_cont_writes
        next_output_index += store_addr_info!(cont_u_back_key_to_index, addr, v, next_output_index)
    end

    return (cont_constraints_key_to_index, cont_u_back_key_to_index)
end

function jacobian_correction(bijection::BijectionDSLProgram, prev_model_trace, proposal_trace, first_pass_results, discard)

    # create input array and mappings input addresses that are needed for Jacobian
    # exclude addresses that were copied explicitly to another address
    (t_key_to_index, u_key_to_index, input_arr) = assemble_input_array_and_maps(
        first_pass_results.t_cont_reads,
        first_pass_results.t_copy_reads,
        first_pass_results.u_cont_reads,
        first_pass_results.u_copy_reads, discard)
    
    # create mappings for output addresses that are needed for Jacobian
    (cont_constraints_key_to_index, cont_u_back_key_to_index) = assemble_output_maps(
        first_pass_results.t_cont_writes,
        first_pass_results.u_cont_writes)

    # this function is the partial application of the continuous part of the
    # bijection, with inputs corresponding to a particular superset of the
    # columns of the reduced Jacobian matrix
    function f_array(input_arr::AbstractArray{T}) where {T <: Real}

        # closing over:
        # - trace, u
        # - u_key_to_index, t_key_to_index, cont_constraints_key_to_index, cont_u_back_key_to_index
        # - proposal_args, proposal_retval

        n_output = length(cont_constraints_key_to_index) + length(cont_u_back_key_to_index)
        output_arr = Vector{T}(undef, n_output)

        jacobian_pass_state = JacobianPassState(
            prev_model_trace, get_choices(proposal_trace), input_arr, output_arr, 
            t_key_to_index, u_key_to_index,
            cont_constraints_key_to_index,
            cont_u_back_key_to_index)

        # mutates the state
        bijection.fn!(
            jacobian_pass_state, get_args(prev_model_trace), get_args(proposal_trace)[2:end], get_retval(proposal_trace))

        # return the output array
        output_arr
    end

    # compute Jacobian matrix of f_array, where columns are inputs, rows are outputs
    J = ForwardDiff.jacobian(f_array, input_arr)
    @assert size(J)[2] == length(input_arr)
    num_outputs = size(J)[1]
    if size(J) != (num_outputs, num_outputs)
        error("Jacobian was not square (size was $(size(J)); the function may not be an bijection")
    end

    # log absolute value of Jacobian determinant
    correction = LinearAlgebra.logabsdet(J)[1]
    if isinf(correction)
        @error "Weight correction is infinite; the function may not be an bijection"
    end
    
    return correction
end


########################################
# different forms of calling bijection #
########################################

function check_round_trip(
            prev_model_trace, prev_model_trace_rt,
            forward_proposal_trace, forward_proposal_trace_rt,
            model_weight, model_weight_inv)

    forward_proposal_choices = get_choices(forward_proposal_trace)
    forward_proposal_choices_rt = get_choices(forward_proposal_trace_rt)
    prev_model_choices = get_choices(prev_model_trace)
    prev_model_choices_rt = get_choices(prev_model_trace_rt)
    if !isapprox(forward_proposal_choices, forward_proposal_choices_rt)
        @error("forward proposal choices: $(sprint(show, "text/plain", forward_proposal_choices))")
        @error("forward proposal choices after round trip: $(sprint(show, "text/plain", forward_proposal_choices_rt))")
        error("bijection round trip check failed")
    end
    if !isapprox(prev_model_choices, prev_model_choices_rt)
        @error "previous model choices: $(sprint(show, "text/plain", prev_model_choices))"
        @error "previous model choices after round trip: $(sprint(show, "text/plain", prev_model_choices_rt))"
        error("bijection round trip check failed")
    end
    if !isapprox(model_weight, -model_weight_inv)
        @error "model weight: $model_weight, inverse model weight: $model_weight_inv"
        error("bijection round trip check failed")
    end
    return nothing
end


"""
    (bijection::BijectionDSLProgram)(
        prev_model_trace::Trace, proposal_trace::Trace,
        new_model::GenerativeFunction, new_model_args::Tuple,
        new_constraints::ChoiceMap)

Apply bijection with a change to the model

Appropriate for use in sequential Monte Carlo (SMC) kernels.
"""
function (bijection::BijectionDSLProgram)(
        prev_model_trace::Trace, forward_proposal_trace::Trace,
        backward_proposal::GenerativeFunction, backward_proposal_args::Tuple,
        new_model::GenerativeFunction, new_model_args::Tuple,
        new_observations::ChoiceMap;
        check=false, prev_observations=EmptyChoiceMap())

    # run the bijection forward
    first_pass_results = run_first_pass(bijection, prev_model_trace, forward_proposal_trace)
    
    # construct new trace and get model weight
    new_model_trace, new_model_score = generate(
        new_model, new_model_args,
        merge(first_pass_results.constraints, new_observations))
    prev_model_score = get_score(prev_model_trace)
    model_weight = new_model_score - prev_model_score

    # jacobian correction
    discard = nothing
    model_weight += jacobian_correction(bijection, prev_model_trace, forward_proposal_trace, first_pass_results, discard)

    # get backward proposal trace
    # TODO check that proposal is fully constrained in the backward direction
    backward_proposal_trace, = generate(backward_proposal, (new_model_trace, backward_proposal_args...), first_pass_results.u_back)

    if check
        forward_proposal_choices = get_choices(forward_proposal_trace)
        bijection_inv = inverse(bijection)
        (prev_model_trace_rt, forward_proposal_trace_rt, inv_model_weight) = bijection_inv(
            new_model_trace, backward_proposal_trace,
            get_gen_fn(forward_proposal_trace), get_args(forward_proposal_trace)[2:end],
            get_gen_fn(prev_model_trace), get_args(prev_model_trace),
            prev_observations; check=false)
        check_round_trip(
            prev_model_trace, prev_model_trace_rt,
            forward_proposal_trace, forward_proposal_trace_rt,
            model_weight, inv_model_weight)
    end

    return (new_model_trace, backward_proposal_trace, model_weight)
end

"""
    (bijection::BijectionDSLProgram)(
        prev_model_trace::Trace, proposal_trace::Trace, new_args::Tuple;
        argdiffs=map((_) -> UnknownChange(), new_args), check=false)

Apply bijection with no change to the model, but a change to the model arguments.

Appropriate for use in sequential Monte Carlo (SMC) kernels.
"""
function (bijection::BijectionDSLProgram)(
        prev_model_trace::Trace, forward_proposal_trace::Trace,
        backward_proposal::GenerativeFunction, backward_proposal_args::Tuple,
        new_model_args::Tuple, observations::ChoiceMap;
        argdiffs=map((_) -> UnknownChange(), new_model_args), check=false,
        prev_observations=EmptyChoiceMap())

    # run the partial bijection forward
    first_pass_results = run_first_pass(bijection, prev_model_trace, forward_proposal_trace)

    # finish running the bijection via update
    (new_model_trace, model_weight, _, discard) = update(
        prev_model_trace, new_model_args, argdiffs,
        merge(first_pass_results.constraints, observations))

    # jacobian correction
    model_weight += jacobian_correction(bijection, prev_model_trace, forward_proposal_trace, first_pass_results, discard)

    # get backward proposal trace
    # TODO check that proposal is fully constrained in the backward direction
    backward_proposal_trace, = generate(backward_proposal, (new_model_trace, backward_proposal_args...), first_pass_results.u_back)

    if check
        forward_proposal_choices = get_choices(forward_proposal_trace)
        bijection_inv = inverse(bijection)
        (prev_model_trace_rt, forward_proposal_trace_rt, inv_model_weight) = bijection_inv(
            new_model_trace, backward_proposal_trace,
            get_gen_fn(forward_proposal_trace), get_args(forward_proposal_trace)[2:end],
            get_args(prev_model_trace), prev_observations; check=false)
        check_round_trip(
            prev_model_trace, prev_model_trace_rt,
            forward_proposal_trace, forward_proposal_trace_rt,
            model_weight, inv_model_weight)
    end

    return (new_model_trace, backward_proposal_trace, model_weight)
end

"""
    (bijection::BijectionDSLProgram)(
        prev_model_trace::Trace, proposal_trace::Trace; check=false)

Apply bijection with no change to model or the model arguments.

Appropriate for use in Markov chain Monte Carlo (MCMC) kernels.
"""
function (bijection::BijectionDSLProgram)(
        prev_model_trace::Trace, forward_proposal_trace::Trace; check=false)

    # run the partial bijection forward
    first_pass_results = run_first_pass(bijection, prev_model_trace, forward_proposal_trace)

    # finish running the bijection via update
    (new_model_trace, model_weight, _, discard) = update(
        prev_model_trace, get_args(prev_model_trace),
        map((_) -> NoChange(), get_args(prev_model_trace)),
        first_pass_results.constraints)

    # jacobian correction
    model_weight += jacobian_correction(bijection, prev_model_trace, forward_proposal_trace, first_pass_results, discard)

    # get backward proposal trace
    # TODO check that proposal is fully constrained in the backward direction
    proposal = get_gen_fn(forward_proposal_trace)
    proposal_args = get_args(forward_proposal_trace)[2:end]
    backward_proposal_trace, = generate(proposal, (new_model_trace, proposal_args...), first_pass_results.u_back)

    if check
        forward_proposal_choices = get_choices(forward_proposal_trace)
        bijection_inv = inverse(bijection)
        (prev_model_trace_rt, forward_proposal_trace_rt, inv_model_weight) = bijection_inv(
            new_model_trace, backward_proposal_trace; check=false)
        check_round_trip(
            prev_model_trace, prev_model_trace_rt,
            forward_proposal_trace, forward_proposal_trace_rt,
            model_weight, inv_model_weight)
    end

    return (new_model_trace, backward_proposal_trace, model_weight)
end

export @bijection
export @read_discrete_from_proposal, @read_discrete_from_model
export @write_discrete_to_proposal, @write_discrete_to_model
export @read_continuous_from_proposal, @read_continuous_from_model
export @write_continuous_to_proposal, @write_continuous_to_model
export @copy_model_to_model, @copy_model_to_proposal
export @copy_proposal_to_proposal, @copy_proposal_to_model
export @bijcall
export pair_bijections!, is_involution!, inverse
