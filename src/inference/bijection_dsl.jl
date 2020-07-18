import ForwardDiff
import MacroTools
import LinearAlgebra

# See this math writeup for an understanding of how this code works:
# docs/tex/mcmc.pdf

"""
    BijectionDSLProgram

A program compiled from the [Trace Bijection DSL](@ref).
"""
mutable struct BijectionDSLProgram
    fn!::Function
    inverse::Union{Nothing,BijectionDSLProgram}
end

"""
    pair_bijections!(f1::BijectionDSLProgram, f2::BijectionDSLProgram)

Assert that a pair of bijections contsructed using the [Trace Bijection DSL](@ref) are inverses of one another.
"""
function pair_bijections!(f1::BijectionDSLProgram, f2::BijectionDSLProgram)
    f1.inverse = f2
    f2.inverse = f1
    return nothing
end

"""
    is_involution!(f::BijectionDSLProgram)

Assert that a bijection constructed with the [Trace Bijection DSL](@ref) is its own inverse.
"""
function is_involution!(f::BijectionDSLProgram)
    f.inverse = f
    return nothing
end

"""
    b::BijectionDSLProgram = inverse(a::BijectionDSLProgram)

Obtain the inverse of a bijection that was constructed with the [Trace Bijection DSL](@ref).

The inverse must have been associated with the bijection either via [`pair_bijections!`](@ref) or [`is_involution!`])(@ref).
"""
function inverse(bijection::BijectionDSLProgram)
    if isnothing(bijection.inverse)
        error("inverse bijection was not defined")
    end
    return bijection.inverse
end

struct ModelInputTraceToken{T}
    args::T
end

struct AuxInputTraceToken{T}
    args::T
end

struct ModelInputTraceRetValToken
end

struct AuxInputTraceRetValToken
end

struct ModelOutputTraceToken 
end

struct AuxOutputTraceToken
end

struct ModelInputAddress{T}
    addr::T
end

struct AuxInputAddress{T}
    addr::T
end

struct ModelOutputAddress{T}
    addr::T
end

struct AuxOutputAddress{T}
    addr::T
end

Base.getindex(::ModelInputTraceToken, addr) = ModelInputAddress(addr) # model_in[addr]
Base.getindex(::ModelOutputTraceToken, addr) = ModelOutputAddress(addr) # model_out[addr]
Base.getindex(::AuxInputTraceToken, addr) = AuxInputAddress(addr) # aux_in[addr]
Base.getindex(::AuxOutputTraceToken, addr) = AuxOutputAddress(addr) # aux_out[addr]
Base.getindex(::ModelInputTraceToken) = ModelInputTraceRetvalToken() # model_in[]
Base.getindex(::AuxInputTraceToken) = AuxInputTraceRetValToken() # aux_in[]
get_args(token::ModelInputTraceToken) = token.args # get_args(model_in)
get_args(token::AuxInputTraceToken) = token.args # get_args(aux_in)

const bij_state = gensym("bij_state")

"""
    @transform f (model_in, aux_in) to (model_out, aux_out)
        ..
    end

Write a program in the [Trace Transform DSL](@ref).
"""
macro transform(f_expr, from_expr, to_symbol::Symbol, to_expr, body)
    syntax_err = "valid syntactic forms:\n@transform f (..) to (..) begin .. end\n@transform f(..) (..) to (..) begin .. end"
    err = false
    if MacroTools.@capture(f_expr, f_(args__))
    elseif MacroTools.@capture(f_expr, f_)
        args = []
    else
        err = true
    end
    err = err || (to_symbol != :to)
    MacroTools.@capture(from_expr,
         (model_in_, aux_in_)) || (err = true)
    MacroTools.@capture(to_expr,
         (model_out_, aux_out_)) || (err = true)
    if err
        @error(syntax_err)
        error("invalid @transform syntax")
    end

    fn! = gensym("$(esc(f))_fn!")

    return quote

        # mutates the state
        function $fn!(
                $(esc(bij_state))::Union{FirstPassState,JacobianPassState},
                $(map(esc, args)...))
            model_args = get_args($(esc(bij_state)).model_trace)
            aux_args = get_args($(esc(bij_state)).aux_trace)
            $(esc(model_in)) = ModelInputTraceToken(model_args)
            $(esc(model_out)) = ModelOutputTraceToken()
            $(esc(aux_in)) = AuxInputTraceToken(aux_args)
            $(esc(aux_out)) = AuxOutputTraceToken()
            $(esc(body))
            return nothing
        end

        Core.@__doc__ $(esc(f)) = BijectionDSLProgram($fn!, nothing)

    end
end

macro tcall(ex)
    MacroTools.@capture(ex, f_(args__)) || error("expected syntax: f(..)")
    return quote $(esc(f)).fn!($(esc(bij_state)), $(map(esc, args)...)) end
end

# handlers

struct DiscreteAnn end
struct ContinuousAnn end

const DISCRETE = [:discrete, :disc]
const CONTINUOUS = [:continuous, :cont]

function typed(annotation::Symbol)
    if annotation in DISCRETE
        return DiscreteAnn()
    elseif annotation in CONTINUOUS
        return ContinuousAnn()
    else
        error("error")
    end
end

macro read(src, ann::QuoteNode)
    return quote read($(esc(bij_state)), $(esc(src)), $(esc(typed(ann.value)))) end
end

macro write(dest, val, ann::QuoteNode)
    return quote write($(esc(bij_state)), $(esc(dest)), $(esc(val)), $(esc(typed(ann.value)))) end
end

macro copy(src, dest)
    return quote copy($(esc(bij_state)), $(esc(src)), $(esc(dest))) end
end

# TODO make more consistent by allowing us to read any hierarchical address,
# including return values of intermediate calls, not just the top-level call.

# TODO add haskey(model_in, addr), and haskey(aux_in, addr)

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
    model_trace

    "the input proposal choice map ``u``"
    aux_trace

    results::FirstPassResults
end

function FirstPassState(model_trace, aux_trace)
    return FirstPassState(model_trace, aux_trace, FirstPassResults())
end

function run_first_pass(bijection::BijectionDSLProgram, model_trace, aux_trace)
    state = FirstPassState(model_trace, aux_trace)
    bijection.fn!(state) # TODO allow for other args to top-level transform function
    return state.results
end

function read(state::FirstPassState, src::ModelInputTraceRetValToken, ::DiscreteAnn)
    return get_retval(state.model_trace)
end

function read(state::FirstPassState, src::AuxInputTraceRetValToken, ::DiscreteAnn)
    return get_retval(state.aux_trace)
end

function read(state::FirstPassState, src::ModelInputAddress, ::DiscreteAnn)
    addr = src.addr
    return state.model_trace[addr]
end

function read(state::FirstPassState, src::ModelInputAddress, ::ContinuousAnn)
    addr = src.addr
    state.results.t_cont_reads[addr] = state.model_trace[addr]
    return state.model_trace[addr]
end

function read(state::FirstPassState, src::AuxInputAddress, ::DiscreteAnn)
    addr = src.addr
    return state.aux_trace[addr]
end

function read(state::FirstPassState, src::AuxInputAddress, ::ContinuousAnn)
    addr = src.addr
    state.results.u_cont_reads[addr] = state.aux_trace[addr]
    return state.aux_trace[addr]
end

function write(state::FirstPassState, dest::ModelOutputAddress, value, ::DiscreteAnn)
    addr = dest.addr
    state.results.constraints[addr] = value
    return value
end

function write(state::FirstPassState, dest::ModelOutputAddress, value, ::ContinuousAnn)
    addr = dest.addr
    has_value(state.results.constraints, addr) && error("Model address $addr already written to")
    state.results.constraints[addr] = value
    state.results.t_cont_writes[addr] = value
    return value
end

function write(state::FirstPassState, dest::AuxOutputAddress, value, ::DiscreteAnn)
    addr = dest.addr
    state.results.u_back[addr] = value
    return value
end

function write(state::FirstPassState, dest::AuxOutputAddress, value, ::ContinuousAnn)
    addr = dest.addr
    has_value(state.results.u_back, addr) && error("Proposal address $addr already written to")
    state.results.u_back[addr] = value
    state.results.u_cont_writes[addr] = value
    return value
end

function copy(state::FirstPassState, src::ModelInputAddress, dest::ModelOutputAddress) 
    from_addr, to_addr = src.addr, dest.addr
    model_choices = get_choices(state.model_trace)
    push!(state.results.t_copy_reads, from_addr)
    if has_value(model_choices, from_addr)
        state.results.constraints[to_addr] = model_choices[from_addr]
    else
        set_submap!(state.results.constraints, to_addr, get_submap(model_choices, from_addr))
    end
    return nothing
end

function copy(state::FirstPassState, src::ModelInputAddress, dest::AuxOutputAddress)
    from_addr, to_addr = src.addr, dest.addr
    model_choices = get_choices(state.model_trace)
    push!(state.results.t_copy_reads, from_addr)
    if has_value(model_choices, from_addr)
        state.results.u_back[to_addr] = model_choices[from_addr]
    else
        set_submap!(state.results.u_back, to_addr, get_submap(model_choices, from_addr))
    end
    return nothing
end

function copy(state::FirstPassState, src::AuxInputAddress, dest::AuxOutputAddress)
    from_addr, to_addr = src.addr, dest.addr
    push!(state.results.u_copy_reads, from_addr)
    aux_choices = get_choices(state.aux_trace)
    if has_value(aux_choices, from_addr)
        state.results.u_back[to_addr] = aux_choices[from_addr]
    else
        set_submap!(state.results.u_back, to_addr, get_submap(aux_choices, from_addr))
    end
    return nothing
end

function copy(state::FirstPassState, src::AuxInputAddress, dest::ModelOutputAddress)
    from_addr, to_addr = src.addr, dest.addr
    push!(state.results.u_copy_reads, from_addr)
    aux_choices = get_choices(state.aux_trace)
    if has_value(aux_choices, from_addr)
        state.results.constraints[to_addr] = aux_choices[from_addr]
    else
        set_submap!(state.results.constraints, to_addr, get_submap(aux_choices, from_addr))
    end
    return nothing
end

#####################################################################
# second pass through bijection (gets automatically differentiated) #
#####################################################################

struct JacobianPassState{T<:Real}
    model_trace
    aux_trace
    input_arr::AbstractArray{T}
    output_arr::Array{T}
    t_key_to_index::Dict
    u_key_to_index::Dict
    cont_constraints_key_to_index::Dict
    cont_u_back_key_to_index::Dict
end

function read(state::JacobianPassState, src::ModelInputTraceRetValToken, ::DiscreteAnn)
    return get_retval(state.model_trace)
end

function read(state::JacobianPassState, src::AuxInputTraceRetValToken, ::DiscreteAnn)
    return get_retval(state.aux_trace)
end

function read(state::JacobianPassState, src::ModelInputAddress, ::DiscreteAnn)
    addr = src.addr
    return state.model_trace[addr]
end

function read(state::JacobianPassState, src::AuxInputAddress, ::DiscreteAnn)
    addr = src.addr
    return state.aux_trace[addr]
end

function _read_continuous(input_arr, addr_info::Int)
    return input_arr[addr_info]
end

function _read_continuous(input_arr, addr_info::Tuple{Int,Int})
    # TODO to handle things other than vectors, store shape in addr info and reshape?
    (start_idx, len) = addr_info
    return input_arr[start_idx:start_idx+len-1]
 end

function read(state::JacobianPassState, src::ModelInputAddress, ::ContinuousAnn)
    addr = src.addr
    if haskey(state.t_key_to_index, addr)
        return _read_continuous(state.input_arr, state.t_key_to_index[addr])
    else
        return state.model_trace[addr]
    end
end

function read(state::JacobianPassState, src::AuxInputAddress, ::ContinuousAnn)
    addr = src.addr
    if haskey(state.u_key_to_index, addr)
        return _read_continuous(state.input_arr, state.u_key_to_index[addr])
    else
        return state.aux_trace[addr]
    end
end

function write(state::JacobianPassState, dest::ModelOutputAddress, value, ::DiscreteAnn)
    return value
end

function write(state::JacobianPassState, dest::AuxOutputAddress, value, ::DiscreteAnn)
    return value
end

function _write_continuous(output_arr, addr_info::Int, value)
    return output_arr[addr_info] = value
end

function _write_continuous(output_arr, addr_info::Tuple{Int,Int}, value)
    (start_idx, len) = addr_info
    return output_arr[start_idx:start_idx+len-1] = value
end

function write(state::JacobianPassState, dest::AuxOutputAddress, value, ::ContinuousAnn)
    addr = dest.addr
    return _write_continuous(state.output_arr, state.cont_u_back_key_to_index[addr], value)
end

function write(state::JacobianPassState, dest::ModelOutputAddress, value, ::ContinuousAnn)
    addr = dest.addr
    return _write_continuous(state.output_arr, state.cont_constraints_key_to_index[addr], value)
end

function copy(state::JacobianPassState, src, dest)
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

    return (cont_constraints_key_to_index, cont_u_back_key_to_index, next_output_index-1)
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
    (cont_constraints_key_to_index, cont_u_back_key_to_index, n_output) = assemble_output_maps(
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

        output_arr = Vector{T}(undef, n_output)

        jacobian_pass_state = JacobianPassState(
            prev_model_trace, proposal_trace, input_arr, output_arr, 
            t_key_to_index, u_key_to_index,
            cont_constraints_key_to_index,
            cont_u_back_key_to_index)

        # mutates the state
        bijection.fn!(jacobian_pass_state)

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
    #if !isapprox(model_weight, -model_weight_inv)
        #@error "model weight: $model_weight, inverse model weight: $model_weight_inv"
        #error("bijection round trip check failed")
    #end
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

export @transform
export @read, @write, @copy, @tcall
export BijectionDSLProgram, pair_bijections!, is_involution!, inverse
