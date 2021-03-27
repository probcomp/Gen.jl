import ForwardDiff
import MacroTools
import LinearAlgebra
import Parameters: @with_kw, @unpack

#######################
# trace transform DSL #
#######################

# See this math writeup for an understanding of how this code works:
# docs/tex/mcmc.pdf

"""
    TraceTransformDSLProgram

A program compiled from the [Trace Transform DSL](@ref).
"""
mutable struct TraceTransformDSLProgram
    fn!::Function
    inverse::Union{Nothing,TraceTransformDSLProgram}
end

"""
    pair_bijections!(f1::TraceTransformDSLProgram, f2::TraceTransformDSLProgram)

Assert that a pair of bijections contsructed using the [Trace Transform DSL](@ref) are
inverses of one another.
"""
function pair_bijections!(f1::TraceTransformDSLProgram, f2::TraceTransformDSLProgram)
    f1.inverse = f2
    f2.inverse = f1
    return nothing
end

"""
    is_involution!(f::TraceTransformDSLProgram)

Assert that a bijection constructed with the [Trace Transform DSL](@ref) is its own inverse.
"""
function is_involution!(f::TraceTransformDSLProgram)
    f.inverse = f
    return nothing
end

"""
    b::TraceTransformDSLProgram = inverse(a::TraceTransformDSLProgram)

Obtain the inverse of a bijection that was constructed with the [Trace Transform DSL](@ref).

The inverse must have been associated with the bijection either via
[`pair_bijections!`](@ref) or [`is_involution!`])(@ref).
"""
function inverse(bijection::TraceTransformDSLProgram)
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
    @transform f[(params...)] (in1 [,in2]) to (out1 [,out2])
        ..
    end

Write a program in the [Trace Transform DSL](@ref).
"""
macro transform(f_expr, src_expr, to_symbol::Symbol, dest_expr, body)
    syntax_err = """valid syntactic forms:
        @transform f (..) to (..) begin .. end
        @transform f(..) (..) to (..) begin .. end"""
    err = false
    if MacroTools.@capture(f_expr, f_(args__))
    elseif MacroTools.@capture(f_expr, f_)
        args = []
    else
        err = true
    end
    err = err || (to_symbol != :to)
    if MacroTools.@capture(src_expr, (model_in_, aux_in_))
    elseif MacroTools.@capture(src_expr, (model_in_))
        aux_in = gensym("dummy_aux")
    else
        err = true
    end
    if MacroTools.@capture(dest_expr, (model_out_, aux_out_))
    elseif MacroTools.@capture(dest_expr, (model_out_))
        aux_out = gensym("dummy_aux")
    else
        err = true
    end
    if err error(syntax_err) end

    fn! = gensym(Symbol(f, "_fn!"))
    return quote
        # mutates the state
        function $fn!(
                $(esc(bij_state))::Union{FirstPassState,JacobianPassState},
                $(map(esc, args)...))
            model_args = get_model_args($(esc(bij_state)))
            aux_args = get_aux_args($(esc(bij_state)))
            $(esc(model_in)) = ModelInputTraceToken(model_args)
            $(esc(model_out)) = ModelOutputTraceToken()
            $(esc(aux_in)) = AuxInputTraceToken(aux_args)
            $(esc(aux_out)) = AuxOutputTraceToken()
            $(esc(body))
            return nothing
        end
        Core.@__doc__ $(esc(f)) = TraceTransformDSLProgram($fn!, nothing)
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

"""
    @read(<source>, <annotation>)

Macro for reading the value of a random choice from an input trace in the [Trace Transform DSL](@ref).

<source> is of the form <trace>[<addr>] where <trace> is an input trace, and <annotation>
is either :discrete or :continuous.
"""
macro read(src, ann::QuoteNode)
    return quote read($(esc(bij_state)), $(esc(src)), $(esc(typed(ann.value)))) end
end

"""
    @write(<destination>, <value>, <annotation>)

Macro for writing the value of a random choice to an output trace in the [Trace Transform DSL](@ref).

<destination> is of the form <trace>[<addr>] where <trace> is an input trace, and
<annotation> is either :discrete or :continuous.
"""
macro write(dest, val, ann::QuoteNode)
    return quote write($(esc(bij_state)), $(esc(dest)), $(esc(val)), $(esc(typed(ann.value)))) end
end

"""
    @copy(<source>, <destination>)

Macro for copying the value of a random choice (or a whole namespace of random choices)
from an input trace to an output trace in the [Trace Transform DSL](@ref).

<destination> is of the form <trace>[<addr>] where <trace> is an input trace,
and <annotation> is either :discrete or :continuous.
"""
macro copy(src, dest)
    return quote copy($(esc(bij_state)), $(esc(src)), $(esc(dest))) end
end

# TODO make more consistent by allowing us to read any hierarchical address,
# including return values of intermediate calls, not just the top-level call.

# TODO add haskey(model_in, addr), and haskey(aux_in, addr)

################################
# first pass through transform #
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
    "Trace containing the input model choice map ``t``"
    model_trace::Trace
    model_choices::ChoiceMap

    "The input proposal choice map ``u``"
    aux_trace::Union{Trace,Nothing}
    aux_choices::ChoiceMap

    results::FirstPassResults
end

FirstPassState(model_trace::Trace, aux_trace::Trace) =
    FirstPassState(model_trace, get_choices(model_trace),
                   aux_trace, get_choices(aux_trace), FirstPassResults())

FirstPassState(model_trace::Trace, aux_trace::Nothing) =
    FirstPassState(model_trace, get_choices(model_trace),
                   aux_trace, EmptyChoiceMap(), FirstPassResults())

function get_model_args(state::FirstPassState)
    return get_args(state.model_trace)
end

function get_aux_args(state::FirstPassState)
    return state.aux_trace === nothing ? () : get_args(state.aux_trace)
end

function run_first_pass(transform::TraceTransformDSLProgram, model_trace, aux_trace)
    state = FirstPassState(model_trace, aux_trace)
    transform.fn!(state) # TODO allow for other args to top-level transform function
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
    state.results.t_cont_reads[addr] = state.model_choices[addr]
    return state.model_trace[addr]
end

function read(state::FirstPassState, src::AuxInputAddress, ::DiscreteAnn)
    addr = src.addr
    return state.aux_trace[addr]
end

function read(state::FirstPassState, src::AuxInputAddress, ::ContinuousAnn)
    addr = src.addr
    state.results.u_cont_reads[addr] = state.aux_choices[addr]
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
# second pass through transform (gets automatically differentiated) #
#####################################################################

struct JacobianPassState{T<:Real}
    model_trace::Trace
    aux_trace::Union{Trace,Nothing}
    input_arr::AbstractArray{T}
    output_arr::Array{T}
    t_key_to_index::Dict
    u_key_to_index::Dict
    cont_constraints_key_to_index::Dict
    cont_u_back_key_to_index::Dict
end

function get_model_args(state::JacobianPassState)
    return get_args(state.model_trace)
end

function get_aux_args(state::JacobianPassState)
    return state.aux_trace === nothing ? () : get_args(state.aux_trace)
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

function assemble_input_array_and_maps(t_cont_reads, t_copy_reads,
                                       u_cont_reads, u_copy_reads,
                                       discard::Union{ChoiceMap,Nothing})
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
        next_output_index +=
            store_addr_info!(cont_constraints_key_to_index, addr, v, next_output_index)
    end

    cont_u_back_key_to_index = Dict()
    for (addr, v) in u_cont_writes
        next_output_index +=
            store_addr_info!(cont_u_back_key_to_index, addr, v, next_output_index)
    end

    return (cont_constraints_key_to_index, cont_u_back_key_to_index, next_output_index-1)
end

function jacobian_correction(transform::TraceTransformDSLProgram,
        prev_model_trace, proposal_trace, first_pass_results, discard)

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
    # transform, with inputs corresponding to a particular superset of the
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
        transform.fn!(jacobian_pass_state)

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

function check_round_trip(trace, trace_rt)
    choices = get_choices(trace)
    choices_rt = get_choices(trace_rt)
    if !isapprox(choices, choices_rt)
        @error("choices: $(sprint(show, "text/plain", choices))")
        @error("choices after round trip: $(sprint(show, "text/plain", choices_rt))")
        error("transform round trip check failed")
    end
    return nothing
end

function check_round_trip(
            prev_model_trace, prev_model_trace_rt,
            forward_proposal_trace, forward_proposal_trace_rt)
    check_round_trip(prev_model_trace, prev_model_trace_rt)
    check_round_trip(forward_proposal_trace, forward_proposal_trace_rt)
    return nothing
end

################################
# TraceTranslator #
################################

"Abstract type for trace translators."
abstract type TraceTranslator end

"""
    (new_trace, log_weight) = (translator::TraceTranslator)(trace)

Apply a trace translator on an input trace, returning a new trace and an incremental
log weight.
"""
(translator::TraceTranslator)(trace::Trace; kwargs...) = error("Not implemented.")

################################
# DeterministicTraceTranslator #
################################

"""
    translator = DeterministicTraceTranslator(;
        p_new::GenerativeFunction, p_args::Tuple=();
        new_observations::ChoiceMap=EmptyChoiceMap()
        f::TraceTransformDSLProgram)

Constructor for a deterministic trace translator.

Run the translator with:

    (output_trace, log_weight) = translator(input_trace)
"""
@with_kw mutable struct DeterministicTraceTranslator <: TraceTranslator
    p_new::GenerativeFunction
    p_args::Tuple = ()
    new_observations::ChoiceMap = EmptyChoiceMap()
    f::TraceTransformDSLProgram # a bijection
end

function inverse(translator::DeterministicTraceTranslator,
        prev_model_trace::Trace, prev_observations::ChoiceMap=EmptyChoiceMap())
    return DeterministicTraceTranslator(
        get_gen_fn(prev_model_trace), get_args(prev_model_trace),
        prev_observations, inverse(translator.f))
end

function run_transform(translator::DeterministicTraceTranslator,
                       prev_model_trace::Trace)
    @unpack p_new, p_args, new_observations, f = translator
    first_pass_results = run_first_pass(f, prev_model_trace, nothing)
    log_abs_determinant = jacobian_correction(
        f, prev_model_trace, nothing, first_pass_results, nothing)
    constraints = merge(first_pass_results.constraints, new_observations)
    (new_model_trace, _) = generate(p_new, p_args, constraints)
    return (new_model_trace, log_abs_determinant)
end

function (translator::DeterministicTraceTranslator)(
        prev_model_trace::Trace; check=false, prev_observations=EmptyChoiceMap())

    # apply trace transform
    (new_model_trace, log_abs_determinant) =
        run_transform(translator, prev_model_trace)

    # compute log weight
    prev_model_score = get_score(prev_model_trace)
    new_model_score = get_score(new_model_trace)
    log_weight = new_model_score - prev_model_score + log_abs_determinant

    if check
        check_observations(get_choices(new_model_trace),
                           translator.new_observations)
        inverter = inverse(translator, prev_model_trace)
        prev_model_trace_rt, _ = run_transform(inverter, new_model_trace)
        check_round_trip(prev_model_trace, prev_model_trace_rt)
    end

    return (new_model_trace, log_weight)
end

##########################
# GeneralTraceTranslator #
##########################

"""
    translator = GeneralTraceTranslator(;
        p_new::GenerativeFunction,
        p_new_args::Tuple = (),
        new_observations::ChoiceMap = EmptyChoiceMap(),
        q_forward::GenerativeFunction,
        q_forward_args::Tuple  = (),
        q_backward::GenerativeFunction,
        q_backward_args::Tuple  = (),
        f::TraceTransformDSLProgram)

Constructor for a general trace translator.

Run the translator with:

    (output_trace, log_weight) = translator(input_trace; check=false, prev_observations=EmptyChoiceMap())

Use `check` to enable a bijection check (this requires that the transform `f` has been
paired with its inverse using [`pair_bijections!](@ref) or [`is_involution`](@ref)).

If `check` is enabled, then `prev_observations` is a choice map containing the observed
random choices in the previous trace.
"""
@with_kw mutable struct GeneralTraceTranslator <: TraceTranslator
    p_new::GenerativeFunction
    p_new_args::Tuple = ()
    new_observations::ChoiceMap = EmptyChoiceMap()
    q_forward::GenerativeFunction
    q_forward_args::Tuple  = ()
    q_backward::GenerativeFunction
    q_backward_args::Tuple  = ()
    f::TraceTransformDSLProgram # a bijection
end

function inverse(translator::GeneralTraceTranslator, prev_model_trace::Trace,
                 prev_observations::ChoiceMap=EmptyChoiceMap())
    return GeneralTraceTranslator(
        get_gen_fn(prev_model_trace), get_args(prev_model_trace),
        prev_observations, translator.q_backward, translator.q_backward_args,
        translator.q_forward, translator.q_forward_args,
        inverse(translator.f))
end

function run_transform(translator::GeneralTraceTranslator,
                       prev_model_trace::Trace, forward_proposal_trace::Trace)
    @unpack f, new_observations = translator
    @unpack p_new, p_new_args, q_backward, q_backward_args = translator
    first_pass_results = run_first_pass(f, prev_model_trace, forward_proposal_trace)
    log_abs_determinant = jacobian_correction(
        f, prev_model_trace, forward_proposal_trace, first_pass_results, nothing)
    constraints = merge(first_pass_results.constraints, new_observations)
    (new_model_trace, _) = generate(p_new, p_new_args, constraints)
    backward_proposal_trace, = generate(
        q_backward, (new_model_trace, q_backward_args...), first_pass_results.u_back)
    return (new_model_trace, backward_proposal_trace, log_abs_determinant)
end

function (translator::GeneralTraceTranslator)(
        prev_model_trace::Trace; check=false, prev_observations=EmptyChoiceMap())

    # sample auxiliary trace
    forward_proposal_trace =
        simulate(translator.q_forward, (prev_model_trace, translator.q_forward_args...,))

    # apply trace transform
    (new_model_trace, backward_proposal_trace, log_abs_determinant) =
        run_transform(translator, prev_model_trace, forward_proposal_trace)

    # compute log weight
    prev_model_score = get_score(prev_model_trace)
    new_model_score = get_score(new_model_trace)
    forward_proposal_score = get_score(forward_proposal_trace)
    backward_proposal_score = get_score(backward_proposal_trace)
    log_weight = new_model_score - prev_model_score +
        backward_proposal_score + forward_proposal_score + log_abs_determinant

    if check
        inverter = inverse(translator, prev_model_trace, prev_observations)
        (prev_model_trace_rt, forward_proposal_trace_rt, _) =
            run_transform(inverter, new_model_trace, backward_proposal_trace)
        check_round_trip(prev_model_trace, prev_model_trace_rt,
                         forward_proposal_trace, forward_proposal_trace_rt)
    end

    return (new_model_trace, log_weight)
end

##################################
# SimpleExtendingTraceTranslator #
##################################

"""
    translator = SimpleExtendingTraceTranslator(;
        p_new_args::Tuple = (),
        p_argdiffs::Tuple = (),
        new_observations::ChoiceMap = EmptyChoiceMap(),
        q_forward::GenerativeFunction,
        q_forward_args::Tuple  = ())

Constructor for a simple extending trace translator.

Run the translator with:

    (output_trace, log_weight) = translator(input_trace)
"""
@with_kw mutable struct SimpleExtendingTraceTranslator <: TraceTranslator
    p_new_args::Tuple = ()
    p_argdiffs::Tuple = ()
    new_observations::ChoiceMap = EmptyChoiceMap()
    q_forward::GenerativeFunction
    q_forward_args::Tuple  = ()
end

function (translator::SimpleExtendingTraceTranslator)(prev_model_trace::Trace)

    # simulate from auxiliary program
    forward_proposal_trace =
        simulate(translator.q_forward, (prev_model_trace, translator.q_forward_args...,))
    forward_proposal_score = get_score(forward_proposal_trace)

    # computing the new trace via update
    constraints = merge(get_choices(forward_proposal_trace), translator.new_observations)
    (new_model_trace, log_model_weight, _, discard) = update(
        prev_model_trace, translator.p_new_args,
        translator.p_argdiffs, constraints)

    if !isempty(discard)
        @error("Can only extend the trace with random choices, not remove them.")
        error("Invalid SimpleExtendingTraceTranslator")
    end

    log_weight = log_model_weight - forward_proposal_score
    return (new_model_trace, log_weight)
end

############################
# SymmetricTraceTranslator #
############################

const TransformFunction = Union{TraceTransformDSLProgram,Function}

"""
    translator = SymmetricTraceTranslator(;
        q::GenerativeFunction,
        q_args::Tuple = (),
        involution::Union{TraceTransformDSLProgram,Function})

Constructor for a symmetric trace translator.

The involution is either constructed via the [`@transform`](@ref) macro (recommended),
or can be provided as a Julia function.

Run the translator with:

    (output_trace, log_weight) = translator(input_trace; check=false, observations=EmptyChoiceMap())

Use `check` to enable the involution check (this requires that the transform `f` has been
marked with [`is_involution`](@ref)).

If `check` is enabled, then `observations` is a choice map containing the observed random
choices, and the check will additionally ensure they are not mutated by the involution.
"""
@with_kw mutable struct SymmetricTraceTranslator{T <: TransformFunction} <: TraceTranslator
    q::GenerativeFunction
    q_args::Tuple = ()
    involution::T # an involution
end

function inverse(translator::SymmetricTraceTranslator, prev_model_trace=nothing)
    return translator
end

function run_transform(translator::SymmetricTraceTranslator,
                       prev_model_trace::Trace, forward_proposal_trace::Trace)
    @unpack involution, q, q_args = translator
    first_pass_results = run_first_pass(involution, prev_model_trace, forward_proposal_trace)
    (new_model_trace, log_model_weight, _, discard) = update(
        prev_model_trace, get_args(prev_model_trace),
        map((_) -> NoChange(), get_args(prev_model_trace)),
        first_pass_results.constraints)
    log_abs_determinant = jacobian_correction(
        involution, prev_model_trace, forward_proposal_trace, first_pass_results, discard)
    backward_proposal_trace, = generate(
        q, (new_model_trace, q_args...), first_pass_results.u_back)
    return (new_model_trace, backward_proposal_trace, log_abs_determinant)
end

function (translator::SymmetricTraceTranslator{TraceTransformDSLProgram})(
        prev_model_trace::Trace; check=false, observations=EmptyChoiceMap())

    # simulate from auxiliary program
    forward_proposal_trace =
        simulate(translator.q, (prev_model_trace, translator.q_args...,))

    # apply trace transform
    (new_model_trace, backward_proposal_trace, log_abs_determinant) =
        run_transform(translator, prev_model_trace, forward_proposal_trace)

    # compute log weight
    prev_model_score = get_score(prev_model_trace)
    new_model_score = get_score(new_model_trace)
    forward_proposal_score = get_score(forward_proposal_trace)
    backward_proposal_score = get_score(backward_proposal_trace)
    log_weight = new_model_score - prev_model_score +
        backward_proposal_score - forward_proposal_score + log_abs_determinant

    if check
        check_observations(get_choices(new_model_trace), observations)
        (prev_model_trace_rt, forward_proposal_trace_rt, _) =
            run_transform(translator, new_model_trace, backward_proposal_trace)
        check_round_trip(prev_model_trace, prev_model_trace_rt,
                         forward_proposal_trace, forward_proposal_trace_rt)
    end

    return (new_model_trace, log_weight)
end

function (translator::SymmetricTraceTranslator{<:Function})(
        prev_model_trace::Trace; check=false, observations=EmptyChoiceMap())

    forward_trace = simulate(translator.q, (prev_model_trace, translator.q_args...,))
    forward_score = get_score(forward_trace)
    forward_choices = get_choices(forward_trace)
    forward_retval = get_retval(forward_trace)
    (new_model_trace, backward_choices, log_weight) = translator.involution(
        prev_model_trace, forward_choices, forward_retval, translator.q_args)
    (backward_score, backward_retval) =
        assess(translator.q, (new_model_trace, translator.q_args...), backward_choices)

    log_weight += (backward_score - forward_score)

    if check
        check_observations(get_choices(new_model_trace), observations)
        (prev_model_trace_rt, forward_choices_rt, _) = translator.involution(
            new_model_trace, backward_choices, backward_retval, translator.q_args)
        (forward_trace_rt, _) = generate(
            translator.q, (prev_model_trace, translator.q_args...), forward_choices_rt)
        check_round_trip(prev_model_trace, prev_model_trace_rt,
                         forward_trace, forward_trace_rt)
    end

    return (new_model_trace, log_weight)
end


export @transform
export @read, @write, @copy, @tcall
export TraceTransformDSLProgram, pair_bijections!, is_involution!, inverse
export TraceTranslator, DeterministicTraceTranslator, SymmetricTraceTranslator,
       SimpleExtendingTraceTranslator, GeneralTraceTranslator
