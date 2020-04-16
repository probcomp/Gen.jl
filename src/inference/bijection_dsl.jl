import ForwardDiff
import MacroTools
import LinearAlgebra

struct BijectionDSLProgram
    fn!::Function
end

# See this math writeup for an understanding of how this code works:
# docs/tex/mcmc.pdf

struct FirstPassState

    "trace containing the input model choice map ``t``"
    trace

    "the input proposal choice map ``u``"
    u::ChoiceMap

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

function FirstPassState(trace, u::ChoiceMap)
    FirstPassState(
        trace, u, choicemap(), choicemap(),
        Dict(), Dict(), Dict(), Dict(),
        DynamicSelection(), DynamicSelection())
end

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

const inv_state = gensym("inv_state")

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

    quote

    # mutates the state
    function $fn!($(esc(inv_state))::Union{FirstPassState,JacobianPassState}, $(map(esc, args)...))
        $(esc(body))
        nothing
    end

    Core.@__doc__ $(esc(f)) = BijectionDSLProgram($fn!)

    end # quote

end # macro bijection()

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
    if haskey(state.u_key_to_index, addr)
        state.input_arr[state.u_key_to_index[addr]]
    else
        state.u[addr]
    end
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
    if haskey(state.t_key_to_index, addr)
        state.input_arr[state.t_key_to_index[addr]]
    else
        state.trace[addr]
    end
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

# call another bijection function

macro bijcall(ex)
    MacroTools.@capture(ex, f_(args__)) || error("expected syntax: f(..)")
    quote $(esc(f)).fn!($(esc(inv_state)), $(map(esc, args)...)) end
end

# apply 

discard_skip_read_addr(addr, discard::ChoiceMap) = !has_value(discard, addr)
discard_skip_read_addr(addr, discard::Nothing) = false

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
            # only used by involutions with update
            continue
        end

        t_key_to_index[addr] = next_input_index
        next_input_index += 1
        push!(input_arr, v)
    end

    u_key_to_index = Dict()
    for (addr, v) in u_cont_reads
        if addr in u_copy_reads
            continue
        end
        u_key_to_index[addr] = next_input_index
        next_input_index += 1
        push!(input_arr, v)
    end

    (t_key_to_index, u_key_to_index, input_arr)
end

function assemble_output_maps(t_cont_writes, u_cont_writes)
    next_output_index = 1

    cont_constraints_key_to_index = Dict()
    for (addr, v) in t_cont_writes
        cont_constraints_key_to_index[addr] = next_output_index
        next_output_index += 1
    end

    cont_u_back_key_to_index = Dict()
    for (addr, v) in u_cont_writes
        cont_u_back_key_to_index[addr] = next_output_index
        next_output_index += 1
    end

    (cont_constraints_key_to_index, cont_u_back_key_to_index)
end

function apply_bijection(
        bijection::BijectionDSLProgram, prev_model_trace, proposal_trace,
        new_model::Union{GenerativeFunction,Nothing}, new_model_args::Union{Tuple,Nothing},
        new_observations::Union{ChoiceMap,Nothing})

    # take the first pass through the bijection, mutating first_pass_state
    first_pass_state = FirstPassState(prev_model_trace, get_choices(proposal_trace))
    bijection.fn!(first_pass_state, get_args(prev_model_trace), get_args(proposal_trace), get_retval(proposal_trace))

    if isnothing(new_model)
        @assert isnothing(new_model_args)
        @assert isnothing(new_observations)

        # involution and use update to incrementally obtain new trace
        (new_model_trace, model_weight, _, discard) = update(
            prev_model_trace,
            get_args(prev_model_trace),
            map((_) -> NoChange(), get_args(prev_model_trace)),
            first_pass_state.constraints)

    else
        @assert !isnothing(new_model_args)
        @assert !isnothing(new_observations)

        # get score for previous model trace
        prev_model_score = get_score(prev_model_trace)

        # obtain new model trace
        new_model_trace, new_model_score = generate(
            new_model, new_model_args,
            merge(first_pass_state.constraints, new_observations))

        # model weight
        model_weight = new_model_score - prev_model_score

        discard = nothing
    end

    # create input array and mappings input addresses that are needed for Jacobian
    # exclude addresses that were copied explicitly to another address
    (t_key_to_index, u_key_to_index, input_arr) = assemble_input_array_and_maps(
        first_pass_state.t_cont_reads,
        first_pass_state.t_copy_reads,
        first_pass_state.u_cont_reads,
        first_pass_state.u_copy_reads, discard)
    
    # create mappings for output addresses that are needed for Jacobian
    (cont_constraints_key_to_index, cont_u_back_key_to_index) = assemble_output_maps(
        first_pass_state.t_cont_writes,
        first_pass_state.u_cont_writes)

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
            jacobian_pass_state, get_args(prev_model_trace), get_args(proposal_trace), get_retval(proposal_trace))

        # return the output array
        output_arr
    end

    (new_model_trace, model_weight, first_pass_state.u_back, input_arr, f_array)
end

# callable

"""
    (bijection::BijectionDSLProgram)(
        prev_model_trace::Trace, proposal_trace::Trace,
        new_model::GenerativeFunction, new_model_args::Tuple,
        new_constraints::ChoiceMap)
"""
function (bijection::BijectionDSLProgram)(
        prev_model_trace::Trace, proposal_trace::Trace,
        new_model=nothing, new_model_args=nothing,
        new_observations=nothing; check=false)

    # apply the bijection, and get metadata back about it, including the
    # function f_array from arrays to arrays, that is the restriction of the
    # bijection to just a subset of the relevant continuous random choices
    (new_model_trace, model_weight, u_back, input_arr, f_array) = apply_bijection(
        bijection, prev_model_trace, proposal_trace, new_model, new_model_args, new_observations)

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

    (new_model_trace, u_back, model_weight + correction)
end

export @bijection
export @read_discrete_from_proposal, @read_discrete_from_model
export @write_discrete_to_proposal, @write_discrete_to_model
export @read_continuous_from_proposal, @read_continuous_from_model
export @write_continuous_to_proposal, @write_continuous_to_model
export @copy_model_to_model, @copy_model_to_proposal
export @copy_proposal_to_proposal, @copy_proposal_to_model
export @bijcall
