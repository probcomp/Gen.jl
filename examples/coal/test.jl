import ForwardDiff
import MacroTools
using Parameters: @with_kw
import LinearAlgebra

# TODO do we allow involution functions to call other involution functions (and recursion?)

# no -- we will allow Julia recursion within the function body, closing over the state variables.
# that should suffice?

# TODO what about moving entire subtrees? -- that should be fine, all the addresses underneath will not be involved in Jacobina.

# each involution function could have a discrete and a continuous portion?
# but the things that the involution function calls may not be involutions, so it should not be called @inv function

# hierarchical trace addresses etc.

# 

@with_kw struct FwdInvState
    trace
    u
    constraints = choicemap()
    u_back = choicemap()
    arr = Vector{Float64}()
    next_input_index = 1
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
    quote read_from_propoosal($(esc(inv_state)), addr) end
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
    quote read_from_model($(esc(inv_state)), addr) end
end

function read_from_model(state::FwdInvState, addr)
    state.t_cont_reads[addr] = state.t[addr]
    state.t[addr]
end

function read_from_model(state::JacInvState, addr)
    state.input_arr[state.t_key_to_index[addr]]
end

# read from model retained

# * it will not contribute to the Jacobian *
# TODO add an optional dynamic check for this -- it should not appear in the discard from update

macro read_from_model_retained(addr)
    quote read_from_model_retained($(esc(inv_state)), addr) end
end

function read_from_model_retained(state::FwdInvState, addr)
    push!(state.marked_as_retained, addr)
    state.t[addr]
end

function read_from_model_retained(state::JacInvState, addr)
    state.t[addr] # read directly from the trace, instead of the array
end

# write_to_proposal

macro write_to_proposal(addr, value)
    quote write_to_proposal($(esc(inv_state)), addr) end
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
    quote write_to_model($(esc(inv_state)), addr) end
end

function write_to_model(state::FwdInvState, addr, value)
    state.constraints[addr] = value
    state.cont_constraints_key_to_index[addr] = state.next_output_index
    state.next_output_index += 1
    value
end

function write_to_model(state::JacInvState, addr, value)
    state.output_arr[state.cont_u_back_key_to_index[addr]] = value
end

# move_model

macro move_model(from_addr, to_addr)
    quote move_model($(esc(inv_state)), from_addr, to_addr) end
end

function move_model(state::FwdInvState, from_addr, to_addr)
    state.constraints[to_addr] = state.t[from_addr]
    push!(state.t_move_reads, from_addr)
    state.t[from_addr]
end

function move_model(state::JacInvState, from_addr, to_addr)
    nothing
end



macro involution(ex)
    MacroTools.@capture(ex, function f_(args__) body_ end) || error("expected syntax: function f(..) .. end")
    MacroTools.@capture(body, (@discrete begin discrete_body_ end; @continuous begin continuous_body_ end))

    quote

    function $f(trace, u)

        constraints = choicemap()
        u_back = choicemap()
    
        $discrete_body
    
        state = InvState(trace=trace, u=u, constraints=constraints, u_back=u_back)
    
        $(esc(inv_state)) = state
        $continuous_body
    
        # add addresses read from t to arr
        for (addr, v) in state.t_cont_reads
            if !(addr in state.t_move_reads) # exclude addresses that were moved to another address
                state.t_key_to_index[addr] = state.next_input_index
                state.next_input_index += 1
                push!(state.arr, v)
            end
        end
    
        function f_array(input_arr::AbstractArray{T}) where {T <: Real}
            $(esc(inv_state)) = InvJacState(state, input_arr)
            $continuous_body
            $(esc(inv_state)).output_arr
        end
    
        return (constraints, u_back, state.arr, f_array, state.t_key_to_index, state.marked_as_retained)
    end

    end # quote

end # macro involution()

function rjmcmc(trace, q, q_args, f)

    # run proposal
    u, q_fwd_score, = propose(q, (trace, q_args...))

    # run involution
    (constraints, u_back, arr, f_array, t_key_to_indexm, marked_as_retained) = f(trace, u)

    # update model trace
    (new_trace, model_weight, _, discard) = update(
        trace, get_args(trace), map((_) -> NoChange(), get_args(trace)), constraints)

    # check the user's retained assertions (TODO disable this check in fast mode)
    for addr in marked_as_retained
        has_value(discard, addr) && error("addr $addr was marked as retained, but was not")
    end

    # Jacobian of involution
    # columns are inputs, rows are outputs
    J = ForwardDiff.Jacobian(h_array, arr)
    @assert size(J)[2] == length(arr)
    num_outputs = size(J)[1]
    
    # remove columns for inputs from the trace that were retained
    keep = fill(true, length(arr))
    for (addr, index) in t_key_to_index
        if !has_value(constraints, discard)
            keep[index] = false
        end
    end
    @assert sum(keep) == num_outputs # must be square
    J = J[:,keep]

    # compute correction
    correction = LinearAlgebra.logabsdet(J)

    # compute proposal backward score
    (q_bwd_score, _) = assess(q, (new_trace, q_args...), u_back)

    # accept or reject
    alpha = weight - q_fwd_score + q_bwd_score - correction # TODO check sign on correction
    if log(rand()) < alpha
        # accept
        return (new_trace, true)
    else
        # reject
        return (trace, false)
    end
end


expr = MacroTools.prewalk(MacroTools.rmlines, macroexpand(Main, :(@involution function f(a, b)
    @discrete begin
        something = 2
    end

    @continuous begin
        x = 3
    end
end)))

println(expr)

exit()


# TODO what about the support requirement?

# we could have two DSLs, discrete and continuous
# discrete would be regular Julia functions (recursion) and would run first?
# continuous
# or allow involution code to call itself recursively, and pass the state?

@involution function f(model_args, randomness_args)

    @discrete begin

        T = model_args[1]
    
        # current number of changepoints
        k = @read_discrete_from_trace(K)
        
        # if k == 0, then we can only do a birth move
        isbirth = (k == 0) || @read_discrete_from_u(IS_BIRTH)
    
        # change k
        @write_discrete_to_constraints(K, isbirth ? k + 1 : k - 1)
    
        # the changepoint to be added or deleted
        i = @read_discrete_from_u(CHOSEN)
        @write_discrete_to_u_back(CHOSEN, i)
    
        if k > 1 || isbirth
            @write_discrete_to_u_back(IS_BIRTH, !isbirth)
        end
    end

    @continuous begin

        if isbirth
    
            cp_new = @read_cont_from_u(NEW_CHANGEPT)
            cp_prev = (i == 1) ? 0. : @read_cont_from_trace((CHANGEPT, i-1))
            cp_next = (i == k+1) ? T : @read_cont_from_trace((CHANGEPT, i))
    
            # set new changepoint
            @write_to_cont_constraints((CHANGEPT, i), cp_new)
    
            # shift up changepoints
            for j=i+1:k+1
                @read_from_trace_and_move_cont((CHANGEPT, j-1), (CHANGEPT, j))
            end
    
            # compute new rates
            h_cur = @read_cont_from_trace_not_preserved((RATE, i))
            u = @read_cont_from_u(U)
            (h_prev, h_next) = new_rates([h_cur, u, cp_new, cp_prev, cp_next])
    
            # set new rates
            @write_to_cont_constraints((RATE, i), h_prev)
            @write_to_cont_constraints((RATE, i+1), h_next)
    
            # shift up rates
            for j=i+2:k+2
                @read_from_trace_and_move_cont((RATE, j-1), (RATE, j))
            end
        else
    
            cp_deleted = @read_cont_from_trace_not_preserved((CHANGEPT, i))
            cp_prev = (i == 1) ? 0. : @read_cont_from_trace((CHANGEPT, i-1))
            cp_next = (i == k) ? T : @read_cont_from_trace((CHANGEPT, i+1))
            @write_to_u_cont_back(NEW_CHANGEPT, cp_deleted)
    
            # shift down changepoints
            for j=i:k-1
                @read_from_trace_and_move_cont((CHANGEPT, j+1), (CHANGEPT, j))
            end
    
            # compute cur rate and u
            h_prev = trace[(RATE, i)]
            h_next = trace[(RATE, i+1)]
            (h_cur, u) = new_rates_inverse([h_prev, h_next, cp_deleted, cp_prev, cp_next])
            J = jacobian(new_rates_inverse, [h_prev, h_next, cp_deleted, cp_prev, cp_next])[:,1:2]
            bwd_choices[U] = u
    
            # set cur rate
            constraints[(RATE, i)] = h_cur
    
            # shift down rates
            for j=i+1:k
                constraints[(RATE, j)] = trace[(RATE, j+1)]
            end
        end
    
    end
end

function f(t::Dict, u::Dict, args)

    # begin generated prelude

    disc_constraints = Dict()
    disc_u_back = Dict()

    cont_constraints = Dict()
    cont_u_back = Dict()

    # input array
    arr = Vector{Float64}()
    next_input_index = 1
    u_key_to_index = Dict()
    t_cont_reads = Dict()
    t_move_reads = Set()

    # output array
    next_output_index = 1
    cont_constraints_key_to_index = Dict()
    cont_u_back_key_to_index = Dict()

    # end generated prelude

    # TODO maybe we should assume an address is going to be kept and left unchanged unless either:
    # 1. it is constrained (i.e. overwritten)
    # 2. it is deleted (the user needs to annotate this for us?) TODO -- actually now, we can figure this out later?
    # by looking at the discard?

    # NOTE: without the deleted annotation, we will 
    # 

    # z = @read_from_trace_preserved(:x) # the address is going to be kept and left unchanged
    # * it will not contribute to the Jacobian *
    # (TODO we can add an optional dynamic check for this -- it should not appear in the discard from update)
    z = begin
        t[:x]
    end

    # z = @read_from_trace_not_preserved(:x) # the address is going to be deleted or changed 
    # (TODO we can add an optional dynamic check for this -- it should appear in the discard from update)
    # * it will contribute to the Jacobian, unless its value was moved to another address *
    z = begin
        t_cont_reads[:x] = t[:x]
        t[:x]
    end

    # value = @read_from_trace_and_move_cont(addr1, addr2)
    # * it will not contribute to the Jacobian *
    value = begin
        cont_constraints[addr2] = t[addr1]
        push!(t_move_reads, addr1)
        t[addr1]
    end

    # w = @read_from_u(:y)
    # * it will contribute to the Jacobian *
    w = begin
        if !haskey(u_key_to_index, :y)
            u_key_to_index[:y] = next_input_index
            next_input_index += 1
            push!(arr, u[:y])
        end
        u[:y]
    end

    # @write_to_cont_constraints(:z, 3 * z)
    # * will contribute to the Jacobian *
    begin
        # TODO check that it wasn't the target of a move, and check that it is not read using read_from_trace_preserved
        cont_constraints[:z] = 3 * z
        cont_constraints_key_to_index[:z] = next_output_index
        next_output_index += 1
    end

    # @write_to_cont_u_back(:w, 2 * w)
    # * will contribute to the Jacobian *
    begin
        cont_u_back[:w] = 2 * w
        cont_u_back_key_to_index[:w] = next_output_index
        next_output_index += 1
    end

    # TODO we would like to only have to 

    # now add addresses read from t to arr
    t_key_to_index = Dict()
    for (addr, v) in t_cont_reads
        if !(addr in t_move_reads) # exclude addresses that were moved to another address
            t_key_to_index[addr] = next_input_index
            next_input_index += 1
            push!(arr, v)
        end
    end

    function f_array(arr::AbstractArray{T}) where {T <: Real}

        # addresses in u are read from the array
        # addresses in t that are moved are read from t, not the array
        # addresses in t that are not moved, and not marked as 'ignorable' are read from the array
        # addresses in t that are not moved, and marked as 'ignorable' are read from t

        # the user can mark an address as ignorable if they know that it is retained
        # (maybe we should make a 'retained' annotation)

        # afterwards, we will need to subset the rows (input) of the Jacobian to exclude those that
        # were retained.. that is, we only include an input row if it was
        # either deleted or constrained, which we know after update has returned.

        # z = @read_from_trace(:x)
        z = arr[t_key_to_index[:x]]

        # z = @read_from_u(:y)
        w = arr[u_key_to_index[:y]]

        new_arr = Array{T,1}(undef, next_output_index-1)

        # @write_to_cont_constraints(:z, 3 * z)
        new_arr[cont_constraints_key_to_index[:z]] = 3 * z

        # @write_to_cont_u_back(:w, 2 * w)
        new_arr[cont_u_back_key_to_index[:w]] = 2 * w

        new_arr
    end

    constraints, u_back, arr, f_array
    #disc_constraints, disc_u_back, cont_constraints, cont_u_back, arr, f_array
end

t = Dict()
t[:x] = 1.2
u = Dict()
u[:y] = 2.3
cont_constraints, cont_u_back, arr, f_array = f(t, u)
println(cont_constraints)
println(cont_u_back)
@time J = ForwardDiff.jacobian(f_array, arr)
@time J = ForwardDiff.jacobian(f_array, arr)
@time J = ForwardDiff.jacobian(f_array, arr)
println(J)

for i=1:10
    t = Dict()
    t[:x] = rand()
    u = Dict()
    u[:y] = rand()
    cont_constraints, cont_u_back, arr, f_array = f(t, u)
    @time J = ForwardDiff.jacobian(f_array, arr)
    println(J)
end


