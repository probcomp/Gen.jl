import ForwardDiff

#function f(arr)
    #arr .* 2
#end
#
#J = ForwardDiff.jacobian(f, [1., 2., 3.])
#println(J)


#####

struct MyAbstractArray <: AbstractArray{Real,1}
    dict::Dict{Any,Real}
    index_to_key::Vector{Any}
    key_to_index::Dict{Any,Int}
end

MyAbstractArray() = MyAbstractArray(Dict{Any,Real}(), Vector{Any}(), Dict{Any,Int}())

Base.size(arr::MyAbstractArray) = length(arr.dict)
Base.getindex(arr::MyAbstractArray, i::Int) = arr.dict[arr.index_to_key[i]]

function record_value!(arr::MyAbstractArray, key::Any, value::Real)
    haskey(arr.dict, key) && error("already has key $key")
    index = length(arr.index_to_key) + 1
    push!(arr.index_to_key, key)
    arr.key_to_index[key] = index
    arr.dict[key] = value
end

function get_value(arr::MyAbstractArray, key::Any)
    arr.dict[key]
end

function f(arr::MyAbstractArray)
    z = get_value(arr, :x)
    w = get_value(arr, :y)
    new_arr = MyAbstractArray()
    record_value!(new_arr, :z, z)
    record_value!(new_arr, :w, w)
    new_arr
end

#### 

function f(t::Dict, u::Dict)

    cont_constraints = Dict()
    cont_u_back = Dict()

    arr = Vector{Float64}()
    next_input_index = 1
    next_output_index = 1
    
    t_key_to_index = Dict()
    u_key_to_index = Dict()
    cont_constraints_key_to_index = Dict()
    cont_u_back_key_to_index = Dict()

    # z = @read_from_trace(:x)
    begin
        z = t[:x]
        if !haskey(t_key_to_index, :x)
            t_key_to_index[:x] = next_input_index
            next_input_index += 1
            push!(arr, t[:x])
        end
    end

    # TODO distinguish between continuous addresses in the trace that need derivatives vs those that don't

    # w = @read_from_u(:y)
    begin
        w = u[:y]
        if !haskey(u_key_to_index, :y)
            u_key_to_index[:y] = next_input_index
            next_input_index += 1
            push!(arr, u[:y])
        end
    end

    # @write_to_cont_constraints(:z, 3 * z)
    begin
        cont_constraints[:z] = 3 * z
        cont_constraints_key_to_index[:z] = next_output_index
        next_output_index += 1
    end

    # @write_to_cont_u_back(:w, 2 * w)
    begin
        cont_u_back[:w] = 2 * w
        cont_u_back_key_to_index[:w] = next_output_index
        next_output_index += 1
    end

    function f_array(arr::AbstractArray{T}) where {T <: Real}

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

    cont_constraints, cont_u_back, arr, f_array
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



# our code should run, and return a function h_array of an array.
# the function should close over the get_index function.

function get_index(addr)
    if addr == :x
        1
    elseif addr == :y
        2
    elseif addr == :z
        1
    elseif addr == :w
        2
    end
end

function f(arr::AbstractArray{T,1}) where {T <: Real}
    z = arr[get_index(:x)]
    w = arr[get_index(:y)]
    new_arr = Array{Real,1}(undef, 2)
    new_arr[get_index(:z)] = z
    new_arr[get_index(:w)] = w
    new_arr
end

#arr = [1.2, 2.3] #MyAbstractArray()
#record_value!(arr, :x, 1.2)
#record_value!(arr, :y, -1.3)
#J = ForwardDiff.jacobian(f, arr)
#println(J)

