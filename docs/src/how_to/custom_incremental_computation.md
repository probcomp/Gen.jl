# [Customizing Incremental Computation](@id custom_incremental_computation_howto)

Iterative inference techniques like Markov Chain Monte Carlo and Sequential Monte Carlo involve repeatedly updating the execution traces of generative models.

In some cases, the output of a deterministic computation within the model can be incrementally computed during each of these updates, instead of being computed from scratch.

To add a custom incremental computation for a deterministic computation, define a concrete subtype of [`CustomUpdateGF`](@ref) with the following methods:

- [`apply_with_state`](@ref)

- [`update_with_state`](@ref)

- [`has_argument_grads`](@ref)

The second type parameter of `CustomUpdateGF` is the type of the state that may be used internally to facilitate incremental computation within `update_with_state`.

For example, we can implement a function for computing the sum of a vector that efficiently computes the new sum when a small fraction of the vector elements change:

```julia
struct MyState
    prev_arr::Vector{Float64}
    sum::Float64
end

struct MySum <: CustomUpdateGF{Float64,MyState} end

function Gen.apply_with_state(::MySum, args)
    arr = args[1]
    s = sum(arr)
    state = MyState(arr, s)
    (s, state)
end

function Gen.update_with_state(::MySum, state, args, argdiffs::Tuple{VectorDiff})
    arr = args[1]
    prev_sum = state.sum
    retval = prev_sum
    for i in keys(argdiffs[1].updated)
        retval += (arr[i] - state.prev_arr[i])
    end
    prev_length = length(state.prev_arr)
    new_length = length(arr)
    for i=prev_length+1:new_length
        retval += arr[i]
    end
    for i=new_length+1:prev_length
        retval -= arr[i]
    end
    state = MyState(arr, retval)
    (state, retval, UnknownChange())
end

Gen.num_args(::MySum) = 1
```
