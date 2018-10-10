using FunctionalCollections: PersistentVector
using LinearAlgebra: eye, chol, dot
import Gen
using Distributions: MvNormal

#######################
# Univariate GP trace #
#######################

struct UnivariateGPTrace
    mean_function::Function
    cov_function::Function
    cov_cholesky::Matrix{Float64} # chol(K + noise * I)
    noise::Float64
    inputs::Vector{Float64}
    outputs::Vector{Float64}
    score::Float64
end

function Gen.get_call_record(trace::UnivariateGPTrace)
    args = (trace.mean_function, trace.cov_function, trace.noise, trace.inputs)
    CallRecord(trace.score, trace.outputs, args)
end

Gen.has_choices(trace::UnivariateGPTrace) = length(trace.inputs) > 1

function Gen.get_assignment(trace::UnivariateGPTrace)
    assignment = DynamicAssignment()
    for (i, output) in enumerate(trace.outputs)
        assignment[i] = output
    end
    assignment
end

########################
# Univariate GP module #
########################

struct UnivariateGP <: Generator{Vector{Float64}, UnivariateGPTrace}
end

const gaussian_process = UnivariateGP()

struct GPArgChangeInfo
    gp_changed::Bool
    inputs_changed::Bool
end

function compute_cov_matrix(cov_function::Function, inputs::Vector{Float64}, noise)
    num_inputs = length(inputs)
    cov_matrix = Matrix{Float64}(undef, num_inputs, num_inputs)
    for i=1:num_inputs
        for j=1:num_inputs
            cov_matrix[i, j] = cov_function(inputs[i], inputs[j])
        end
    end
    cov_matrix += noise * eye(num_inputs)
    @assert isposdef(cov_matrix)
    return cov_matrix
end

function compute_log_marg_lik(outputs, alpha, L::AbstractMatrix{Float64})
    num_inputs = length(outputs)
    log_marg_lik = -0.5 * dot(outputs, alpha)
    for i=1:num_inputs
        log_marg_lik -= log(L[i, i])
    end
    log_marg_lik -= (num_inputs/2.) * log(2 * pi)
    return log_marg_lik
end

function Gen.simulate(::UnivariateGP, args::Tuple)
    (mean_function::Function, cov_function::Function, noise, inputs) = args
    num_inputs = length(inputs)

    # compute mean vector and covariance matrix
    means = map(mean_function, inputs)
    cov_matrix = compute_cov_matrix(cov_function, inputs, noise)
    L = chol(cov_matrix)
    
    # sample outputs
    outputs = rand(MvNormal(means, cov_matrix))
    alpha = L' \ (L \ outputs)

    # compute the score, which is also the weight
    score = compute_log_marg_lik(outputs, alpha, L)
    
    trace = UnivariateGPTrace(mean_function, cov_function, L,
                              noise, inputs, outputs, score)
    return trace
end


function Gen.generate(::UnivariateGP, args::Tuple, constraints::Assignment)
    (mean_function::Function, cov_function::Function, noise, inputs) = args
    num_inputs = length(inputs)

    # extract outputs from constraints. currently, we assume that all inputs
    # have corresponding outputs in constraints
    outputs = Float64[constraints[i] for i=1:num_inputs]
    
    # compute mean vector and covariance matrix
    means = map(mean_function, inputs)
    cov_matrix = compute_cov_matrix(cov_function, inputs, noise)
    L = chol(cov_matrix)
    alpha = L' \ (L \ outputs)

    # compute the score, which is also the weight
    score = compute_log_marg_lik(outputs, alpha, L)
    weight = score
    
    trace = UnivariateGPTrace(mean_function, cov_function, L,
                              noise, inputs, outputs, score)
    return (trace, weight)
end

function Gen.fix_update(::UnivariateGP, args, arg_change::GPArgChangeInfo,
                    trace::UnivariateGPTrace, constraints::Assignment)
    (mean_function::Function, cov_function::Function, noise, inputs) = args 
    
    if arg_change.gp_changed
        error("Not implemented")
    end

    if !isempty(constraints)
        error("Not implemented")
    end

    # get previous inputs and outputs
    prev_args = get_call_record(trace).args
    (_, _, prev_inputs, _) = prev_args
    prev_num_inputs = length(prev_inputs)
    num_inputs = length(inputs)
    prev_outputs = trace.outputs

    # new test inputs
    added_inputs = inputs[prev_num_inputs+1:num_inputs]

    # TODO compute distribution on new outputs, update L, etc.
    
    weight = 0.
    discard = EmptyAssignment()
    retchange = nothing
    new_trace = UnivariateGPTrace(mean_function, cov_function, L,
                                  inputs, outputs)
    return (new_trace, weight, discard, retchange)
end

# TODO for MCMC over the GP..

function Gen.update(::UnivariateGP, args, change::GPArgChangeInfo,
                trace::UnivariateGPTrace, constraints::Assignment)
    (mean_function::Function, cov_function::Function, noise, inputs) = args
    num_inputs = length(inputs)
    
    if arg_change.inputs_changed
        error("Not implemented")
    end

    if !isempty(constraints)
        error("Not implemented")
    end

    # gp may have changed. compute new mean vector and covariance matrix.
    means = map(mean_function, inputs)
    cov_matrix = compute_cov_matrix(cov_function, inputs, noise)
    L = chol(cov_matrix)
    alpha = L' \ (L \ outputs)

    # compute new score
    score = compute_log_marg_lik(outputs, alpha, L)

    discard = EmptyAssignment()
    weight = score - get_call_record(trace).score
    new_trace = UnivariateGPTrace(mean_function, cov_function, L,
                                  inputs, outputs, score)
    return (new_trace, weight, discard, retchange)
end

function Gen.backprop_trace(::UnivariateGP, trace::UnivariateGPTrace,
                        selection::AddressSet, retval_grad)

    # TODO to backpropagate to the mean and covariance functions we will need
    # some representation of the gradient for these data types?

    error("Not implemented")
end

export gaussian_process
