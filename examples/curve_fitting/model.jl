@gen function generate_datum(x::Float64, (grad)(coeffs::Vector{Float64}),
                             prob_outlier::Float64, noise::Float64)
    
    # use a heuristic to change the noise
    if @trace(bernoulli(prob_outlier), :is_outlier)
        (mu, sigma) = (0., 10.)
    else
        y_mean = coeffs' * [x^i for i=0:length(coeffs)-1]
        (mu, sigma) = (y_mean, noise)
    end
    @trace(normal(mu, sigma), :y)
end

# prior over degree of polynomial
const degree_prior = [0.25, 0.25, 0.25, 0.25]
#const degree_prior = [1.00]

@gen function model(xs::Vector{Float64})

    # generate degree (either 1, 2, 3, or 4)
    degree = @trace(categorical(degree_prior), :degree)

    # generate coefficients
    coeffs = [@trace(normal(0, 1), (:c, i)) for i=1:degree+1]
        
    # other parameters
    prob_outlier = 0.1
    noise = @trace(gamma(2, 2), :noise)

    # generate data
    ys = Vector{Float64}(undef, length(xs))
    for (i, x) in enumerate(xs)
        ys[i] = @trace(generate_datum(x, coeffs, prob_outlier, noise), i)
    end

    return ys
end

