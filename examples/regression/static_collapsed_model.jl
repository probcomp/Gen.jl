import Distributions
import Gen: random, logpdf

struct TwoNormals <: SimpleGenerativeFunction{Float64} end
const two_normals = TwoNormals()

function logpdf(::TwoNormals, x, mu, sigma1, sigma2)
    if sigma1 < 0 || sigma2 < 0
        return -Inf
    end
    l1 = Distributions.logpdf(Distributions.Normal(mu, sigma1), x) + log(0.5)
    l2 = Distributions.logpdf(Distributions.Normal(mu, sigma2), x) + log(0.5)
    m = max(l1, l2)
    m + log(exp(l1 - m) + exp(l2 - m))
end

function random(::TwoNormals, mu, sigma1, sigma2)
    normal(mu, bernoulli(0.5) ? sigma1 : sigma2)
end

@gen (static) function generate_datum(mean::Float64, inlier_std::Float64, outlier_std::Float64)
    @trace(two_normals(mean, inlier_std, outlier_std), :z)
end

generate_data = Map(generate_datum)

@gen (static) function model(xs::Vector{Float64})
    n = length(xs)
    slope::Float64 = @trace(normal(0, 2), :slope)
    intercept::Float64 = @trace(normal(0, 2), :intercept)
    log_inlier_std::Float64 = @trace(normal(0, 2), :log_inlier_std)
    log_outlier_std::Float64 = @trace(normal(0, 2), :log_outlier_std)
    inlier_std = exp(log_inlier_std)
    outlier_std = exp(log_outlier_std)
    means = slope * xs .+ intercept
    ys = @trace(generate_data(means, fill(inlier_std, n), fill(outlier_std, n)), :data)
    return ys
end
