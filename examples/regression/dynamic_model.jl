#########
# model #
#########

# TODO put this into FunctionalCollections:
import FunctionalCollections
Base.IndexStyle(::Type{<:FunctionalCollections.PersistentVector}) = IndexLinear()

@gen function datum(x::Float64, (grad)(inlier_std::Float64), (grad)(outlier_std), (grad)(slope), (grad)(intercept))::Float64
    is_outlier = @trace(bernoulli(0.5), :z)
    std = is_outlier ? inlier_std : outlier_std
    y = @trace(normal(x * slope + intercept, std), :y)
    return y
end

data = Map(datum)

@gen function model(xs::Vector{Float64})
    n = length(xs)
    inlier_std = exp(@trace(normal(0, 2), :log_inlier_std))
    outlier_std = exp(@trace(normal(0, 2), :log_outlier_std))
    slope = @trace(normal(0, 2), :slope)
    intercept = @trace(normal(0, 2), :intercept)
    ys = @trace(data(xs, fill(inlier_std, n), fill(outlier_std, n),
               fill(slope, n), fill(intercept, n)), :data)
    return ys
end
