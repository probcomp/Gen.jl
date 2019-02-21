using Gen: ifelse

@gen (static) function datum(x::Float64, (grad)(inlier_std::Float64), (grad)(outlier_std::Float64),
                             (grad)(slope::Float64), (grad)(intercept::Float64))
    is_outlier = @trace(bernoulli(0.5), :z)
    std = ifelse(is_outlier, inlier_std, outlier_std)
    y = @trace(normal(x * slope + intercept, std), :y)
    return y
end

data = Map(datum)

@gen (static) function model(xs::Vector{Float64})
    n = length(xs)
    inlier_log_std = @trace(normal(0, 2), :log_inlier_std)
    outlier_log_std = @trace(normal(0, 2), :log_outlier_std)
    inlier_std = exp(inlier_log_std)
    outlier_std = exp(outlier_log_std)
    slope = @trace(normal(0, 2), :slope)
    intercept = @trace(normal(0, 2), :intercept)
    @trace(data(xs, fill(inlier_std, n), fill(outlier_std, n),
        fill(slope, n), fill(intercept, n)), :data)
end
