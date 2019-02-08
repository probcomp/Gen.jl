@gen (static) function datum(x::Float64, (grad)(inlier_std::Float64), (grad)(outlier_std::Float64),
                          (grad)(slope::Float64), (grad)(intercept::Float64))
    is_outlier::Bool = @trace(bernoulli(0.5), :z)
    std = is_outlier ? inlier_std : outlier_std
    y::Float64 = @trace(normal(x * slope + intercept, std), :y)
    return y
end

data = Map(datum)

function compute_argdiff(inlier_std_diff, outlier_std_diff, slope_diff, intercept_diff)
    if all([c == NoChoiceDiff() for c in [
            inlier_std_diff, outlier_std_diff, slope_diff, intercept_diff]])
        noargdiff
    else
        unknownargdiff
    end
end

@gen (static) function model(xs::Vector{Float64})
    n = length(xs)
    inlier_log_std::Float64 = @trace(normal(0, 2), :log_inlier_std)
    outlier_log_std::Float64 = @trace(normal(0, 2), :log_outlier_std)
    inlier_std = exp(inlier_log_std)
    outlier_std = exp(outlier_log_std)
    slope::Float64 = @trace(normal(0, 2), :slope)
    intercept::Float64 = @trace(normal(0, 2), :intercept)
    @diff inlier_std_diff = @choicediff(:log_inlier_std)
    @diff outlier_std_diff = @choicediff(:log_outlier_std)
    @diff slope_diff = @choicediff(:slope)
    @diff intercept_diff = @choicediff(:intercept)
    @diff argdiff = compute_argdiff(
            inlier_std_diff, outlier_std_diff, slope_diff, intercept_diff)
    @trace(data(xs, fill(inlier_std, n), fill(outlier_std, n),
               fill(slope, n), fill(intercept, n)),
          :data, argdiff)
end
