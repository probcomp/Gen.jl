using Gen
import Random

##############################################################
# variational inference with multi-sample gradient estimator #
##############################################################

Random.seed!(1)

@gen function model()
    slope = @trace(normal(-1, exp(0.5)), :slope)
    intercept = @trace(normal(1, exp(2.0)), :intercept)
end

@gen function approx()
    @param slope_mu::Float64
    @param slope_log_std::Float64
    @param intercept_mu::Float64
    @param intercept_log_std::Float64
    @trace(normal(slope_mu, exp(slope_log_std)), :slope)
    @trace(normal(intercept_mu, exp(intercept_log_std)), :intercept)
end
init_param!(approx, :slope_mu, 0.)
init_param!(approx, :slope_log_std, 0.)
init_param!(approx, :intercept_mu, 0.)
init_param!(approx, :intercept_log_std, 0.)

observations = choicemap()
update = ParamUpdate(GradientDescent(1., 1000), approx)
black_box_vi!(model, (), observations, approx, (), update, 20;
    iters=10000, samples_per_iter=100, verbose=true, geometric=false)
slope_mu = get_param(approx, :slope_mu)
slope_log_std = get_param(approx, :slope_log_std)
intercept_mu = get_param(approx, :intercept_mu)
intercept_log_std = get_param(approx, :intercept_log_std)
println((slope_mu, slope_log_std, intercept_mu, intercept_log_std))
@assert isapprox(slope_mu, -1., atol=0.01)
@assert isapprox(slope_log_std, 0.5, atol=0.01)
@assert isapprox(intercept_mu, 1., atol=0.01)
@assert isapprox(intercept_log_std, 2.0, atol=0.01)

#######################################################
# variational inference with score function estimator #
#######################################################

Random.seed!(1)

@gen function model()
    slope = @trace(normal(-1, exp(0.5)), :slope)
    intercept = @trace(normal(1, exp(2.0)), :intercept)
end

@gen function approx()
    @param slope_mu::Float64
    @param slope_log_std::Float64
    @param intercept_mu::Float64
    @param intercept_log_std::Float64
    @trace(normal(slope_mu, exp(slope_log_std)), :slope)
    @trace(normal(intercept_mu, exp(intercept_log_std)), :intercept)
end
init_param!(approx, :slope_mu, 0.)
init_param!(approx, :slope_log_std, 0.)
init_param!(approx, :intercept_mu, 0.)
init_param!(approx, :intercept_log_std, 0.)

observations = choicemap()
update = ParamUpdate(GradientDescent(1., 500), approx)
black_box_vi!(model, (), observations, approx, (), update;
    iters=500, samples_per_iter=100, verbose=false)
slope_mu = get_param(approx, :slope_mu)
slope_log_std = get_param(approx, :slope_log_std)
intercept_mu = get_param(approx, :intercept_mu)
intercept_log_std = get_param(approx, :intercept_log_std)
println((slope_mu, slope_log_std, intercept_mu, intercept_log_std))
@assert isapprox(slope_mu, -1., atol=0.01)
@assert isapprox(slope_log_std, 0.5, atol=0.01)
@assert isapprox(intercept_mu, 1., atol=0.01)
@assert isapprox(intercept_log_std, 2.0, atol=0.01)
