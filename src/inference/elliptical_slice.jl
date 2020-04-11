
"""
    new_trace = elliptical_slice(
        trace, addr, mu, cov;
        check=false, observations=EmptyChoiceMap())

Apply an elliptical slice sampling update to a given random choice with a multivariate normal prior.

Also takes the mean vector and covariance matrix of the prior.

[Reference URL](http://proceedings.mlr.press/v9/murray10a/murray10a.pdf)
"""
function elliptical_slice(
        trace, addr, mu, cov; check=false, observations=EmptyChoiceMap())

    # sample nu
    nu = mvnormal(zeros(length(mu)), cov)

    # sample u
    u = uniform(0, 1)

    # sample initial theta and define bracket
    theta = uniform(0, 2*pi)
    theta_min, theta_max = (theta - 2*pi, theta)

    # previous value
    f = trace[addr] .- mu

    new_f = f * cos(theta) + nu * sin(theta)
    new_trace, weight = update(trace; constraints=choicemap((addr, new_f .+ mu)))
    while weight <= log(u)
        if theta < 0
            theta_min = theta
        else
            theta_max = theta
        end
        theta = uniform(theta_min, theta_max)
        new_f = f * cos(theta) + nu * sin(theta)
        new_trace, weight = update(trace; constraints=choicemap((addr, new_f .+ mu)))
    end
    check && check_observations(get_choices(new_trace), observations)
    return new_trace
end

check_is_kernel(::typeof(elliptical_slice)) = true
is_custom_primitive_kernel(::typeof(elliptical_slice)) = false
reversal(::typeof(elliptical_slice)) = elliptical_slice

export elliptical_slice
