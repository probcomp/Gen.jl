
"""
    new_trace = elliptical_slice(trace, addr, mu, cov)

Apply an elliptical slice sampling update to a given random choice with a multivariate normal prior.

Also takes the mean vector and covariance matrix of the prior.

[Reference URL](http://proceedings.mlr.press/v9/murray10a/murray10a.pdf)
"""
function elliptical_slice(trace, addr, mu, cov)
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)

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
    new_trace, weight = update(trace, args, argdiffs, choicemap((addr, new_f .+ mu)))
    while weight <= log(u)
        if theta < 0
            theta_min = theta
        else
            theta_max = theta
        end
        theta = uniform(theta_min, theta_max)
        new_f = f * cos(theta) + nu * sin(theta)
        new_trace, weight = update(trace, args, argdiffs, choicemap((addr, new_f .+ mu)))
    end
    return new_trace
end

export elliptical_slice
