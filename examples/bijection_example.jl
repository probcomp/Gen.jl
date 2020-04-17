using Gen

@gen function model1()
    x ~ normal(0, 1)
    y ~ normal(0, 1)
    obsx ~ normal(x, 0.1)
    obsy ~ normal(y, 0.1)
end

@gen function model2()
    r ~ gamma(1, 1)
    theta ~ uniform(-pi, pi)
    obsr ~ normal(r, 1.)
end

# no randomness needed in either direction
@gen function q()
end

@bijection function h(model_args, proposal_args, proposal_retval)

    x = @read_continuous_from_model(:x)
    y = @read_continuous_from_model(:y)

    r = sqrt(x*x + y*y)
    theta = atan(y, x)

    @write_continuous_to_model(:r, r)
    @write_continuous_to_model(:theta, theta)
end

@bijection function h_inv(model_args, proposal_args, proposal_retval)

    r = @read_continuous_from_model(:r)
    theta = @read_continuous_from_model(:theta)

    x = r * cos(theta)
    y = r * sin(theta)

    @write_continuous_to_model(:x, x)
    @write_continuous_to_model(:y, y)
end

function do_inference()

    model1_obs = choicemap((:obsx, 1.), (:obsy, 1.))
    model2_obs = choicemap((:obsr, 1.))

    # generate initial trace of first conditioned model
    trace, init_weight = generate(model1, (), model1_obs)
    display(get_choices(trace))
    
    # do some inference in the first conditioned model
    @gen x_prop(tr) = ({:x} ~ normal(tr[:x], 0.1))
    @gen y_prop(tr) = ({:y} ~ normal(tr[:y], 0.1))
    for i=1:10
        trace, = mh(trace, x_prop, ())
        trace, = mh(trace, y_prop, ())
    end
    
    # change variables into second conditioned model
    (trace, u_back, incr_weight1) = h(
        trace, simulate(q, ()), model2, (),
        model2_obs; check=true)
    display(u_back)
    display(get_choices(trace))
    
    # do some inference in the second conditioned model
    @gen r_prop(tr) = ({:r} ~ normal(tr[:r], 0.1))
    @gen theta_prop(tr) = ({:theta} ~ normal(tr[:theta], 0.1))
    for i=1:10
        trace, = mh(trace, r_prop, ())
        trace, = mh(trace, theta_prop, ())
    end
    
    # change variables back into model1
    (trace, u_back, incr_weight2) = h_inv(
        trace, simulate(q, ()), model1, (),
        model1_obs; check=true)
    display(u_back)
    display(get_choices(trace))
    
    weight = init_weight + incr_weight1 + incr_weight2
    println("importance weight: $weight")
end

do_inference()
