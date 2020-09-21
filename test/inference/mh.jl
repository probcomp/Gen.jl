@gen function involutive_mcmc_test_model()
    x ~ bernoulli(0.5)
    if x
        y ~ normal(0, 1)
    else
        z ~ normal(1, 1)
    end
end

@gen function involutive_mcmc_test_proposal(trace)
end

@testset "Involutive MCMC" begin

    @transform dsl_involution (model_in, aux_in) to (model_out, aux_out) begin
        x = @read(model_in[:x], :discrete)
        @write(model_out[:x], !x, :discrete)
        if x
            @copy(model_in[:y], model_out[:z])
        else
            @copy(model_in[:z], model_out[:y])
        end
    end

    function julia_involution(trace, forward_choices, forward_retval, q_args)
        argdiffs = map((_) -> NoChange(), get_args(trace))
        constraints = choicemap((:x, !trace[:x]))
        if trace[:x]
            constraints[:z] = trace[:y]
        else
            constraints[:y] = trace[:z]
        end
        (new_trace, weight, _, _) = update(trace, get_args(trace), argdiffs, constraints)
        # NOTE: Jacobian is 1.0
        return (new_trace, choicemap(), weight)
    end

    trace, _ = generate(involutive_mcmc_test_model, (), choicemap((:x, true), (:y, 0.5)))
    for involution in [dsl_involution, julia_involution]
    
        new_trace, accepted = involutive_mcmc(trace, involutive_mcmc_test_proposal, (), involution; check=true)
        # it should be accepted since the new and old densities are the same (0.5 is halfway between means)
        @test accepted
        @test !new_trace[:x]
        @test new_trace[:z] == 0.5
    end

end
