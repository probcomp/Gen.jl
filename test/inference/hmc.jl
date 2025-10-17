@testset "hmc tests" begin
    import LinearAlgebra, Random
    
    # smoke test a function without retval gradient
    @gen function foo()
        x = @trace(normal(0, 1), :x)
        return x
    end

    (trace, _) = generate(foo, ())
    (new_trace, accepted) = hmc(trace, select(:x))

    # smoke test a function with retval gradient
    @gen (grad) function foo()
        x = @trace(normal(0, 1), :x)
        return x
    end

    (trace, _) = generate(foo, ())
    (new_trace, accepted) = hmc(trace, select(:x))

    # For Normal(0,1), grad should be -x
    (_, values_trie, gradient_trie) = choice_gradients(trace, select(:x), 0)
    values = to_array(values_trie, Float64)
    grad = to_array(gradient_trie, Float64)
    @test values ≈ -grad

    # smoke test with vector metric
    @gen function bar()
        x = @trace(normal(0, 1), :x)
        y = @trace(normal(0, 1), :y)
        return x + y
    end

    (trace, _) = generate(bar, ())
    metric_vec = [1.0, 2.0]
    (new_trace, accepted) = hmc(trace, select(:x, :y); metric=metric_vec)

    # smoke test with Diagonal metric
    (trace, _) = generate(bar, ())
    metric_diag = LinearAlgebra.Diagonal([1.0, 2.0])
    (new_trace, accepted) = hmc(trace, select(:x, :y); metric=metric_diag)

    # smoke test with Dense matrix metric
    (trace, _) = generate(bar, ())
    metric_dense = [1.0 0.1; 0.1 2.0]
    (new_trace, accepted) = hmc(trace, select(:x, :y); metric=metric_dense)

    # smoke test with vector metric and retval gradient
    @gen (grad) function bar_grad()
        x = @trace(normal(0, 1), :x)
        y = @trace(normal(0, 1), :y)
        return x + y
    end

    (trace, _) = generate(bar_grad, ())
    metric_vec = [0.5, 1.5]
    (new_trace, accepted) = hmc(trace, select(:x, :y); metric=metric_vec)

    # For each Normal(0,1), grad should be -x
    (_, values_trie, gradient_trie) = choice_gradients(trace, select(:x, :y), 0)
    values = to_array(values_trie, Float64)
    grad = to_array(gradient_trie, Float64)
    @test values ≈ -grad

    # smoke test with Diagonal metric and retval gradient
    (trace, _) = generate(bar_grad, ())
    metric_diag = LinearAlgebra.Diagonal([0.5, 1.5])
    (new_trace, accepted) = hmc(trace, select(:x, :y); metric=metric_diag)

    # smoke test with Dense matrix metric and retval gradient
    (trace, _) = generate(bar_grad, ())
    metric_dense = [0.5 0.2; 0.2 1.5]
    (new_trace, accepted) = hmc(trace, select(:x, :y); metric=metric_dense)
end

@testset "hmc metric behavior" begin
    import LinearAlgebra, Random

    # test that different metrics produce different behavior
    @gen function test_metric_effect()
        x = @trace(normal(0, 1), :x)
        y = @trace(normal(0, 1), :y)
        return x + y
    end

    (trace1, _) = generate(test_metric_effect, ())
    

    # Set RNG to a known state for comparison
    Random.seed!(1)

    # Run HMC with identity metric (default)
    (trace_identity, _) = hmc(trace1, select(:x, :y); L=5)
    
    # Reset RNG to same state for comparison
    Random.seed!(1)
    
    # Run HMC with scaled metric (should behave differently)
    metric_scaled = [10.0, 0.1]  # Very different scales
    (trace_scaled, _) = hmc(trace1, select(:x, :y); L=5, metric=metric_scaled)
    
    # With same RNG sequence but different metrics, should get different results
    @test get_choices(trace_identity) != get_choices(trace_scaled)

    # With same metric but different representations, should get similar results
    # Test many times to check statistical similarity
    acceptances_diag = Float64[]
    acceptances_dense = Float64[]
    
    for i in 1:50
        # Reset to predictable state for each iteration
        Random.seed!(i)
        (_, accepted_diag) = hmc(trace1, select(:x, :y); 
                                metric=LinearAlgebra.Diagonal([2.0, 3.0]))
        
        # Reset to same state for comparison
        Random.seed!(i)
        (_, accepted_dense) = hmc(trace1, select(:x, :y); 
                                 metric=[2.0 0.0; 0.0 3.0])
        
        # Collect acceptance results
        push!(acceptances_diag, float(accepted_diag))
        push!(acceptances_dense, float(accepted_dense))
    end
    
    # # Should have similar acceptance rates (within 20%)
    rate_diag = Distributions.mean(acceptances_diag)
    rate_dense = Distributions.mean(acceptances_dense)
    @test abs(rate_diag - rate_dense) < 0.2


end

@testset "Bad metric catches" begin
    @gen function bar()
        x = @trace(normal(0, 1), :x)
        y = @trace(normal(0, 1), :y)
        return x + y
    end

    bad_metrics =([-1.0 -20.0; 0.0 1.0], # Bad dense,
                   LinearAlgebra.Diagonal([-1.0, -20.0]), # Bad diag
                   [-5.0, 20.0],  # Bad vector diag
    )

    for bad_metric in bad_metrics
        (trace, _) = generate(bar, ())
        @test_throws Exception hmc(trace, select(:x, :y); metric=bad_metric)
    end
    
end