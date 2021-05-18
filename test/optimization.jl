@testset "optimization" begin

@testset "Julia parameter store" begin

    store = JuliaParameterStore()

    @gen function foo()
        @param theta::Float64
        @param phi::Vector{Float64}
    end
    register_parameters!(foo, [:theta, :phi])

    # before the parameters are initialized in the store

    @test Gen.get_local_parameters(store, foo) == Dict{Symbol,Any}()

    @test_throws KeyError get_gradient((foo, :theta), store)
    @test_throws KeyError get_parameter_value((foo, :theta), store)
    @test_throws KeyError increment_gradient!((foo, :theta), 1.0, store)
    @test_throws KeyError reset_gradient!((foo, :theta), store)
    @test_throws KeyError Gen.set_parameter_value!((foo, :theta), 1.0, store)
    @test_throws KeyError Gen.get_gradient_accumulator((foo, :theta), store)

    @test_throws KeyError get_gradient((foo, :phi), store)
    @test_throws KeyError get_parameter_value((foo, :phi), store)
    @test_throws KeyError increment_gradient!((foo, :phi), [1.0, 1.0], store)
    @test_throws KeyError reset_gradient!((foo, :phi), store)
    @test_throws KeyError Gen.set_parameter_value!((foo, :phi), [1.0, 1.0], store)
    @test_throws KeyError Gen.get_gradient_accumulator((foo, :phi), store)

    # after the parameters are initialized in the store

    init_parameter!((foo, :theta), 1.0, store)
    init_parameter!((foo, :phi), [1.0, 2.0], store)

    dict = Gen.get_local_parameters(store, foo)
    @test length(dict) == 2
    @test dict[:theta] == 1.0
    @test dict[:phi] == [1.0, 2.0]

    @test get_gradient((foo, :theta), store) == 0.0
    @test get_parameter_value((foo, :theta), store) == 1.0
    increment_gradient!((foo, :theta), 1.1, store)
    @test get_gradient((foo, :theta), store) == 1.1
    increment_gradient!((foo, :theta), 1.1, 2.0, store)
    @test get_gradient((foo, :theta), store) == (1.1 + 2.2)
    reset_gradient!((foo, :theta), store)
    @test get_gradient((foo, :theta), store) == 0.0
    Gen.set_parameter_value!((foo, :theta), 2.0, store)
    @test get_parameter_value((foo, :theta), store) == 2.0
    @test get_value(Gen.get_gradient_accumulator((foo, :theta), store)) == 0.0

    @test get_gradient((foo, :phi), store) == [0.0, 0.0]
    @test get_parameter_value((foo, :phi), store) == [1.0, 2.0]
    increment_gradient!((foo, :phi), [1.1, 1.2], store)
    @test get_gradient((foo, :phi), store) == [1.1, 1.2]
    increment_gradient!((foo, :phi), [1.1, 1.2], 2.0, store)
    @test get_gradient((foo, :phi), store) == ([1.1, 1.2] .+ (2.0 * [1.1, 1.2]))
    reset_gradient!((foo, :phi), store)
    @test get_gradient((foo, :phi), store) == [0.0, 0.0]
    Gen.set_parameter_value!((foo, :phi), [2.0, 3.0], store)
    @test get_parameter_value((foo, :phi), store) == [2.0, 3.0]
    @test Gen.get_value(Gen.get_gradient_accumulator((foo, :phi), store)) == [0.0, 0.0]

    # check that the default global Julia store was unaffected
    @test_throws KeyError get_parameter_value((foo, :theta))
    @test_throws KeyError get_gradient((foo, :theta))
    @test_throws KeyError increment_gradient!((foo, :theta), 1.0)

    # FixedStepGradientDescent
    init_parameter!((foo, :theta), 1.0, store)
    init_parameter!((foo, :phi), [1.0, 2.0], store)
    increment_gradient!((foo, :theta), 2.0, store)
    increment_gradient!((foo, :phi), [1.0, 3.0], store)
    optimizer = init_optimizer(FixedStepGradientDescent(1e-2), [(foo, :theta)], store)
    apply_update!(optimizer) # update just theta
    @test get_gradient((foo, :theta), store) == 0.0
    @test get_parameter_value((foo, :theta), store) == 1.0 + (2.0 * 1e-2)
    @test get_gradient((foo, :phi), store) == [1.0, 3.0] # unchanged
    @test get_parameter_value((foo, :phi), store) == [1.0, 2.0] # unchanged
    optimizer = init_optimizer(FixedStepGradientDescent(1e-2), [(foo, :phi)], store)
    apply_update!(optimizer) # update just phi
    @test get_gradient((foo, :phi), store) == [0.0, 0.0]
    @test get_parameter_value((foo, :phi), store) == ([1.0, 2.0] .+ 1e-2 * [1.0, 3.0])

    # DecayStepGradientDescent
    # TODO

    # default_parameter_context and default_julia_parameter_store
end

@testset "composite optimizer" begin

end


end
