@gen function kernel_dsl_test_model()
    x ~ normal(0.0, 1.0)
    y ~ normal(x, 1.0)
    return y
end

@pkern function k1(tr, t)
    metadata = false
    for _ in 1 : t
        tr, acc = mh(tr, select(:x))
        if acc
            metadata = true
        end
    end
    return tr, metadata
end

@gen function k1_proposal(tr)
    x ~ normal(0.0, 1.0)
end

@kern function k1_non_primitive(tr, t)
    for _ in 1 : t
        tr ~ mh(tr, k1_proposal, ())
    end
end

@testset "Kernel DSL" begin
    tr = simulate(kernel_dsl_test_model, ())
    original = get_choices(tr)[:x]
    tr, acc = k1(tr, 1)
    if acc
        @test original != get_choices(tr)[:x]
    else
        @test original == get_choices(tr)[:x]
    end
    original = get_choices(tr)[:x]
    tr, acc = k1_non_primitive(tr, 1)
    tr, acc = Gen.reversal(k1_non_primitive)(tr, 1)
end
