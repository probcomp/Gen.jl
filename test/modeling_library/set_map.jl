using FunctionalCollections: push, disj
@testset "multiset" begin
    ms = MultiSet()
    @test ms isa MultiSet{Any}

    ms = MultiSet{Int}()
    ms = push(ms, 1)
    @test length(ms) == 1
    ms = push(ms, 1)
    @test length(ms) == 2
    @test collect(ms) == [1, 1]

    ms = push(ms, 2)
    @test length(ms) == 3
    @test 2 in ms
    @test 1 in ms
    ms = remove_one(ms, 1)
    @test 1 in ms
    @test 2 in ms
    ms = push(ms, 2)
    @test length(ms) == 3
    ms = disj(ms, 2)
    @test length(ms) == 1

    @test push(push(ms, 2), 2) == MultiSet([1, 2, 2])
    @test MultiSet([2, 1, 2, 5]) == MultiSet([2, 5, 2, 1])

    total = 0
    for el in MultiSet([2, 1, 2, 5])
        total += el
    end
    @test total == 2+1+2+5

    @test setmap(x -> x^2, Set([-2, -1, 0, 1])) == MultiSet([4, 1, 1, 0])
end

@testset "SetMap" begin
    priors = [
        [0.1, 0.3, 0.6],
        [0.2, 0.6, 0.2],
        [0.6, 0.2, 0.2]
    ]
    tr, weight = generate(SetMap(categorical), (Set(priors),), choicemap((priors[1], 2)))
    @test tr[priors[1]] == 2
    @test tr[priors[2]] in (1, 2, 3)
    @test tr[priors[3]] in (1, 2, 3)
    @test isapprox(weight, log(0.3))

    tr = simulate(SetMap(categorical), (Set(priors),))
    exp_score = sum(logpdf(categorical, tr[priors[i]], priors[i]) for i=1:3)
    @test isapprox(get_score(tr), exp_score)

    current1 = tr[priors[1]]
    new = current1 == 1 ? 2 : 1

    new_tr, weight, _, discard = update(tr, (Set(priors),), (NoChange(),), choicemap((priors[1], new)))
    expected_weight = logpdf(categorical, new, priors[1]) - logpdf(categorical, tr[priors[1]], priors[1])
    @test isapprox(weight, expected_weight)
    @test isapprox(get_score(new_tr) - get_score(tr), expected_weight)
    @test discard == choicemap((priors[1], tr[priors[1]]))

    new_tr, weight, _, discard = update(tr, (Set(priors[1:2]),), (UnknownChange(),), EmptyAddressTree())
    expected_weight = -logpdf(categorical, tr[priors[3]], priors[3])
    @test isapprox(weight, expected_weight)
    @test isapprox(get_score(new_tr), sum(logpdf(categorical, tr[priors[i]], priors[i]) for i=1:2))
    @test discard == choicemap((priors[3], tr[priors[3]]))
end