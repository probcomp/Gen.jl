@testset "dist DSL" begin
  @dist f(x) = exp(normal(x, 0.001))
  @test isapprox(1, f(0); atol = 5)

  @dist labeled_cat(labels, probs) = labels[categorical(probs)]
  @test labeled_cat([:a, :b], [0., 1.]) == :b
  @test isapprox(logpdf(labeled_cat, :b, [:a, :b], [0.5, 0.5]), log(0.5))
  @test logpdf(labeled_cat, :c, [:a, :b], [0.5, 0.5]) == -Inf

  dict = Dict(1 => :a, 2 => :b)
  @dist dict_cat(probs) = dict[categorical(probs)]
  @test dict_cat([0., 1.]) == :b
  @test isapprox(logpdf(dict_cat, :b, [0.5, 0.5]), log(0.5))
  @test logpdf(dict_cat, :c, [0.5, 0.5]) == -Inf

  @enum Fruit apple orange
  @dist enum_cat(probs) = Fruit(categorical(probs) - 1)
  @test enum_cat([0., 1.]) == orange
  @test isapprox(logpdf(enum_cat, orange, [0.5, 0.5]), log(0.5))
  @test logpdf(enum_cat, orange, [1.0]) == -Inf

  @dist symbol_cat(labels::Vector{Symbol}, probs) = labels[categorical(probs)]
  @test symbol_cat([:a, :b], [0., 1.]) == :b
  @test_throws MethodError symbol_cat(["a", "b"], [0., 1.])
  @test logpdf(symbol_cat, :c, [:a, :b], [0.5, 0.5]) == -Inf
  @test_throws MethodError logpdf(symbol_cat, "c", [:a, :b], [0.5, 0.5])

  @dist int_bounded_uniform(low::Int, high::Int) = uniform(low, high)
  @test 0.0 <= int_bounded_uniform(0, 1) <= 1
  @test_throws MethodError int_bounded_uniform(-0.5, 0.5)
  @test logpdf(int_bounded_uniform, 0.5, 0, 1) == 0
  @test_throws MethodError logpdf(int_bounded_uniform, 0.0, -0.5, 0.5)
end
