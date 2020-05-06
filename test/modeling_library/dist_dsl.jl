@testset "dist DSL (untyped)" begin
  # Test transformed distributions
  @dist f(x) = exp(normal(x, 0.001))
  @test isapprox(1, f(0); atol = 5)

  # Test relabeled distributions with labels provided as an Array
  @dist labeled_cat(labels, probs) = labels[categorical(probs)]
  @test labeled_cat([:a, :b], [0., 1.]) == :b
  @test isapprox(logpdf(labeled_cat, :b, [:a, :b], [0.5, 0.5]), log(0.5))
  @test logpdf(labeled_cat, :c, [:a, :b], [0.5, 0.5]) == -Inf

  # Test relabeled distributions with labels provided in a Dict
  dict = Dict(1 => :a, 2 => :b)
  @dist dict_cat(probs) = dict[categorical(probs)]
  @test dict_cat([0., 1.]) == :b
  @test isapprox(logpdf(dict_cat, :b, [0.5, 0.5]), log(0.5))
  @test logpdf(dict_cat, :c, [0.5, 0.5]) == -Inf

  # Test relabeled distributions with Enum labels
  @enum Fruit apple orange
  @dist enum_cat(probs) = Fruit(categorical(probs) - 1)
  @test enum_cat([0., 1.]) == orange
  @test isapprox(logpdf(enum_cat, orange, [0.5, 0.5]), log(0.5))
  @test logpdf(enum_cat, orange, [1.0]) == -Inf

  # Regression test for Issue #253
  @dist real_minus_uniform(a, b) = 1 - Gen.uniform(a, b)
  @test real_minus_uniform(1, 2) < 0
  @test logpdf(real_minus_uniform, -0.5, 1, 2) == 0.0
end

# User-defined type for testing purposes
struct MyLabel
  name::Symbol
end

@testset "dist DSL (typed)" begin
  # Test typed relabeled distributions
  @dist symbol_cat(labels::Vector{Symbol}, probs) = labels[categorical(probs)]
  @test symbol_cat([:a, :b], [0., 1.]) == :b
  @test_throws MethodError symbol_cat(["a", "b"], [0., 1.])
  @test logpdf(symbol_cat, :c, [:a, :b], [0.5, 0.5]) == -Inf
  @test_throws MethodError logpdf(symbol_cat, "c", [:a, :b], [0.5, 0.5])

  # Test typed parameters
  @dist int_bounded_uniform(low::Int, high::Int) = uniform(low, high)
  @test 0.0 <= int_bounded_uniform(0, 1) <= 1
  @test_throws MethodError int_bounded_uniform(-0.5, 0.5)
  @test logpdf(int_bounded_uniform, 0.5, 0, 1) == 0
  @test_throws MethodError logpdf(int_bounded_uniform, 0.0, -0.5, 0.5)

  # Test relabeled distributions with user-defined types
  @dist mylabel_cat(labels::Vector{MyLabel}, probs) = labels[categorical(probs)]
  @test mylabel_cat([MyLabel(:a), MyLabel(:b)], [0., 1.]) == MyLabel(:b)
  @test_throws MethodError mylabel_cat([:a, :b], [0., 1.])
  @test logpdf(mylabel_cat, MyLabel(:a), [MyLabel(:a)], [1.0]) == 0
  @test_throws MethodError logpdf(mylabel_cat, :a, [MyLabel(:a)], [1.0])
end
