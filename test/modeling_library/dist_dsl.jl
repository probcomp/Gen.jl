@testset "dist DSL (untyped)" begin
  # Test transformed distributions
  @dist f(x) = exp(normal(x, 0.001))
  @test isapprox(1, f(0); atol = 5)
  @test isapprox(logpdf(f, 1., 0.), logpdf(normal, 0., 0., 0.001))

  # Test gradients of transformed distributions
  @dist shifted_normal(mu, sigma) = normal(mu, sigma) + 1.
  @test logpdf(shifted_normal, 1., 0., 1.) == logpdf(normal, 0., 0., 1.)
  @test logpdf_grad(shifted_normal, 0., 0., 1.) == logpdf_grad(normal, -1., 0., 1.)

  # Test gradients of transformed distributions with no parameters
  @dist shifted_std_normal() = normal(0., 1.) + 1.
  @test logpdf(shifted_std_normal, 1.) == logpdf(normal, 0., 0., 1.)
  @test logpdf_grad(shifted_std_normal, 0.) == (logpdf_grad(normal, -1., 0., 1.)[1],)

  # Test gradients of multivariate distributions
  @dist vec_normal(mu, sigma) = broadcasted_normal(broadcast(+, mu, 1.0), broadcast(*, sigma, 2.0))
  @test logpdf(vec_normal, zeros(2), zeros(2), ones(2)) ==
        logpdf(broadcasted_normal, zeros(2), ones(2), 2 .* ones(2))
  transformed_grads = logpdf_grad(vec_normal, zeros(2), zeros(2), ones(2))
  orig_grads = logpdf_grad(broadcasted_normal, zeros(2), ones(2), 2 .* ones(2)) 
  @test transformed_grads[1] == orig_grads[1]
  @test transformed_grads[2] == orig_grads[2]
  @test transformed_grads[3] == 2.0 * orig_grads[3]
      
  # Test gradients of multivariate distributions with multi-dimensional arguments
  @dist transformed_mvnormal(mu, sigma) = mvnormal(broadcast(+, mu, 1.0), broadcast(*, sigma, 2.0))
  @test logpdf(transformed_mvnormal, zeros(2), zeros(2), ones(2, 2)) ==
        logpdf(mvnormal, zeros(2), ones(2), 2 .* ones(2, 2))
  transformed_grads = logpdf_grad(transformed_mvnormal, zeros(2), zeros(2), ones(2, 2))
  orig_grads = logpdf_grad(mvnormal, zeros(2), ones(2), 2 .* ones(2, 2)) 
  @test transformed_grads[1] == orig_grads[1]
  @test transformed_grads[2] == orig_grads[2]
  @test transformed_grads[3] == 2.0 * orig_grads[3]

  # Test relabeled distributions with labels provided as an Array
  @dist labeled_cat(labels, probs) = labels[categorical(probs)]
  @test labeled_cat([:a, :b], [0., 1.]) == :b
  @test isapprox(logpdf(labeled_cat, :b, [:a, :b], [0.5, 0.5]), log(0.5))
  @test logpdf_grad(labeled_cat, :b, [:a, :b], [0.5, 0.5]) == (nothing, logpdf_grad(categorical, 2, [0.5, 0.5])...)
  @test logpdf(labeled_cat, :c, [:a, :b], [0.5, 0.5]) == -Inf

  # Test relabeled distributions with labels provided in a Dict
  dict = Dict(1 => :a, 2 => :b)
  @dist dict_cat(probs) = dict[categorical(probs)]
  @test dict_cat([0., 1.]) == :b
  @test isapprox(logpdf(dict_cat, :b, [0.5, 0.5]), log(0.5))
  @test logpdf_grad(dict_cat, :b, [0.5, 0.5]) == logpdf_grad(categorical, 2, [0.5, 0.5])
  @test logpdf(dict_cat, :c, [0.5, 0.5]) == -Inf

  # Test relabeled distributions with Enum labels
  @enum Fruit apple orange
  @dist enum_cat(probs) = Fruit(categorical(probs) - 1)
  @test enum_cat([0., 1.]) == orange
  @test isapprox(logpdf(enum_cat, orange, [0.5, 0.5]), log(0.5))
  @test logpdf(enum_cat, orange, [1.0]) == -Inf
  @test logpdf_grad(enum_cat, orange, [0.5, 0.5]) == logpdf_grad(categorical, 2, [0.5, 0.5])

  # Regression test for https://github.com/probcomp/Gen/issues/253
  @dist real_minus_uniform(a, b) = 1 - Gen.uniform(a, b)
  @test real_minus_uniform(1, 2) < 0
  @test logpdf(real_minus_uniform, -0.5, 1, 2) == 0.0
  @test logpdf_grad(real_minus_uniform, -0.5, 1, 2) == logpdf_grad(uniform, 1.5, 1, 2)
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
  @test logpdf_grad(symbol_cat, :b, [:a, :b], [0.5, 0.5]) == (nothing, logpdf_grad(categorical, 2, [0.5, 0.5])...)
  @test_throws MethodError logpdf(symbol_cat, "c", [:a, :b], [0.5, 0.5])

  # Test typed parameters
  @dist int_bounded_uniform(low::Int, high::Int) = uniform(low, high)
  @test 0.0 <= int_bounded_uniform(0, 1) <= 1
  @test_throws MethodError int_bounded_uniform(-0.5, 0.5)
  @test logpdf(int_bounded_uniform, 0.5, 0, 1) == 0
  @test logpdf_grad(int_bounded_uniform, 0.5, 0, 1) == logpdf_grad(uniform, 0.5, 0, 1)
  @test_throws MethodError logpdf(int_bounded_uniform, 0.0, -0.5, 0.5)

  # Test relabeled distributions with user-defined types
  @dist mylabel_cat(labels::Vector{MyLabel}, probs) = labels[categorical(probs)]
  @test mylabel_cat([MyLabel(:a), MyLabel(:b)], [0., 1.]) == MyLabel(:b)
  @test_throws MethodError mylabel_cat([:a, :b], [0., 1.])
  @test logpdf(mylabel_cat, MyLabel(:a), [MyLabel(:a)], [1.0]) == 0
  @test_throws MethodError logpdf(mylabel_cat, :a, [MyLabel(:a)], [1.0])
end
