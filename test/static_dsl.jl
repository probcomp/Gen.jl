struct Params
    prob_outlier::Float64
    slope::Float64
    intercept::Float64
    inlier_std::Float64
    outlier_std::Float64
end

@staticgen function datum(x, @grad(params::Params))
    is_outlier::Bool = @addr(bernoulli(params.prob_outlier), :z)
    std::Float64 = is_outlier ? params.inlier_std : params.outlier_std
    y::Float64 = @addr(normal(x * params.slope + params.intercept, std), :y)
    return y
end

@testset "static DSL" begin

ir = Gen.get_ir(typeof(datum))

# argument nodes
@test length(ir.arg_nodes) == 2
x = ir.arg_nodes[1]
params = ir.arg_nodes[2]
@test x.name == :x
@test x.typ == Any
@test !x.compute_grad
@test params.name == :params
@test params.typ == Params
@test params.compute_grad

# choice nodes
@test length(ir.choice_nodes) == 2

# call nodes
@test length(ir.call_nodes) == 0

# is_outlier
is_outlier = ir.choice_nodes[1]
@test is_outlier.name == :is_outlier
@test is_outlier.addr == :z
@test is_outlier.typ == Bool
@test is_outlier.dist == bernoulli
@test length(is_outlier.inputs) == 1

# y
y = ir.choice_nodes[2]
@test y.name == :y
@test y.addr == :y
@test y.typ == Float64
@test y.dist == normal
@test length(y.inputs) == 2

# y_mean
y_mean = y.inputs[1]
@test isa(y_mean, Gen.JuliaNode)
@test y_mean.typ == Any
@test length(y_mean.inputs) == 2
in1 = y_mean.inputs[1]
in2 = y_mean.inputs[2]
@test (in1 == x && in2 == params) || (in2 == x && in1 === params)

# std
std = y.inputs[2]
@test isa(std, Gen.JuliaNode)
@test std.name == :std
@test std.typ == Float64
@test length(std.inputs) == 2
in1 = std.inputs[1]
in2 = std.inputs[2]
@test (in1 == is_outlier && in2 == params) || (in2 == is_outlier && in1 === params)

# prob outlier
prob_outlier = is_outlier.inputs[1]
@test isa(prob_outlier, Gen.JuliaNode)
@test length(prob_outlier.inputs) == 1
@test prob_outlier.inputs[1] == params
@test prob_outlier.typ == Any

@test ir.return_node == y

end # @testset "static DSL"
