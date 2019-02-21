#######################
# static DSL function #
#######################

using FunctionalCollections: PersistentVector

struct Params
    prob_outlier::Float64
    slope::Float64
    intercept::Float64
    inlier_std::Float64
    outlier_std::Float64
end

@gen (static) function datum(x::Float64, (grad)(params::Params))
    is_outlier = @trace(bernoulli(params.prob_outlier), :z)
    std = is_outlier ? params.inlier_std : params.outlier_std
    y = @trace(normal(x * params.slope + params.intercept, std), :y)
    return y
end

data_fn = Map(datum)

@gen (static) function model(xs::Vector{Float64})
    n = length(xs)
    inlier_std = @trace(gamma(1, 1), :inlier_std)
    outlier_std = @trace(gamma(1, 1), :outlier_std)
    slope = @trace(normal(0, 2), :slope)
    intercept = @trace(normal(0, 2), :intercept)
    params::Params = Params(0.5, inlier_std, outlier_std, slope, intercept)
    ys = @trace(data_fn(xs, fill(params, n)), :data)
    return ys
end

@gen (static) function at_choice_example_1(i::Int)
    ret = @trace(bernoulli(0.5), :x => i)
end

# @trace(choice_at(bernoulli)(0.5, i), :x)

@gen (static) function at_choice_example_2(i::Int)
    ret = @trace(bernoulli(0.5), :x => i => :y)
end

# @trace(call_at(choice_at(bernoulli))(0.5, i, :y), :x)

@gen function foo(mu)
    @trace(normal(mu, 1), :y)
end

@gen (static) function at_call_example_1(i::Int)
    mu = 1.123
    ret = @trace(foo(mu), :x => i)
end

@gen (static) function at_call_example_2(i::Int)
    mu = 1.123
    ret = @trace(foo(mu), :x => i => :y)
end

@testset "static DSL" begin

function get_node_by_name(ir, name::Symbol)
    nodes = filter((node) -> (node.name == name), ir.nodes)
    @assert length(nodes) == 1
    nodes[1]
end

#####################
# check IR of datum #
#####################

ir = Gen.get_ir(typeof(datum))

# argument nodes
@test length(ir.arg_nodes) == 2
x = ir.arg_nodes[1]
params = ir.arg_nodes[2]
@test x.name == :x
@test x.typ == :Float64
@test !x.compute_grad
@test params.name == :params
@test params.typ == :Params
@test params.compute_grad

# choice nodes and call nodes
@test length(ir.choice_nodes) == 2
@test length(ir.call_nodes) == 0

# is_outlier
is_outlier = ir.choice_nodes[1]
@test is_outlier.name == :is_outlier
@test is_outlier.addr == :z
@test is_outlier.typ == QuoteNode(Bool)
@test is_outlier.dist == bernoulli
@test length(is_outlier.inputs) == 1

# std
std = get_node_by_name(ir, :std)
@test isa(std, Gen.JuliaNode)
@test std.name == :std
@test std.typ == QuoteNode(Any)
@test length(std.inputs) == 2
in1 = std.inputs[1]
in2 = std.inputs[2]
@test (in1 === is_outlier && in2 === params) || (in2 === is_outlier && in1 === params)

# y
y = ir.choice_nodes[2]
@test y.name == :y
@test y.addr == :y
@test y.typ == QuoteNode(Float64)
@test y.dist == normal
@test length(y.inputs) == 2
@test y.inputs[2] === std

# y_mean
y_mean = y.inputs[1]
@test isa(y_mean, Gen.JuliaNode)
@test y_mean.typ == QuoteNode(Any)
@test length(y_mean.inputs) == 2
in1 = y_mean.inputs[1]
in2 = y_mean.inputs[2]
@test (in1 === x && in2 === params) || (in2 === x && in1 === params)

# prob outlier
prob_outlier = is_outlier.inputs[1]
@test isa(prob_outlier, Gen.JuliaNode)
@test length(prob_outlier.inputs) == 1
@test prob_outlier.inputs[1] === params
@test prob_outlier.typ == QuoteNode(Any)

@test ir.return_node === y

#####################
# check IR of model #
#####################

ir = Gen.get_ir(typeof(model))
@test length(ir.arg_nodes) == 1
xs = ir.arg_nodes[1]
@test xs.name == :xs
@test xs.typ == :(Vector{Float64})
@test !xs.compute_grad

# choice nodes and call nodes
@test length(ir.choice_nodes) == 4
@test length(ir.call_nodes) == 1

# inlier_std
inlier_std = ir.choice_nodes[1]
@test inlier_std.name == :inlier_std
@test inlier_std.addr == :inlier_std
@test inlier_std.typ == QuoteNode(Float64)
@test inlier_std.dist == gamma
@test length(inlier_std.inputs) == 2

# outlier_std 
outlier_std = ir.choice_nodes[2]
@test outlier_std.name == :outlier_std
@test outlier_std.addr == :outlier_std
@test outlier_std.typ == QuoteNode(Float64)
@test outlier_std.dist == gamma
@test length(outlier_std.inputs) == 2

# slope 
slope = ir.choice_nodes[3]
@test slope.name == :slope
@test slope.addr == :slope
@test slope.typ == QuoteNode(Float64)
@test slope.dist == normal
@test length(slope.inputs) == 2

# intercept 
intercept = ir.choice_nodes[4]
@test intercept.name == :intercept
@test intercept.addr == :intercept
@test intercept.typ == QuoteNode(Float64)
@test intercept.dist == normal
@test length(intercept.inputs) == 2

# data
ys = ir.call_nodes[1]
@test ys.name == :ys
@test ys.addr == :data
@test ys.typ == QuoteNode(PersistentVector{Float64})
@test ys.generative_function == data_fn
@test length(ys.inputs) == 2
@test ys.inputs[1] == xs

# params
params = get_node_by_name(ir, :params)
@test isa(params, Gen.JuliaNode)
@test params.name == :params
@test params.typ == :Params
@test length(params.inputs) == 4
@test slope in params.inputs
@test intercept in params.inputs
@test inlier_std in params.inputs
@test outlier_std in params.inputs

# n
n = get_node_by_name(ir, :n)
@test isa(n, Gen.JuliaNode)
@test n.name == :n
@test n.typ == QuoteNode(Any) 
@test length(n.inputs) == 1
@test n.inputs[1] === xs

# params_vec
params_vec = ys.inputs[2]
@test isa(params_vec, Gen.JuliaNode)
@test params_vec.typ == QuoteNode(Any)
@test length(params_vec.inputs) == 2
in1 = params_vec.inputs[1]
in2 = params_vec.inputs[2]
@test (in1 === params && in2 === n) || (in2 === params && in1 === n)

@test ir.return_node === ys

#####################
## at_choice syntax #
#####################

# at_choice_example_1
ir = Gen.get_ir(typeof(at_choice_example_1))
i = get_node_by_name(ir, :i)
ret = get_node_by_name(ir, :ret)
@test isa(ret, Gen.GenerativeFunctionCallNode)
@test ret.addr == :x
@test length(ret.inputs) == 2
@test isa(ret.inputs[1], Gen.JuliaNode) # () -> 0.5
@test ret.inputs[2] === i
at = ret.generative_function
@test isa(at, Gen.ChoiceAtCombinator)
@test at.dist == bernoulli

# at_choice_example_2
ir = Gen.get_ir(typeof(at_choice_example_2))
i = get_node_by_name(ir, :i)
ret = get_node_by_name(ir, :ret)
@test isa(ret, Gen.GenerativeFunctionCallNode)
@test ret.addr == :x
@test length(ret.inputs) == 3
@test isa(ret.inputs[1], Gen.JuliaNode) # () -> 0.5
@test isa(ret.inputs[2], Gen.JuliaNode) # () -> :y
@test ret.inputs[3] === i
at = ret.generative_function
@test isa(at, Gen.CallAtCombinator)
at2 = at.kernel
@test isa(at2, Gen.ChoiceAtCombinator)
@test at2.dist == bernoulli


##################
# at_call syntax #
##################

# at_call_example_1
ir = Gen.get_ir(typeof(at_call_example_1))
i = get_node_by_name(ir, :i)
ret = get_node_by_name(ir, :ret)
@test isa(ret, Gen.GenerativeFunctionCallNode)
@test ret.addr == :x
@test length(ret.inputs) == 2
@test isa(ret.inputs[1], Gen.JuliaNode) # () -> 0.5
@test ret.inputs[2] === i
at = ret.generative_function
@test isa(at, Gen.CallAtCombinator)
@test at.kernel == foo

#at_call_example_2
ir = Gen.get_ir(typeof(at_call_example_2))
i = get_node_by_name(ir, :i)
ret = get_node_by_name(ir, :ret)
@test isa(ret, Gen.GenerativeFunctionCallNode)
@test ret.addr == :x
@test length(ret.inputs) == 3
@test isa(ret.inputs[1], Gen.JuliaNode) # () -> 0.5
@test isa(ret.inputs[1], Gen.JuliaNode) # () -> 0.5
@test isa(ret.inputs[2], Gen.JuliaNode) # () -> :y
@test ret.inputs[3] === i
at = ret.generative_function
@test isa(at, Gen.CallAtCombinator)
at2 = at.kernel
@test isa(at2, Gen.CallAtCombinator)
@test at2.kernel == foo

##########################
# traced diff annotation #
##########################

@gen (static, diffs) function bar(x)
    return x
end

@test Gen.get_track_diffs(typeof(bar))

@gen (static) function bar(x)
    return x
end

@test !Gen.get_track_diffs(typeof(bar))


end # @testset "static DSL"
