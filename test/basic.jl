@testset "basic block" begin

eval(:(@gen function bar()
    if @addr(bernoulli(0.4), :branch)
        @addr(normal(0, 1), :x)
    else
        @addr(normal(0, 1), :y)
    end
end))

eval(:(@compiled @gen function foo()
    @addr(bar(), :a)
    @addr(normal(0, 1), :b)
    @addr(normal(0, 1), :c)
end))

Gen.load_generated_functions()


############
## update ##
############

@testset "update" begin

# get a trace which follows the first branch
constraints = DynamicAssignment()
constraints[:a => :branch] = true
(trace,) = generate(foo, (), constraints)
x = get_assignment(trace)[:a => :x]
b = get_assignment(trace)[:b]
c = get_assignment(trace)[:c]

# force to follow the second branch
b_new = 0.123
constraints = DynamicAssignment()
constraints[:a => :branch] = false
constraints[:b] = b_new
constraints[:a => :y] = 2.3
(new_trace, weight, discard, retchange) = update(
    foo, (), nothing, trace, constraints)

# test discard
@test get_leaf_node(discard, :a => :branch) == true
@test get_leaf_node(discard, :b) == b
@test get_leaf_node(discard, :a => :x) == x
@test length(collect(get_leaf_nodes(discard))) == 1
@test length(collect(get_internal_nodes(discard))) == 1

# test new trace
new_assignment = get_assignment(new_trace)
@test get_leaf_node(new_assignment, :a => :branch) == false
@test get_leaf_node(new_assignment, :b) == b_new
@test length(collect(get_leaf_nodes(new_assignment))) == 2
@test length(collect(get_internal_nodes(new_assignment))) == 1
y = new_assignment[:a => :y]

# test score and weight
prev_score = (
    logpdf(bernoulli, true, 0.4) +
    logpdf(normal, x, 0, 1) +
    logpdf(normal, b, 0, 1) +
    logpdf(normal, c, 0, 1))
expected_new_score = (
    logpdf(bernoulli, false, 0.4) +
    logpdf(normal, y, 0, 1) +
    logpdf(normal, b_new, 0, 1) +
    logpdf(normal, c, 0, 1))
expected_weight = expected_new_score - prev_score
@test isapprox(expected_new_score, get_call_record(new_trace).score)
@test isapprox(expected_weight, weight)

# test retchange (should be nothing by default)
@test retchange === nothing

end


################
## fix_update ##
################

@testset "fix_update" begin

# get a trace which follows the first branch
constraints = DynamicAssignment()
constraints[:a => :branch] = true
(trace,) = generate(foo, (), constraints)
x = get_assignment(trace)[:a => :x]
b = get_assignment(trace)[:b]
c = get_assignment(trace)[:c]

# force to follow the second branch and change b
b_new = 0.123
constraints = DynamicAssignment()
constraints[:a => :branch] = false
constraints[:b] = b_new
(new_trace, weight, discard, retchange) = fix_update(
    foo, (), nothing, trace, constraints)

# test discard
@test get_leaf_node(discard, :a => :branch) == true
@test get_leaf_node(discard, :b) == b
@test length(collect(get_leaf_nodes(discard))) == 1
@test length(collect(get_internal_nodes(discard))) == 1

# test new trace
new_assignment = get_assignment(new_trace)
@test get_leaf_node(new_assignment, :a => :branch) == false
@test get_leaf_node(new_assignment, :b) == b_new
@test length(collect(get_leaf_nodes(new_assignment))) == 2
@test length(collect(get_internal_nodes(new_assignment))) == 1
y = new_assignment[:a => :y]

# test score and weight
expected_new_score = (
    logpdf(bernoulli, false, 0.4) +
    logpdf(normal, y, 0, 1) +
    logpdf(normal, b_new, 0, 1) +
    logpdf(normal, c, 0, 1))
expected_weight = logpdf(bernoulli, false, 0.4) - logpdf(bernoulli, true, 0.4) + logpdf(normal, b_new, 0, 1) - logpdf(normal, b, 0, 1)
@test isapprox(expected_new_score, get_call_record(new_trace).score)
@test isapprox(expected_weight, weight)

# test retchange (should be nothing by default)
@test retchange === nothing

end

end

