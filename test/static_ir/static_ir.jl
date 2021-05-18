using Gen: generate_generative_function

@testset "static IR" begin

counter = 0

#@gen (static, nojuliacache) function bar(y, b)
    #u = begin counter += 1; y + b end
    #v = @trace(normal(0, 1), :v)
    #z = u + v
    #return z
#end

builder = StaticIRBuilder()
y = add_argument_node!(builder, name=:y)
b = add_argument_node!(builder, name=:b)
zero = add_constant_node!(builder, 0.)
one = add_constant_node!(builder, 1.)
u = add_julia_node!(builder, (y, b) -> begin counter += 1; y + b end, inputs=[y, b], name=:u)
v = add_addr_node!(builder, normal, inputs=[zero, one], addr=:v, name=:v)
z = add_julia_node!(builder, (u, v) -> u + v, inputs=[u, v], name=:z)
set_return_node!(builder, z)
ir = build_ir(builder)
bar = eval(generate_generative_function(ir, :bar, track_diffs=false, cache_julia_nodes=false))
@test occursin("== Static IR ==", repr("text/plain", ir))

#@gen (static, nojuliacache) function foo(a, b)
    #@param theta::Float64
    #x = @trace(normal(0, 1), :x)
    #y = @trace(normal(x, a), :y)
    #z = @trace(bar(y, b), :z)
    #w = z + 1 + a + theta
    #return w
#end

builder = StaticIRBuilder()
a = add_argument_node!(builder, name=:a)
b = add_argument_node!(builder, name=:b)
theta = add_trainable_param_node!(builder, :theta, typ=QuoteNode(Float64))
zero = add_constant_node!(builder, 0.)
one = add_constant_node!(builder, 1.)
x = add_addr_node!(builder, normal, inputs=[zero, one], addr=:x, name=:x)
y = add_addr_node!(builder, normal, inputs=[x, a], addr=:y, name=:y)
z = add_addr_node!(builder, bar, inputs=[y, b], addr=:z, name=:z)
w = add_julia_node!(builder, (z, a, theta) -> z + 1 + a + theta, inputs=[z, a, theta], name=:w)
set_return_node!(builder, w)
ir = build_ir(builder)
foo = eval(generate_generative_function(ir, :foo, track_diffs=false, cache_julia_nodes=false))
@test occursin("== Static IR ==", repr("text/plain", ir))

theta_val = rand()
init_parameter!((foo, :theta), theta_val)

#@gen (static, nojuliacache) function const_fn()
    #return 1
#end

builder = StaticIRBuilder()
one = add_constant_node!(builder, 2)
set_return_node!(builder, one)
ir = build_ir(builder)
const_fn = eval(generate_generative_function(ir, :const_fn, track_diffs=false, cache_julia_nodes=false))
@test occursin("== Static IR ==", repr("text/plain", ir))

Gen.load_generated_functions()

@testset "Julia call" begin
    @test const_fn() == 2
end

function expected_score(a, b, x, y, v)
    (logpdf(normal, x, 0, 1) +
        logpdf(normal, y, x, a) +
        logpdf(normal, v, 0, 1))
end

@testset "simulate" begin
    a, b = 1, 2
    trace = simulate(foo, (a, b))
    x = get_choices(trace)[:x]
    y = get_choices(trace)[:y]
    v = get_choices(trace)[:z => :v]
    z = y + b + v

    @test isapprox(expected_score(a, b, x, y, v), get_score(trace))

    expected_retval = z + 1 + a + theta_val
    @test isapprox(expected_retval, get_retval(trace))
end

@testset "generate" begin
    a, b = 1, 2
    y = 1.123
    v = 2.345
    constraints = choicemap((:y, y), (:z => :v, v))
    (trace, weight) = generate(foo, (a, b), constraints)
    x = trace[:x] # filled in using ancestral sampling
    z = y + b + v

    @test get_choices(trace)[:y] == y
    @test get_choices(trace)[:z => :v] == v

    @test isapprox(expected_score(a, b, x, y, v), get_score(trace))

    proposal_score = logpdf(normal, x, 0, 1)
    expected_weight = get_score(trace)  - proposal_score
    @test isapprox(weight, expected_weight)
end

@testset "project" begin
    a, b = 1, 2
    y = 1.123
    v = 2.345
    x = 3.456
    constraints = choicemap((:y, y), (:z => :v, v), (:x, x))
    (trace, weight) = generate(foo, (a, b), constraints)
    weight = project(trace, select(:y, :z => :v))
    expected_weight = logpdf(normal, y, x, a) + logpdf(normal, v, 0, 1)
    @test isapprox(weight, expected_weight)
end

@testset "update without tracked diffs" begin

    # generate initial trace
    a = 1
    b = 2
    trace = simulate(foo, (a, b))
    x_init = get_choices(trace)[:x]
    y_init = get_choices(trace)[:y]
    v_init = get_choices(trace)[:z => :v]
    w_init = get_retval(trace)

    # case 1: diff(a) is NoChange(), :y is not constrained, diff(b) is NoChange()
    counter = 0
    (new_trace, weight, retdiff, discard) = update(trace, (a, b), (NoChange(), NoChange()), choicemap())
    @test get_choices(new_trace)[:x] == x_init
    @test get_choices(new_trace)[:y] == y_init
    @test get_choices(new_trace)[:z => :v] == v_init
    @test get_retval(new_trace) == w_init
    @test isapprox(weight, 0.)
    @test retdiff == NoChange()
    @test isempty(discard)
    @test counter == 0
    @test isapprox(get_score(new_trace), expected_score(a, b, x_init, y_init, v_init))

    # case 2: diff(a) is NoChange(), no constraints, diff(b) is UnknownChange()
    a_new = 3
    counter = 0
    (new_trace, weight, retdiff, discard) = update(trace, (a_new, b), (UnknownChange(), NoChange()), choicemap())
    @test get_choices(new_trace)[:x] == x_init
    @test get_choices(new_trace)[:y] == y_init
    @test get_choices(new_trace)[:z => :v] == v_init
    @test isapprox(get_retval(new_trace), w_init + a_new - a)
    @test isapprox(weight, logpdf(normal, y_init, x_init, a_new) - logpdf(normal, y_init, x_init, a))
    @test retdiff == UnknownChange() # because w depends on a
    @test isempty(discard)
    @test counter == 0 # y blocks the dependnece on b on a
    @test isapprox(get_score(new_trace), expected_score(a_new, b, x_init, y_init, v_init))

    # case 3: diff(a) is UnknownChange(), no constraints, diff(b) is NoChange()
    counter = 0
    (new_trace, weight, retdiff, discard) = update(trace, (a, b), (NoChange(), UnknownChange()), choicemap())
    @test get_choices(new_trace)[:x] == x_init
    @test get_choices(new_trace)[:y] == y_init
    @test get_choices(new_trace)[:z => :v] == v_init
    @test get_retval(new_trace) == w_init
    @test isapprox(weight, 0.)
    @test retdiff == UnknownChange() # because w depends on z which depends on b
    @test isempty(discard)
    @test counter == 1 # because the call to bar depends on b
    @test isapprox(get_score(new_trace), expected_score(a, b, x_init, y_init, v_init))

    # case 4: diff(a) is NoChange(), :y is constrained, diff(b) is NoChange()
    counter = 0
    y_new = 1.123
    (new_trace, weight, retdiff, discard) = update(trace, (a, b), (NoChange(), NoChange()), choicemap((:y, y_new)))
    @test get_choices(new_trace)[:x] == x_init
    @test get_choices(new_trace)[:y] == y_new
    @test get_choices(new_trace)[:z => :v] == v_init
    @test isapprox(get_retval(new_trace), w_init + y_new - y_init)
    @test isapprox(weight, logpdf(normal, y_new, x_init, a) - logpdf(normal, y_init, x_init, a))
    @test retdiff == UnknownChange() # because the call to bar()
    @test discard[:y] == y_init
    @test counter == 1 # because the call to bar depends on y, which was constrained
    @test isapprox(get_score(new_trace), expected_score(a, b, x_init, y_new, v_init))

    # case 5: diff(a) is NoChange(), diff(b) is NoChange(), constrain :z => :v
    counter = 0
    v_new = 2.345
    (new_trace, weight, retdiff, discard) = update(trace, (a, b), (NoChange(), NoChange()), choicemap((:z => :v, v_new)))
    @test get_choices(new_trace)[:x] == x_init
    @test get_choices(new_trace)[:y] == y_init
    @test get_choices(new_trace)[:z => :v] == v_new
    @test isapprox(get_retval(new_trace), w_init + v_new - v_init)
    @test isapprox(weight, logpdf(normal, v_new, 0, 1) - logpdf(normal, v_init, 0, 1))
    @test retdiff == UnknownChange() # because the call to bar()
    @test discard[:z => :v] == v_init
    @test counter == 1 # because the call to bar depends on y, which was constrained
    @test isapprox(get_score(new_trace), expected_score(a, b, x_init, y_init, v_new))
end

@testset "regenerate without tracked diffs" begin

    # generate initial trace
    a = 1
    b = 2
    trace = simulate(foo, (a, b))
    x_init = get_choices(trace)[:x]
    y_init = get_choices(trace)[:y]
    v_init = get_choices(trace)[:z => :v]
    w_init = get_retval(trace)

    # case 1: diff(a) is NoChange(), no selected choices, diff(b) is NoChange()
    counter = 0
    (new_trace, weight, retdiff) = regenerate(trace, (a, b), (NoChange(), NoChange()), select())
    @test get_choices(new_trace)[:x] == x_init
    @test get_choices(new_trace)[:y] == y_init
    @test get_choices(new_trace)[:z => :v] == v_init
    @test get_retval(new_trace) == w_init
    @test isapprox(weight, 0.)
    @test retdiff == NoChange()
    @test counter == 0
    @test isapprox(get_score(new_trace), expected_score(a, b, x_init, y_init, v_init))

    # case 2: diff(a) is NoChange(), no selected choices, diff(b) is UnknownChange()
    a_new = 3
    counter = 0
    (new_trace, weight, retdiff) = regenerate(trace, (a_new, b), (UnknownChange(), NoChange()), select())
    @test get_choices(new_trace)[:x] == x_init
    @test get_choices(new_trace)[:y] == y_init
    @test get_choices(new_trace)[:z => :v] == v_init
    @test isapprox(get_retval(new_trace), w_init + a_new - a)
    @test isapprox(weight, logpdf(normal, y_init, x_init, a_new) - logpdf(normal, y_init, x_init, a))
    @test retdiff == UnknownChange() # because w depends on a
    @test counter == 0 # y blocks the dependnece on b on a
    @test isapprox(get_score(new_trace), expected_score(a_new, b, x_init, y_init, v_init))

    # case 3: diff(a) is UnknownChange(), no selected choices, diff(b) is NoChange()
    counter = 0
    (new_trace, weight, retdiff) = regenerate(trace, (a, b), (NoChange(), UnknownChange()), select())
    @test get_choices(new_trace)[:x] == x_init
    @test get_choices(new_trace)[:y] == y_init
    @test get_choices(new_trace)[:z => :v] == v_init
    @test get_retval(new_trace) == w_init
    @test isapprox(weight, 0.)
    @test retdiff == UnknownChange() # because w depends on z which depends on b
    @test counter == 1 # because the call to bar depends on b
    @test isapprox(get_score(new_trace), expected_score(a, b, x_init, y_init, v_init))


    # case 4: diff(a) is NoChange(), :y is selected, diff(b) is NoChange()
    counter = 0
    (new_trace, weight, retdiff) = regenerate(trace, (a, b), (NoChange(), NoChange()), select(:y))
    y_new = get_choices(new_trace)[:y]
    @test y_new != y_init
    @test get_choices(new_trace)[:x] == x_init
    @test get_choices(new_trace)[:z => :v] == v_init
    @test isapprox(get_retval(new_trace), w_init + y_new - y_init)
    @test isapprox(weight, 0.)
    @test retdiff == UnknownChange() # because the call to bar()
    @test counter == 1 # because the call to bar depends on y, which was constrained
    @test isapprox(get_score(new_trace), expected_score(a, b, x_init, y_new, v_init))

    # case 5: diff(a) is NoChange(), diff(b) is NoChange(), select :z => :v
    counter = 0
    (new_trace, weight, retdiff) = regenerate(trace, (a, b), (NoChange(), NoChange()), select(:z => :v))
    v_new = get_choices(new_trace)[:z => :v]
    @test v_new != v_init
    @test get_choices(new_trace)[:x] == x_init
    @test get_choices(new_trace)[:y] == y_init
    @test isapprox(get_retval(new_trace), w_init + v_new - v_init)
    @test isapprox(weight, 0.)
    @test retdiff == UnknownChange() # because the call to bar()
    @test counter == 1 # because the call to bar depends on y, which was constrained
    @test isapprox(get_score(new_trace), expected_score(a, b, x_init, y_init, v_new))

end

include("gradients.jl")


# functions to test tracked diffs

# NOTE: this function will always return NoChange() for its retdiff

counter = 0

#@gen (static) function bar(a)
    #x = begin counter += 1; 0 end
    #return x
#end

builder = StaticIRBuilder()
a = add_argument_node!(builder, name=:a)
x = add_julia_node!(builder, () -> begin counter += 1; 0 end, inputs=[], name=:x)
set_return_node!(builder, x)
ir = build_ir(builder)
bar = eval(generate_generative_function(ir, :bar, track_diffs=false, cache_julia_nodes=false))

# this function will dynamically obtain NoChange() from the retdiff of bar(),
# which will cause its own retdiff to be NoChange():

#@gen (static) function foo(a)
    #x = @trace(bar(a), :x)
    #y = x + 1
    #return y
#end

builder = StaticIRBuilder()
a = add_argument_node!(builder, name=:a)
x = add_addr_node!(builder, bar, inputs=[a], addr=:x, name=:x)
y = add_julia_node!(builder, (x) -> x + 1, inputs=[x], name=:y)
set_return_node!(builder, y)
ir = build_ir(builder)

# generate a version of the function with tracked diffs
foo = eval(generate_generative_function(ir, :foo, track_diffs=true, cache_julia_nodes=false))

# generate a version of the function without tracked diffs
foo_without_tracked_diffs = eval(generate_generative_function(ir, :foo, track_diffs=false, cache_julia_nodes=false))

Gen.load_generated_functions()

@testset "update with tracked diffs" begin

    # generate initial trace from function with tracked diffs
    a = 1
    trace = simulate(foo, (a,))

    # case 1: diff(a) is NoChange, it will statically avoid running bar()
    counter = 0
    (new_trace, weight, retdiff, _) = update(trace, (a,), (NoChange(),), choicemap())
    @test counter == 0
    @test retdiff == NoChange()

    # case 2: diff(a) is UnknownChange, it will run bar(), but still return retdiff of NoChange()
    counter = 0
    (new_trace, weight, retdiff, _) = update(trace, (a,), (UnknownChange(),), choicemap())
    @test counter == 1
    @test retdiff == NoChange()

    # generate initial trace from function without tracked diffs
    a = 1
    trace = simulate(foo_without_tracked_diffs, (a,))

    # case 1: diff(a) is NoChange, it will statically avoid running bar()
    counter = 0
    (new_trace, weight, retdiff, _) = update(trace, (a,), (NoChange(),), choicemap())
    @test counter == 0
    @test retdiff == NoChange()

    # case 2: diff(a) is UnknownChange, it will run bar()
    counter = 0
    (new_trace, weight, retdiff, _) = update(trace, (a,), (UnknownChange(),), choicemap())
    @test counter == 1
    @test retdiff == UnknownChange()

end

@testset "regenerate with tracked diffs" begin

    # generate initial trace from function with tracked diffs
    a = 1
    trace = simulate(foo, (a,))

    # case 1: diff(a) is NoChange, it will statically avoid running bar()
    counter = 0
    (new_trace, weight, retdiff) = regenerate(trace, (a,), (NoChange(),), select())
    @test counter == 0
    @test retdiff == NoChange()

    # case 2: diff(a) is UnknownChange, it will run bar(), but still return retdiff of NoChange()
    counter = 0
    (new_trace, weight, retdiff) = regenerate(trace, (a,), (UnknownChange(),), select())
    @test counter == 1
    @test retdiff == NoChange()

    # generate initial trace from function without tracked diffs
    a = 1
    trace = simulate(foo_without_tracked_diffs, (a,))

    # case 1: diff(a) is NoChange, it will statically avoid running bar()
    counter = 0
    (new_trace, weight, retdiff) = regenerate(trace, (a,), (NoChange(),), select())
    @test counter == 0
    @test retdiff == NoChange()

    # case 2: diff(a) is UnknownChange, it will run bar()
    counter = 0
    (new_trace, weight, retdiff) = regenerate(trace, (a,), (UnknownChange(),), select())
    @test counter == 1
    @test retdiff == UnknownChange()
end

counter = 0

#@gen (static) function foo(a, b)
    #u = begin a + b; counter += 1 end
    #x = @trace(normal(u, 1), :x)
    #return x
#end

builder = StaticIRBuilder()
a = add_argument_node!(builder, name=:a)
b = add_argument_node!(builder, name=:b)
u = add_julia_node!(builder, (a, b) -> begin counter += 1; a + b end, inputs=[a, b], name=:u)
one = add_constant_node!(builder, 1.)
x = add_addr_node!(builder, normal, inputs=[u, one], addr=:x, name=:x)
set_return_node!(builder, x)
ir = build_ir(builder)
foo = eval(generate_generative_function(ir, :foo, track_diffs=false, cache_julia_nodes=true))

Gen.load_generated_functions()

@testset "cached julia nodes" begin

    counter = 0
    trace = simulate(foo, (1, 2))
    @test counter == 1

    counter = 0
    update(trace, (1, 2), (NoChange(), NoChange()), choicemap((:x, 1.0)))
    @test counter == 0 # would be 1 if nojuliacache was used

    counter = 0
    update(trace, (1, 2), (NoChange(), NoChange()), choicemap((:x, 1.0)))
    @test counter == 0 # would be 1 if nojuliacache was used

    counter = 0
    regenerate(trace, (1, 2), (NoChange(), NoChange()), select(:x))
    @test counter == 0 # would be 1 if nojuliacache was used

    counter = 0
    update(trace, (1, 3), (NoChange(), UnknownChange()), choicemap((:x, 1.0)))
    @test counter == 1

    counter = 0
    update(trace, (1, 3), (NoChange(), UnknownChange()), choicemap((:x, 1.0)))
    @test counter == 1

    counter = 0
    regenerate(trace, (1, 3), (NoChange(), UnknownChange()), select(:x))
    @test counter == 1
end

@testset "regression test for https://github.com/probcomp/Gen/issues/168" begin
    @gen (static) function model(var)
        mean = @trace(normal(0, 1), :mean)
        T = @trace(normal(mean, var), :T)
        return T
    end
    load_generated_functions()
    selection = StaticSelection(select(:mean))
    (tr, _) = generate(model, (1,))
    # At the time the issue was filed, this line produced a crash
    (tr, ) = mh(tr, selection)
end

end # @testset "static IR"
