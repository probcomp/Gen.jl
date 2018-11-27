using Gen
using Gen: ArgumentNode, RandomChoiceNode, GenerativeFunctionCallNode, RegularNode, JuliaNode, ReceivedArgDiffNode, StaticIRNode, StaticIR, generate_trace_type_and_methods, generate_generative_function

@gen function callee(a, b)
    return a + b + 1.0
end

# construct a static IR
arg_nodes = [ArgumentNode(:arg1, Float64), ArgumentNode(:arg2, Any)]
constant_node_1 = JuliaNode(:(nothing), Dict{Symbol,RegularNode}(), gensym("a"), Nothing) 
constant_node_2 = JuliaNode(:(nothing), Dict{Symbol,RegularNode}(), gensym("a"), Nothing) 
choice_nodes = [
    RandomChoiceNode(normal, RegularNode[arg_nodes[1], arg_nodes[2]],
        :choice1, :name1, Float64),
    RandomChoiceNode(normal, RegularNode[arg_nodes[1], arg_nodes[2]],
        :choice2, :name2, Any)
]
call_nodes = [
    GenerativeFunctionCallNode(callee, RegularNode[arg_nodes[1], choice_nodes[2]], :call1,
        constant_node_1, :name3, Float64)
]
return_node = choice_nodes[1]
retdiff_node = constant_node_2
received_argdiff_node = ReceivedArgDiffNode(:argdiff, Nothing)
nodes = StaticIRNode[]
append!(nodes, arg_nodes)
push!(nodes, constant_node_1)
push!(nodes, constant_node_2)
append!(nodes, choice_nodes)
append!(nodes, call_nodes)
push!(nodes, retdiff_node)
push!(nodes, received_argdiff_node)
ir = StaticIR(nodes, arg_nodes, choice_nodes, call_nodes, return_node, retdiff_node, received_argdiff_node)

# generate a trace type
(trace_defn, trace_struct_name) = generate_trace_type_and_methods(ir, :foo)
gen_fn_defn = generate_generative_function(ir, :foo, trace_struct_name)

println(trace_defn)
eval(trace_defn)
println(gen_fn_defn)
eval(gen_fn_defn)

println(foo)
println(typeof(foo))

Gen.load_generated_functions()

constraints = DynamicAssignment()
constraints[:choice1] = 2.1
static_constraints = StaticAssignment(constraints)
code = Gen.codegen_generate(typeof(foo), Tuple{Float64,Float64}, typeof(static_constraints))
println(code)

(trace, weight) = generate(foo, (1.2, 2.5), static_constraints)
println(trace)
println(weight)
