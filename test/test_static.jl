using Gen
using Gen: ArgumentNode, RandomChoiceNode, GenerativeFunctionCallNode, ConstantNode, RegularNode, ReceivedArgDiffNode, StaticIRNode, StaticIR, generate_trace_type_and_methods, generate_generative_function

@gen function callee(a, b)
    return a + b + 1.0
end

# construct a static IR
arg_nodes = [ArgumentNode(:arg1, Float64), ArgumentNode(:arg2, Any)]
choice_nodes = [
    RandomChoiceNode(normal, RegularNode[arg_nodes[1], arg_nodes[2]],
        :choice1, :name1, Float64),
    RandomChoiceNode(normal, RegularNode[arg_nodes[1], arg_nodes[2]],
        :choice2, :name2, Any)
]
call_nodes = [
    GenerativeFunctionCallNode(callee, RegularNode[arg_nodes[1]], :call1,
        ConstantNode(nothing), :name3, Float64)
]
return_node = choice_nodes[1]
retdiff_node = ConstantNode(nothing)
received_argdiff_node = ReceivedArgDiffNode(:argdiff, Nothing)
nodes = StaticIRNode[]
append!(nodes, arg_nodes)
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
