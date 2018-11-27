using Gen
using Gen: ArgumentNode, RandomChoiceNode, GenerativeFunctionCallNode, ConstantNode, RegularNode, StaticIR, generate_trace_struct

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
ir = StaticIR(arg_nodes, choice_nodes, call_nodes, return_node)

# generate a trace type
trace_defn = generate_trace_struct(ir, :foo)
println(trace_defn)
eval(trace_defn)
