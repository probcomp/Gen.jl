label(node::ArgumentNode) = String(node.name)
label(node::JuliaNode) = String(node.name)
function label(node::GenerativeFunctionCallNode)
    if node.generative_function isa Distribution
        "$(node.generative_function) $(node.addr) $(node.name)"
    else
        "$(node.addr) $(node.name)"
    end
end

function draw_graph(ir::StaticIR, graphviz, fname)
    dot = graphviz.Digraph()
    nodes_to_name = Dict{StaticIRNode,String}()
    for node in ir.nodes
        nodes_to_name[node] = label(node)
    end
    for node in ir.nodes
        if isa(node, ArgumentNode)
            shape = "diamond"
            color = "white"
            parents = []
        elseif isa(node, GenerativeFunctionCallNode) && node.generative_function isa Distribution
            shape = "ellipse"
            color = "white"
            parents = node.inputs
        elseif isa(node, GenerativeFunctionCallNode)
            shape = "star"
            color = "white"
            parents = node.inputs
        elseif isa(node, JuliaNode)
            shape = "box"
            color = "white"
            parents = values(node.inputs)
        end
        if node === ir.return_node
            color = "lightblue"
        end
        dot[:node](nodes_to_name[node], nodes_to_name[node], shape=shape, fillcolor=color, style="filled")
        for parent in parents
            dot[:edge](nodes_to_name[parent], nodes_to_name[node])
        end
    end
    dot[:render](fname, view=true)
end

function draw_graph(gen_fn::StaticIRGenerativeFunction, graphviz, fname)
    draw_graph(get_ir(typeof(gen_fn)), graphviz, fname)
end

export draw_graph
