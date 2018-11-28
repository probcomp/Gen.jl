using PyCall
@pyimport graphviz as gv

label(node::ArgumentNode) = String(node.name)
label(node::JuliaNode) = String(node.name)
label(node::RandomChoiceNode) = "$(node.dist) $(node.addr) $(node.name)"
label(node::GenerativeFunctionCallNode) = "$(typeof(node.generative_function)) $(node.addr) $(node.name)"
label(node::DiffJuliaNode) = String(node.name)
label(node::ReceivedArgDiffNode) = String(node.name)
label(node::ChoiceDiffNode) = "$(node.choice_node.addr) $(node.name)"
label(node::CallDiffNode) = "$(node.call_node.addr) $(node.name)"

function render_graph(ir::StaticIR, fname)
    dot = gv.Digraph()
    nodes_to_name = Dict{StaticIRNode,String}()
    for node in ir.nodes
        nodes_to_name[node] = label(node)
    end
    for node in ir.nodes
        if isa(node, ArgumentNode)
            shape = "diamond"
            color = "white"
            parents = []
        elseif isa(node, RandomChoiceNode)
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
        elseif isa(node, DiffJuliaNode)
            shape = "box"
            color = "red"
            parents = values(node.inputs)
        elseif isa(node, ReceivedArgDiffNode)
            shape = "diamond"
            color = "red"
            parents = []
        elseif isa(node, ChoiceDiffNode)
            shape = "circle"
            color = "red"
            parents = [node.choice_node]
        elseif isa(node, CallDiffNode)
            shape = "star"
            color = "red"
            parents = [node.call_node]
        end
        if node === ir.return_node
            color = "lightblue"
        end
        if node === ir.retdiff_node
            color = "orange"
        end
        dot[:node](nodes_to_name[node], nodes_to_name[node], shape=shape, fillcolor=color, style="filled")
        for parent in parents
            dot[:edge](nodes_to_name[parent], nodes_to_name[node])
        end
    end
    dot[:render](fname, view=true)
end
