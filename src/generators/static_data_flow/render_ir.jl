using PyCall
@pyimport graphviz as gv

function render_graph(ir::DataFlowIR, fname)
    # graphviz
    dot = gv.Digraph() # comment = ?
    nodes_to_name = Dict{Node,String}()
    for node in ir.all_nodes
        io = IOBuffer()
        print(io, node)
        nodes_to_name[node] = String(take!(io))
    end
    for node in ir.all_nodes
        if isa(node, ValueNode)
            shape = "box"
        else
            shape = "ellipse"
        end
        if node in ir.incremental_nodes
            color = "lightblue"
        else
            color = "lightgray"
        end
        if ir.output_node == node
            @assert !(node in ir.incremental_nodes)
            color = "firebrick1"
        end
        if ir.retchange_node == node
            color = "darkolivegreen2"
        end
        dot[:node](nodes_to_name[node], nodes_to_name[node], shape=shape, color=color, style="filled")
        for parent in parents(node)
            dot[:edge](nodes_to_name[parent], nodes_to_name[node])
        end
    end
    for node in values(ir.addr_gen_nodes)
        dot[:edge](nodes_to_name[node.change_node], nodes_to_name[node], style="dashed")
    end
    dot[:render](fname, view=true)
end
