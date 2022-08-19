Base.show(io::IO, ::MIME"text/plain", ir::StaticIR) =
    print_ir(io, ir)

function print_ir(io::IO, ir::StaticIR)
    println(io, "== Static IR ==")
    args = join((string(arg.name) for arg in ir.arg_nodes), ", ")
    println(io, "Arguments: ($args)")
    for node in ir.nodes
        node in ir.arg_nodes && continue
        print(io, "  "); print_ir(io, node); println(io)
    end
    print(io, "  return $(ir.return_node.name)")
end

function print_ir(io::IO, node::TrainableParameterNode)
    print(io, "@param $(node.name)::$(node.typ)")
end

function print_ir(io::IO, node::JuliaNode)
    inputs = join((string(i.name) for i in node.inputs), ", ")
    print(io, "$(node.name) = $(node.fn)($inputs)")
end

function print_ir(io::IO, node::GenerativeFunctionCallNode)
    inputs = join((string(i.name) for i in node.inputs), ", ")
    gen_fn_name = ir_name(node.generative_function)
    print(io, "$(node.name) = @trace($(gen_fn_name)($inputs), :$(node.addr))")
end

ir_name(fn::GenerativeFunction) = nameof(typeof(fn))
ir_name(fn::DynamicDSLFunction) = nameof(fn.julia_function)
