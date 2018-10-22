#######################
# parsing AST into IR #
#######################

struct BasicBlockParseError <: Exception
    expr::Any
end

function parse_addr(expr::Expr)
    @assert expr.head == :macrocall && expr.args[1] == Symbol("@addr")
    line::LineNumberNode = expr.args[2]
    rest = expr.args[3:end]
    if length(rest) == 2
        change_expr = nothing
    elseif length(rest) == 3
        change_expr = rest[3]
    else
        throw(BasicBlockParseError(expr))
    end
    call = rest[1]
    if !isa(call, Expr) || call.head != :call
        throw(BasicBlockParseError(call))
    end
    generator_or_dist = Main.eval(call.args[1])
    if isa(generator_or_dist, Distribution) && change_expr != nothing
        error("Cannot pass change values to @addr for distributions")
    end
    args = call.args[2:end]
    addr = rest[2]
    if isa(addr, Expr) && length(addr.args) == 1 && isa(addr.args[1], Symbol)
        addr = addr.args[1]
    elseif isa(addr, QuoteNode)
        addr = addr.value
    end
    (addr::Symbol, generator_or_dist, args, change_expr, line)
end

function parse_change(expr::Expr)   
    @assert expr.head == :macrocall && expr.args[1] == Symbol("@change")
    rest = isa(expr.args[2], LineNumberNode) ? expr.args[3:end] : expr.args[2:end]
    if length(rest) != 1
        throw(BasicBlockParseError(expr))
    end
    addr = rest[1]
    if isa(addr, Symbol)
        pass
    elseif isa(addr, Expr) && length(addr.args) == 1 && isa(addr.args[1], Symbol)
        addr = addr.args[1]
    elseif isa(addr, QuoteNode)
        addr = addr.value
    else
        throw(BasicBlockParseError(addr))
    end
    addr::Symbol
end

function parse_lhs(lhs::Expr)
    if lhs.head == :(::)
        name = lhs.args[1]
        typ = Main.eval(lhs.args[2])
        return (name, typ)
    else
        throw(BasicBlockParseError(lhs))
    end
end

function parse_lhs(lhs::Symbol)
    name = lhs
    (name, Any)
end

function parse_param(expr)
    @assert expr.head == :macrocall && expr.args[1] == Symbol("@param")
    rest = isa(expr.args[2], LineNumberNode) ? expr.args[3:end] : expr.args[2:end]
    if length(rest) != 1
        throw(BasicBlockParseError(rest))
    end
    decl = rest[1]
    if isa(decl, Symbol)
        name = decl
        typ = Any
    elseif isa(decl, Expr) && decl.head == :(::)
        name = decl.args[1]
        typ = Main.eval(decl.args[2])
    else
        throw(BasicBlockParseError(rest[1]))
    end
    (name, typ)
end


function generate_ir(args, body, output_ad, args_ad)
    ir = DataFlowIR(output_ad, args_ad)
    if !isa(body, Expr) || body.head != :block
        throw(BasicBlockParseError(body))
    end
    for arg in args
        if isa(arg, Symbol)
            name = arg
            typ = Any
        elseif isa(arg, Expr) && arg.head == :(::)
            name = arg.args[1]
            typ = Main.eval(arg.args[2])
        else
            throw(BasicBlockParseError(body))
        end
        add_argument!(ir, name, typ)
    end
    local line::LineNumberNode
    for statement in body.args
        if isa(statement, LineNumberNode)
            line = statement
            continue
        end
        if !isa(statement, Expr)
            throw(BasicBlockParseError(statement))
        end
        if statement.head == :macrocall && statement.args[1] == Symbol("@param")
            (name, typ) = parse_param(statement)
            add_param!(ir, name, typ)
        elseif statement.head == :macrocall && statement.args[1] == Symbol("@addr")
            # an @addr statement without a left-hand-side
            (addr, dist_or_gen, args, change_expr, line) = parse_addr(statement)
            if isa(dist_or_gen, Distribution)
                @assert change_expr == nothing
                add_addr!(ir, addr, line, dist_or_gen, args)
            else
                # change_expr may be nothing, indicating nothing is known
                add_addr!(ir, addr, line, dist_or_gen, args, change_expr)
            end
        elseif statement.head == :(=)
            lhs = statement.args[1]
            rhs = statement.args[2]
            (name, typ) = parse_lhs(lhs)
            if rhs.head == :macrocall && rhs.args[1] == Symbol("@addr")
                (addr, dist_or_gen, args, change_expr, line) = parse_addr(rhs)
                if isa(dist_or_gen, Distribution)
                    @assert change_expr == nothing
                    add_addr!(ir, addr, line, dist_or_gen, args, typ, name)
                else
                    # change_expr may be nothing, indicating nothing is known
                    add_addr!(ir, addr, line, dist_or_gen, args, typ, name, change_expr)
                end
            elseif rhs.head == :macrocall && rhs.args[1] == Symbol("@argschange")
                if (length(rhs.args) == 2 && isa(rhs.args[2], LineNumberNode)) || length(rhs.args) == 1
                    add_argschange!(ir, typ, name)
                else
                    dump(rhs)
                    throw(BasicBlockParseError(statement))
                end
            elseif rhs.head == :macrocall && rhs.args[1] == Symbol("@change")
                addr = parse_change(rhs)
                add_change!(ir, addr, typ, name)
            else
                add_julia!(ir, rhs, typ, name, line)
            end
        elseif statement.head == :return
            if length(statement.args) != 1
                throw(BasicBlockParseError(statement))
            end
            expr = statement.args[1]
            if isa(expr, Symbol)
                set_return!(ir, expr)
            elseif isa(expr, Expr) && expr.head == :(::)
                expr_value = expr.args[1]
                typ = Main.eval(expr.args[2])
                set_return!(ir, expr_value, typ)
            else
                throw(BasicBlockParseError(statement))
            end
        elseif statement.head == :macrocall && statement.args[1] == Symbol("@retchange")
            if length(statement.args) != 2
                throw(BasicBlockParseError(statement))
            end
            expr = statement.args[2]
            if isa(expr, Symbol)
                set_retchange!(ir, expr)
            elseif isa(expr, Expr) && expr.head == :(::)
                expr_value = expr.args[1]
                typ = Main.eval(expr.args[2])
                set_retchange!(ir, expr_value, typ)
            else
                throw(BasicBlockParseError(statement))
            end
        else
            throw(BasicBlockParseError(statement))
        end
    end
    finish!(ir)
    ir
end

function compiled_gen_parse_inner(ast)
    @assert isa(ast, Expr) && ast.head == :macrocall && ast.args[1] == Symbol("@gen")
    ast = isa(ast.args[2], LineNumberNode) ? ast.args[3] : ast.args[2]
    if ast.head != :function
        error("syntax error in @compiled at $(ast) in $(ast.head)")
    end
    if length(ast.args) != 2
        error("syntax error in @compile at $(ast) in $(ast.args)")
    end
    signature = ast.args[1]
    body = ast.args[2]
    if signature.head != :call
        error("syntax error in @compiled at $(ast) in $(signature)")
    end
    name = signature.args[1]
    args = signature.args[2:end]
    has_argument_grads = (map(marked_for_ad, args)...,)
    args = map(strip_marked_for_ad, args)
    (name, args, body, has_argument_grads)
end

function compiled_gen_parse(ast)
    if ast.head == :macrocall && ast.args[1] == Symbol("@gen")
        (name, args, body, args_ad) = compiled_gen_parse_inner(ast)
        return (name, args, body, false, args_ad)
    elseif ast.head == :macrocall && ast.args[1] == Symbol("@ad")
        sub_ast = isa(ast.args[2], LineNumberNode) ? ast.args[3] : ast.args[2]
        if (isa(sub_ast, Expr)
            && sub_ast.head == :macrocall
            && sub_ast.args[1] == Symbol("@gen"))
            (name, args, body, args_ad) = compiled_gen_parse_inner(sub_ast)
            return (name, args, body, true, args_ad)
        end
    end 
    error("syntax error in @compiled, expected @compiled @gen .. or @compiled @ad @gen ..")
end


##########################################
# macro-expand into generator definition #
##########################################

macro ssa(ast)

    # NOTE: must be called at top-level (since it defines a new data type)

    # parse AST
    (name, args, body, output_ad, args_ad) = compiled_gen_parse(ast)

    # geneate data-flow intermediate representation (IR)
    ir = generate_ir(args, body, output_ad, args_ad)

    # generate code from IR
    (gradient_function_defns, node_to_gradient) = generate_gradient_functions(ir)
    (trace_type_defn, trace_type) = generate_trace_data_type(ir, name)
    (generator_type_defn, generator_type) = generate_generator_type(
        ir, trace_type, name, node_to_gradient)

    # construct generator, assign to name
    Expr(:block,
        trace_type_defn,
        generator_type_defn,
        gradient_function_defns...,
        quote $(esc(name)) = $(esc(generator_type))() end)
end

export @ssa
