const state = gensym("state")

function parse_dsl_fn_signature(ast, dsl, constructor::Symbol)
    if ast.head != :function
        error("syntax error in $dsl at $(ast) in $(ast.head)")
    end
    if length(ast.args) != 2
        error("syntax error in $dsl at $(ast) in $(ast.args)")
    end
    signature = ast.args[1]
    body = ast.args[2]
    if signature.head != :call
        error("syntax error in $dsl at $(ast) in $(signature)")
    end
    function_name = signature.args[1]
    args = signature.args[2:end]
    escaped_args = map(esc, args)
    fn_args = [esc(state), escaped_args...]
    Expr(:call, constructor,
        Expr(:function, Expr(:tuple, fn_args...), esc(body)))
end

macro addr(expr::Expr, addr)
    if expr.head != :call
        error("syntax error in @addr at $(expr)")
    end
    fn = esc(expr.args[1])
    args = map(esc, expr.args[2:end])
    Expr(:call, :addr, esc(state), fn, Expr(:tuple, args...), esc(addr))
end

macro addr(expr::Expr, addr, delta)
    if expr.head != :call
        error("syntax error in @addr at $(expr)")
    end
    fn = esc(expr.args[1])
    args = map(esc, expr.args[2:end])
    Expr(:call, :addr, esc(state), fn, Expr(:tuple, args...), esc(addr), esc(delta))
end

macro splice(expr::Expr)
    if expr.head != :call
        error("syntax error in @splice at $(expr)")
    end
    invocation = expr.args[1]
    args = esc(Expr(:tuple, expr.args[2:end]...))
    Expr(:call, :splice, esc(state), esc(invocation), args)
end

macro read(addr)
    Expr(:call, :read, esc(state), esc(addr))
end

export @addr
export @read
export @splice
