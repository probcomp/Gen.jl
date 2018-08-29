const state = gensym("state")

macro addr(expr::Expr, addr)
    if expr.head != :call
        error("syntax error in @addr at $(expr)")
    end
    fn = esc(expr.args[1])
    args = map(esc, expr.args[2:end])
    Expr(:call, :addr, esc(state), fn, Expr(:tuple, args...), esc(addr))
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
