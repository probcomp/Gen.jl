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

marked_for_ad(arg::Symbol) = false

marked_for_ad(arg::Expr) = (arg.head == :macrocall && arg.args[1] == Symbol("@ad"))

strip_marked_for_ad(arg::Symbol) = arg

function strip_marked_for_ad(arg::Expr) 
    if (arg.head == :macrocall && arg.args[1] == Symbol("@ad"))
		if length(arg.args) == 3 && isa(arg.args[2], LineNumberNode)
			arg.args[3]
		elseif length(arg.args) == 2
			arg.args[2]
		else
            error("Syntax error at $arg")
        end
    else
        arg
    end
end

export @addr
export @read
export @splice
