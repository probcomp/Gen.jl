module SwitchComb

include("../src/Gen.jl")
using .Gen

@gen (grad) function foo((grad)(x::Float64), (grad)(y::Float64))
    std::Float64 = 3.0
    z = @trace(normal(x + y, std), :z)
    return z
end

@gen (grad) function baz((grad)(x::Float64), (grad)(y::Float64))
    std::Float64 = 3.0
    z = @trace(normal(x + 2 * y, std), :z)
    return z
end

sc = Switch(foo, baz)
chm, _, _ = propose(sc, (0.3, 5.0, 3.0))
display(chm)

tr = simulate(sc, (0.3, 5.0, 3.0))
display(get_choices(tr))

chm = choicemap()
chm[:cond] = true
tr, _ = generate(sc, (0.3, 5.0, 3.0), chm)
display(get_choices(tr))

end # module
