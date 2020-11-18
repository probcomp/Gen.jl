module SwitchComb

include("../src/Gen.jl")
using .Gen

# ------------ Toplevel caller ------------ #

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

sc = Switch(Dict(:x => 1, :y => 2), foo, baz)
chm, _, _ = propose(sc, (2, 5.0, 3.0))
display(chm)

tr = simulate(sc, (2, 5.0, 3.0))
display(get_choices(tr))

chm = choicemap()
chm[:z] = 5.0
tr, _ = generate(sc, (2, 5.0, 3.0), chm)
display(get_choices(tr))

# ------------ Static DSL ------------ #

@gen (static) function bam(s::Symbol)
    x ~ sc(s, 5.0, 3.0)
end
Gen.@load_generated_functions()

tr = simulate(bam, (:x, ))
display(get_choices(tr))

end # module
