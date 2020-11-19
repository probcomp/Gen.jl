module SwitchComb

include("../../src/Gen.jl")
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

# Standard.
sc = Switch(foo, baz)
chm, _, _ = propose(sc, (2, 5.0, 3.0))

tr = simulate(sc, (2, 5.0, 3.0))

chm = choicemap()
chm[:z] = 5.0
tr, _ = generate(sc, (2, 5.0, 3.0), chm)

# Cases.
sc = Switch(Dict(:x => 1, :y => 2), foo, baz)
chm, _, _ = propose(sc, (:x, 5.0, 3.0))

tr = simulate(sc, (:x, 5.0, 3.0))

chm = choicemap()
chm[:z] = 5.0
tr, _ = generate(sc, (:x, 5.0, 3.0), chm)

# ------------ Static DSL ------------ #

@gen (static) function bang((grad)(x::Float64), (grad)(y::Float64))
    std::Float64 = 3.0
    z = @trace(normal(x + y, std), :z)
    return z
end

@gen (static) function fuzz((grad)(x::Float64), (grad)(y::Float64))
    std::Float64 = 3.0
    z = @trace(normal(x + 2 * y, std), :z)
    return z
end

sc = Switch(bang, fuzz)

@gen (static) function bam(s::Int)
    x ~ sc(s, 5.0, 3.0)
    return x
end
Gen.@load_generated_functions()

tr = simulate(bam, (1, ))

chm = choicemap((:x => :z, 5.0))
new_tr, w, rd, discard = update(tr, (2, ), (UnknownChange(), ), chm)
display(discard)
display(get_choices(new_tr))

new_tr, w = regenerate(tr, (1, ), (UnknownChange(), ), select())

sel = select(:x => :z)
arg_grads, cvs, cgs = choice_gradients(tr, sel, 1.0)
display(arg_grads)
display(cgs)

end # module
