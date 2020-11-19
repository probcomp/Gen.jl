@testset "switch combinator" begin

    # ------------ Trace ------------ #

    @gen function swtrg()
        z ~ normal(3.0, 5.0)
        return z
    end

    @testset "switch trace" begin
        tr = simulate(swtrg, ())
        swtr = Gen.SwitchTrace(swtrg, 1, tr, get_retval(tr), (), get_score(tr), 0.0)
        @test swtr[:z] == tr[:z]
        @test project(swtr, AllSelection()) == project(swtr.branch, AllSelection())
        @test project(swtr, EmptySelection()) == swtr.noise
    end

    # ------------ Bare combinator ------------ #

    # Model chunk.
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
    # ----.

    @testset "simulate" begin
        tr = simulate(sc, (1, 5.0, 3.0))
        @test isapprox(get_score(tr), logpdf(normal, tr[:z], 5.0 + 3.0, 3.0))
        tr = simulate(sc, (2, 5.0, 3.0))
        @test isapprox(get_score(tr), logpdf(normal, tr[:z], 5.0 + 2 * 3.0, 3.0))
    end

    @testset "generate" begin
        chm = choicemap()
        chm[:z] = 5.0
        tr, w = generate(sc, (2, 5.0, 3.0), chm)
        assignment = get_choices(tr)
        @test assignment[:z] == 5.0
        @test isapprox(w, logpdf(normal, 5.0, 5.0 + 2 * 3.0, 3.0))
    end

    @testset "assess" begin
        chm = choicemap()
        chm[:z] = 5.0
        w, ret = assess(sc, (2, 5.0, 3.0), chm)
        @test isapprox(w, logpdf(normal, 5.0, 5.0 + 2 * 3.0, 3.0))
    end

    @testset "propose" begin
        chm, w = propose(sc, (2, 5.0, 3.0))
        @test isapprox(w, logpdf(normal, chm[:z], 5.0 + 2 * 3.0, 3.0))
    end

    @testset "update" begin
        tr = simulate(sc, (1, 5.0, 3.0))
        old_sc = get_score(tr)
        chm = choicemap((:x => :z, 5.0))
        new_tr, w, rd, discard = update(tr, (2, 5.0, 3.0), 
                                        (UnknownChange(), NoChange(), NoChange()), 
                                        chm)
        @test old_sc == get_score(new_tr) - w
        chm = choicemap((:x => :z, 10.0))
        new_tr, w, rd, discard = update(tr, (1, 5.0, 3.0), 
                                        (UnknownChange(), NoChange(), NoChange()), 
                                        chm)
        @test old_sc == get_score(new_tr) - w
    end

    @testset "regenerate" begin
        tr = simulate(sc, (2, 5.0, 3.0))
        old_sc = get_score(tr)
        sel = select(:z)
        new_tr, w, rd = regenerate(tr, (2, 5.0, 3.0), 
                               (UnknownChange(), NoChange(), NoChange()), 
                               sel)
        @test old_sc == get_score(new_tr) - w
        new_tr, w, rd = regenerate(tr, (1, 5.0, 3.0), 
                               (UnknownChange(), NoChange(), NoChange()), 
                               sel)
        @test old_sc == get_score(new_tr) - w
    end

    @testset "choice gradients" begin
        tr = simulate(sc, (2, 5.0, 3.0))
        sel = select(:z)
        arg_grads, cvs, cgs = choice_gradients(tr, sel, 1.0)
    end

    @testset "accumulate parameter gradients" begin
    end

    # ------------ Hierarchy ------------ #

    # Model chunk.
    @gen (grad) function bang((grad)(x::Float64), (grad)(y::Float64))
        std::Float64 = 3.0
        z = @trace(normal(x + y, std), :z)
        return z
    end
    @gen (grad) function fuzz((grad)(x::Float64), (grad)(y::Float64))
        std::Float64 = 3.0
        z = @trace(normal(x + 2 * y, std), :z)
        return z
    end
    sc = Switch(bang, fuzz)
    @gen (grad) function bam(s::Int)
        x ~ sc(s, 5.0, 3.0)
        return x
    end
    # ----.

    @testset "simulate" begin
        tr = simulate(bam, (2, ))
    end

    @testset "generate" begin
        chm = choicemap()
        chm[:x => :z] = 5.0
        tr, _ = generate(sc, (2, 5.0, 3.0), chm)
    end

    @testset "assess" begin
    end

    @testset "propose" begin
    end

    @testset "update" begin
        tr = simulate(bam, (2, ))
        old_sc = get_score(tr)
        chm = choicemap((:x => :z, 5.0))
        new_tr, w, rd, discard = update(tr, (2, ), (UnknownChange(), ), chm)
        @test old_sc == get_score(new_tr) - w
        chm = choicemap((:x => :z, 10.0))
        new_tr, w, rd, discard = update(tr, (1, ), (UnknownChange(), ), chm)
        @test old_sc == get_score(new_tr) - w
    end

    @testset "regenerate" begin
        tr = simulate(bam, (2, ))
        old_sc = get_score(tr)
        new_tr, w = regenerate(tr, (1, ), (UnknownChange(), ), select())
        @test old_sc == get_score(new_tr) - w
        new_tr, w = regenerate(tr, (2, ), (UnknownChange(), ), select())
        @test old_sc == get_score(new_tr) - w
        new_tr, w = regenerate(tr, (1, ), (UnknownChange(), ), select())
        @test old_sc == get_score(new_tr) - w
    end

    @testset "choice gradients" begin
        tr = simulate(bam, (2, ))
        sel = select(:x => :z)
        arg_grads, cvs, cgs = choice_gradients(tr, sel, 1.0)
    end

    @testset "accumulate parameter gradients" begin
    end
end
