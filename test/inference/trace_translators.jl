@testset "trace translators" begin

@testset "DeterministicTraceTranslator" begin

    @gen function p1()
        r ~ inv_gamma(1, 1)
        theta ~ uniform(-pi/2, pi/2)
    end

    @gen function p2()
        x ~ normal(0, 1)
        y ~ normal(0, 1)
    end

    @transform f (t1) to (t2) begin
        r = @read(t1[:r], :continuous)
        theta = @read(t1[:theta], :continuous)
        @write(t2[:x], r * cos(theta), :continuous)
        @write(t2[:y], r * sin(theta), :continuous)
    end

    @transform finv (t2) to (t1) begin
        x = @read(t2[:x], :continuous)
        y = @read(t2[:y], :continuous)
        r = sqrt(x^2 + y^2)
        @write(t1[:r], sqrt(x^2 + y^2), :continuous)
        @write(t1[:theta], atan(y, x), :continuous)
    end

    pair_bijections!(f, finv)

    translator = DeterministicTraceTranslator(p2, (), choicemap(), f)
    t1, _ = generate(p1, (), choicemap(:theta => 0))
    t2, log_weight = translator(t1; check=true)
    @test t2[:y] == 0

    translator = DeterministicTraceTranslator(p1, (), choicemap(), finv)
    t2, _ = generate(p2, (), choicemap(:y => 0, :x => 1))
    t1, log_weight = translator(t2; check=true)
    @test t1[:theta] == 0

end

@testset "SimpleExtendingTraceTranslator" begin

    @gen function model(T::Int)
        for t in 1:T
            z = {(:z, t)} ~ normal(0, 1)
            x = {(:x, t)} ~ normal(z, 1)
        end
    end

    @gen function proposal(trace::Trace, x)
        t = get_args(trace)[1] + 1
        {(:z, t)} ~ normal(x, 1)
    end

    translator = SimpleExtendingTraceTranslator(
        p_new_args=(2,), p_argdiffs=(UnknownChange(),),
        new_observations=choicemap((:x, 2) => 5.0),
        q_forward=proposal, q_forward_args=(5.0,))
    t1  = simulate(model, (1,))
    t2, log_weight = translator(t1)

    prop_choices = choicemap((:z, 2) => t2[(:z, 2)])
    prop_weight, _ = assess(proposal, (t1, 5.0), prop_choices)
    constraints = merge(prop_choices, choicemap((:x, 2) => 5.0))
    t3, up_weight, _, _ = update(t1, (2,), (UnknownChange(),), constraints)
    @test log_weight == up_weight - prop_weight

end

@testset "SymmetricTraceTranslator" begin

    @gen function model()
        z ~ bernoulli(0.5)
        if z
            i ~ uniform_discrete(1, 10)
        else
            x ~ uniform(0, 1)
        end
    end

    @gen function proposal(trace)
        if trace[:z]
            dx ~ uniform(0.0, 0.1)
        end
    end

    @transform involution (p1_trace, q1_trace) to (p2_trace, q2_trace) begin
        if @read(p1_trace[:z], :discrete)
            @write(p2_trace[:z], false, :discrete)
            i = @read(p1_trace[:i], :discrete)
            dx = @read(q1_trace[:dx], :continuous)
            x = (i-1)/10 + dx
            @write(p2_trace[:x], x, :continuous)
        else
            @write(p2_trace[:z], true, :discrete)
            x = @read(p1_trace[:x], :continuous)
            i = Int(ceil(x * 10))
            @write(p2_trace[:i], i, :discrete)
            @write(q2_trace[:dx], x - (i-1)/10, :continuous)
        end
    end

    is_involution!(involution)

    translator = SymmetricTraceTranslator(proposal, (), involution)

    t1, _ = generate(model, (), choicemap(:z => false, :x => 0.95))
    t2, log_weight = translator(t1; check=true)
    @test t2[:z] == true && t2[:i] == 10

end

@testset "GeneralTraceTranslator" begin

    @gen function p1()
        x ~ uniform(0, 1)
        y ~ uniform(0, 1)
    end

    @gen function p2()
        i ~ uniform_discrete(1, 10) # interval [(i-1)/10, i/10]
        j ~ uniform_discrete(1, 10) # interval [(j-1)/10, j/10]
    end

    @gen function q1(p1_trace) end

    @gen function q2(p2_trace)
        dx ~ uniform(0.0, 0.1)
        dy ~ uniform(0.0, 0.1)
    end

    @transform f (p1_trace, q1_trace) to (p2_trace, q2_trace) begin
        x = @read(p1_trace[:x], :continuous)
        y = @read(p1_trace[:y], :continuous)
        i = Int(ceil(x * 10))
        j = Int(ceil(y * 10))
        @write(p2_trace[:i], i, :discrete)
        @write(p2_trace[:j], j, :discrete)
        @write(q2_trace[:dx], x - (i-1)/10, :continuous)
        @write(q2_trace[:dy], y - (j-1)/10, :continuous)
    end

    @transform f_inv (p2_trace, q2_trace) to (p1_trace, q1_trace) begin
        i = @read(p2_trace[:i], :discrete)
        j = @read(p2_trace[:j], :discrete)
        dx = @read(q2_trace[:dx], :continuous)
        dy = @read(q2_trace[:dy], :continuous)
        x = (i-1)/10 + dx
        y = (j-1)/10 + dy
        @write(p1_trace[:x], x, :continuous)
        @write(p1_trace[:y], y, :continuous)
    end

    pair_bijections!(f, f_inv)

    translator = GeneralTraceTranslator(
        p_new=p2, p_new_args=(),
        new_observations=choicemap(),
        q_forward=q1, q_forward_args=(),
        q_backward=q2, q_backward_args=(), f=f)

    t1, _ = generate(p1, (), choicemap(:x => 0.05, :y => 0.95))
    t2, log_weight = translator(t1; check=true)
    @test t2[:i] == 1 && t2[:j] == 10

end

end
