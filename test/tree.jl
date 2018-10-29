using Gen: get_child, get_child_num, get_parent

@testset "tree node numbering" begin

    # max_branch = 1
    @test get_child(1, 1, 1) == 2
    @test get_child(2, 1, 1) == 3
    @test get_child(3, 1, 1) == 4
    @test get_child_num(2, 1) == 1
    @test get_child_num(3, 1) == 1
    @test get_child_num(4, 1) == 1
    @test get_parent(2, 1) == 1
    @test get_parent(3, 1) == 2
    @test get_parent(4, 1) == 3
    
    # max_branch = 2
    @test get_child(1, 1, 2) == 2
    @test get_child(1, 2, 2) == 3
    @test get_child(2, 1, 2) == 4
    @test get_child(2, 2, 2) == 5
    @test get_child(3, 1, 2) == 6
    @test get_child_num(2, 2) == 1
    @test get_child_num(3, 2) == 2
    @test get_child_num(4, 2) == 1
    @test get_child_num(5, 2) == 2
    @test get_child_num(6, 2) == 1
    @test get_parent(2, 2) == 1
    @test get_parent(3, 2) == 1
    @test get_parent(4, 2) == 2
    @test get_parent(5, 2) == 2
    @test get_parent(6, 2) == 3

    # max_branch = 3
    @test get_child(1, 1, 3) == 2
    @test get_child(1, 2, 3) == 3
    @test get_child(1, 3, 3) == 4
    @test get_child(2, 1, 3) == 5
    @test get_child(2, 2, 3) == 6
    @test get_child(2, 3, 3) == 7
    @test get_child(3, 1, 3) == 8
    @test get_child(3, 2, 3) == 9
    @test get_child(3, 3, 3) == 10
    @test get_child(4, 1, 3) == 11
    @test get_child(4, 2, 3) == 12
    @test get_child(4, 3, 3) == 13
    @test get_child_num(2, 3) == 1
    @test get_child_num(3, 3) == 2
    @test get_child_num(4, 3) == 3
    @test get_child_num(5, 3) == 1
    @test get_child_num(6, 3) == 2
    @test get_child_num(7, 3) == 3
    @test get_child_num(8, 3) == 1
    @test get_child_num(9, 3) == 2
    @test get_child_num(10, 3) == 3
    @test get_child_num(11, 3) == 1
    @test get_child_num(12, 3) == 2
    @test get_child_num(13, 3) == 3
    @test get_parent(2, 3) == 1
    @test get_parent(3, 3) == 1
    @test get_parent(4, 3) == 1
    @test get_parent(5, 3) == 2
    @test get_parent(6, 3) == 2
    @test get_parent(7, 3) == 2
    @test get_parent(8, 3) == 3
    @test get_parent(9, 3) == 3
    @test get_parent(10, 3) == 3
    @test get_parent(11, 3) == 4
    @test get_parent(12, 3) == 4
    @test get_parent(13, 3) == 4
end

@testset "simple pcfg" begin

    @gen function pcfg_production(_::Nothing)
        production_rule = @addr(categorical([0.24, 0.26, 0.23, 0.27]), :rule)
        if production_rule == 1
            # aSa; one child
            return (production_rule, Nothing[nothing])
        elseif production_rule == 2
            # bSb; one child
            return (production_rule, Nothing[nothing])
        elseif production_rule == 3
            # aa; no child
            return (production_rule, Nothing[])
        else
            # bb; no child
            return (production_rule, Nothing[])
        end
    end
    
    @gen function pcfg_aggregation(production_rule::Int, child_outputs::Vector{String})
        prefix = @addr(bernoulli(0.4), :prefix) ? "." : "-"
        if production_rule == 1
            @assert length(child_outputs) == 1
            return "($(prefix)a$(child_outputs[1])a)"
        elseif production_rule == 2
            @assert length(child_outputs) == 1
            return "($(prefix)b$(child_outputs[1])b)"
        elseif production_rule == 3
            @assert length(child_outputs) == 0
            return "($(prefix)aa)"
        else
            @assert length(child_outputs) == 0
            return "($(prefix)bb)"
        end
    end

    pcfg = Tree(pcfg_production, pcfg_aggregation, 1, Nothing, Int, String, Nothing, Nothing, Nothing)

    # test that each of the 6 most probably strings is produced
    Random.seed!(1)
    strings = Set{String}()
    for i=1:1000
        trace = simulate(pcfg, (nothing, 1))
        push!(strings, get_call_record(trace).retval)
    end
    @test "(.aa)" in strings
    @test "(-aa)" in strings
    @test "(.bb)" in strings
    @test "(-bb)" in strings
    @test "(.a(.aa)a)" in strings
    @test "(-a(.aa)a)" in strings
    @test "(.a(-aa)a)" in strings
    @test "(-a(-aa)a)" in strings
    @test "(.b(.bb)b)" in strings
    @test "(-b(.bb)b)" in strings
    @test "(.b(-bb)b)" in strings
    @test "(-b(-bb)b)" in strings
    @test "(.a(.bb)a)" in strings
    @test "(-a(.bb)a)" in strings
    @test "(.a(-bb)a)" in strings
    @test "(-a(-bb)a)" in strings
    @test "(.b(.aa)b)" in strings
    @test "(-b(.aa)b)" in strings
    @test "(.b(-aa)b)" in strings
    @test "(-b(-aa)b)" in strings
    
    # test generate on a complete assignment that produces "(.b(.a(.b(-bb)b)a)b)"
    # sequence of production rules: 2 -> 1 -> 2 -> 4
    expected_weight = log(0.26) + log(0.24) + log(0.26) + log(0.27) + log(0.4) + log(0.4) + log(0.4) + log(0.6)
    constraints = DynamicAssignment()
    constraints[(1, Val(:production)) => :rule] = 2
    constraints[(1, Val(:aggregation)) => :prefix] = true
    constraints[(2, Val(:production)) => :rule] = 1
    constraints[(2, Val(:aggregation)) => :prefix] = true
    constraints[(3, Val(:production)) => :rule] = 2
    constraints[(3, Val(:aggregation)) => :prefix] = true
    constraints[(4, Val(:production)) => :rule] = 4
    constraints[(4, Val(:aggregation)) => :prefix] = false
    (trace, actual_weight) = generate(pcfg, (nothing, 1), constraints)
    @test isapprox(actual_weight, expected_weight)
    @test get_call_record(trace).score == actual_weight
    @test get_call_record(trace).args == (nothing, 1)
    @test get_call_record(trace).retval == "(.b(.a(.b(-bb)b)a)b)"
    assignment = get_assignment(trace)
    @test assignment[(1, Val(:production)) => :rule] == 2
    @test assignment[(1, Val(:aggregation)) => :prefix] == true
    @test assignment[(2, Val(:production)) => :rule] == 1
    @test assignment[(2, Val(:aggregation)) => :prefix] == true
    @test assignment[(3, Val(:production)) => :rule] == 2
    @test assignment[(3, Val(:aggregation)) => :prefix] == true
    @test assignment[(4, Val(:production)) => :rule] == 4
    @test assignment[(4, Val(:aggregation)) => :prefix] == false
end
