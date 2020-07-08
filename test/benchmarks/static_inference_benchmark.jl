using Gen

@gen (static, diffs) function foo()
    a ~ normal(0, 1)
    b ~ normal(a, 1)
    c ~ normal(b, 1)
end

@load_generated_functions

observations = StaticChoiceMap(choicemap((:b,2), (:c,1.5)))
tr, _ = generate(foo, (), observations)

function run_inference(trace)
    tr = trace
    for _=1:10^3
        tr, acc = mh(tr, select(:a))
    end
end

for _=1:4
    @time run_inference(tr)
end