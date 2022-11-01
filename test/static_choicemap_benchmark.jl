using Gen

function many_shallow(cm::ChoiceMap)
    for _=1:10^5
        cm[:a]
    end
end
function many_nested(cm::ChoiceMap)
    for _=1:10^5
        cm[:b => :c]
    end
end

# many_shallow(cm) = perform_many_lookups(cm, :a)
# many_nested(cm) = perform_many_lookups(cm, :b => :c)

scm = StaticChoiceMap(a=1, b=StaticChoiceMap(c=2))

println("static choicemap nonnested lookup:")
for _=1:4
    @time many_shallow(scm)
end

println("static choicemap nested lookup:")
for _=1:4
    @time many_nested(scm)
end

@gen (static) function inner()
    c ~ normal(0, 1)
end
@gen (static) function outer()
    a ~ normal(0, 1)
    b ~ inner()
end

tr, _ = generate(outer, ())
choices = get_choices(tr)

println("static gen function choicemap nonnested lookup:")
for _=1:4
    @time many_shallow(choices)
end

println("static gen function choicemap nested lookup:")
for _=1:4
    @time many_nested(choices)
end
