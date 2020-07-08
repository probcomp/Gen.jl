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

cm = choicemap((:a, 1), (:b => :c, 2))

println("dynamic choicemap nonnested lookup:")
for _=1:4
    @time many_shallow(cm)
end

println("dynamic choicemap nested lookup:")
for _=1:4
    @time many_nested(cm)
end