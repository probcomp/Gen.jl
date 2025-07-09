var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Gen Introduction",
    "title": "Gen Introduction",
    "category": "page",
    "text": ""
},

{
    "location": "#Gen-Introduction-1",
    "page": "Gen Introduction",
    "title": "Gen Introduction",
    "category": "section",
    "text": "Gen is an extensible and reasonably performant probabilistic computing platform that makes it easier to develop probabilistic inference and learning applications."
},

{
    "location": "#Generative-Functions,-Traces,-and-Assignments-1",
    "page": "Gen Introduction",
    "title": "Generative Functions, Traces, and Assignments",
    "category": "section",
    "text": "Stochastic generative processes are represented in Gen as generative functions. Generative functions are Julia functions that have been annotated using the @gen macro. The generative function below takes a vector of x-coordinates, randomly generates the slope and intercept parameters of a line, and returns a random vector of y-coordinates sampled near that line at the given x-coordinates:@gen function regression_model(xs::Vector{Float64})\n    slope = normal(0, 2)\n    intercept = normal(0, 2)\n    ys = Vector{Float64}(undef, length(xs))\n    for (i, x) in enumerate(xs)\n        ys[i] = normal(slope * xs + intercept, 1.)\n    end\n    return ys\nendWe can evaluate the generative function:ys = regression_model([-5.0, -3.0, 0.0, 3.0, 5.0])Above we have used a generative function to implement a simulation. However, what distinguishes generative functions from plain-old simulators is their ability to be traced. When we trace a generative function, we record the random choices that it makes, as well as additional data about the evaluation of the function. This capability makes it possible to implement algorithms for probabilistic inference. To trace a random choice, we need to give it a unique address, using the @addr keyword. Here we give addresses to each of the random choices:@gen function regression_model(xs::Vector{Float64})\n    slope = @addr(normal(0, 2), \"slope\")\n    intercept = @addr(normal(0, 2), \"intercept\")\n    ys = Vector{Float64}(undef, length(xs))\n    for (i, x) in enumerate(xs)\n        ys[i] = @addr(normal(slope * xs[i] + intercept, 1.), \"y-$i\")\n    end\n    return ys\nendAddresses can be arbitrary Julia values except for Pair. Julia symbols, strings, and integers are common types to use for addresses.We trace a generative function using the simulate method. We provide the arguments to the function in a tuple:xs = [-5.0, -3.0, 0.0, 3.0]\ntrace = simulate(regression_model, (xs,))The trace that is returned is a form of stack trace of the generative function that contains, among other things, the values for the random choices that were annotated with @addr. To extract the values of the random choices from a trace, we use the method get_assmt:assignment = get_assmt(trace)The get_assmt method returns an assignment, which is a trie (prefix tree) that contains the values of random choices. Printing the assignment gives a pretty-printed representation:print(assignment)│\n├── \"y-2\" : -1.7800101128038626\n│\n├── \"y-3\" : 0.1832573462320619\n│\n├── \"intercept\" : 1.7434641799887896\n│\n├── \"y-4\" : 5.074512278024528\n│\n├── \"slope\" : 1.5232349541190595\n│\n└── \"y-1\" : -4.978881121779669Generative functions can also call other generative functions; these calls can also be traced using @addr:@gen function generate_params()\n    slope = @addr(normal(0, 2), \"slope\")\n    intercept = @addr(normal(0, 2), \"intercept\")\n    return (slope, intercept)\nend\n\n@gen function generate_datum(x, slope, intercept)\n    return @addr(normal(slope * x + intercept, 1.), \"y\")\nend\n\n@gen function regression_model(xs::Vector{Float64})\n    (slope, intercept) = @addr(generate_params(), \"parameters\")\n    ys = Vector{Float64}(undef, length(xs))\n    for (i, x) in enumerate(xs)\n        ys[i] = @addr(generate_datum(xs[i], slope, intercept), i)\n    end\n    return ys\nendThis results in a hierarchical assignment:trace = simulate(regression_model, (xs,))\nassignment = get_assmt(trace)\nprint(assignment)│\n├── 2\n│   │\n│   └── \"y\" : -3.264252749715529\n│\n├── 3\n│   │\n│   └── \"y\" : -2.3036480286819865\n│\n├── \"parameters\"\n│   │\n│   ├── \"intercept\" : -0.8767368668034233\n│   │\n│   └── \"slope\" : 0.9082675922758383\n│\n├── 4\n│   │\n│   └── \"y\" : 2.4971551239517695\n│\n└── 1\n    │\n    └── \"y\" : -7.561723378403817We can read values from a assignment using the following syntax:assignment[\"intercept\"]To read the value at a hierarchical address, we provide a Pair where the first element of the pair is the first part ofthe hierarchical address, and the second element is the rest of the address. For example:assignment[1 => \"y\"]Julia provides the operator => for constructing Pair values. Long hierarchical addresses can be constructed by chaining this operator, which associates right:assignment[1 => \"y\" => :foo => :bar]Generative functions can also write to hierarchical addresses directly:@gen function regression_model(xs::Vector{Float64})\n    slope = @addr(normal(0, 2), \"slope\")\n    intercept = @addr(normal(0, 2), \"intercept\")\n    ys = Vector{Float64}(undef, length(xs))\n    for (i, x) in enumerate(xs)\n        ys[i] = @addr(normal(slope * xs[i] + intercept, 1.), i => \"y\")\n    end\n    return ys\nendtrace = simulate(regression_model, (xs,))\nassignment = get_assmt(trace)\nprint(assignment)│\n├── \"intercept\" : -1.340778590777462\n│\n├── \"slope\" : -2.0846094796654686\n│\n├── 2\n│   │\n│   └── \"y\" : 3.64234023192473\n│\n├── 3\n│   │\n│   └── \"y\" : -1.5439406188116667\n│\n├── 4\n│   │\n│   └── \"y\" : -8.655741483764384\n│\n└── 1\n    │\n    └── \"y\" : 9.451320138931484"
},

{
    "location": "#Implementing-Inference-Algorithms-1",
    "page": "Gen Introduction",
    "title": "Implementing Inference Algorithms",
    "category": "section",
    "text": ""
},

{
    "location": "#Implementing-Gradient-Based-Learning-1",
    "page": "Gen Introduction",
    "title": "Implementing Gradient-Based Learning",
    "category": "section",
    "text": ""
},

{
    "location": "#Probabilistic-Modules-1",
    "page": "Gen Introduction",
    "title": "Probabilistic Modules",
    "category": "section",
    "text": ""
},

{
    "location": "#Compiled-Generative-Functions-1",
    "page": "Gen Introduction",
    "title": "Compiled Generative Functions",
    "category": "section",
    "text": ""
},

{
    "location": "#Incremental-Computation-1",
    "page": "Gen Introduction",
    "title": "Incremental Computation",
    "category": "section",
    "text": "Getting good asymptotic scaling for iterative local search algorithms like MCMC or MAP optimization relies on the ability to update a trace efficiently in two common scenarios: (i) when a small number of random choice(s) are changed; or (ii) when there is a small change to the arguments of the function.To enable efficient trace updates, generative functions use argdiffs and retdiffs:An argdiff describes the change made to the arguments of the generative function, relative to the arguments in the previous trace.\nA retdiff describes the change to the return value of a generative function, relative to the return value in the previous trace.The update generative function API methods update, fix_update, and extend accept the argdiff value, alongside the new arguments to the function, the previous trace, and other parameters; and return the new trace and the retdiff value."
},

{
    "location": "#Argdiffs-1",
    "page": "Gen Introduction",
    "title": "Argdiffs",
    "category": "section",
    "text": "An argument difference value, or argdiff, is associated with a pair of argument tuples args::Tuple and new_args::Tuple. The update methods for a generative function accept different types of argdiff values, that depend on the generative function. Two singleton data types are provided:noargdiff::NoArgDiff, for expressing that there is no difference to the argument.\nunknownargdiff::UnknownArgDiff, for expressing that there is an unknown difference in the arguments.Generative functions may or may not accept these types as argdiffs, depending on the generative function."
},

{
    "location": "#Retdiffs-1",
    "page": "Gen Introduction",
    "title": "Retdiffs",
    "category": "section",
    "text": "A return value difference value, or retdiff, is associated with a pair of traces. The update methods for a generative function return different types of retdiff values, depending on the generative function. The only requirement placed on retdiff values is that they implement the isnodiff method, which takes a retdiff value and returns true if there was no change in the return value, and otherwise returns false."
},

{
    "location": "#Gen.NewChoiceDiff",
    "page": "Gen Introduction",
    "title": "Gen.NewChoiceDiff",
    "category": "type",
    "text": "NewChoiceDiff()\n\nSingleton indicating that there was previously no random choice at this address.\n\n\n\n\n\n"
},

{
    "location": "#Gen.NoChoiceDiff",
    "page": "Gen Introduction",
    "title": "Gen.NoChoiceDiff",
    "category": "type",
    "text": "NoChoiceDiff()\n\nSingleton indicating that the value of the random choice did not change.\n\n\n\n\n\n"
},

{
    "location": "#Gen.PrevChoiceDiff",
    "page": "Gen Introduction",
    "title": "Gen.PrevChoiceDiff",
    "category": "type",
    "text": "PrevChoiceDiff(prev)\n\nWrapper around the previous value of the random choice indicating that it may have changed.\n\n\n\n\n\n"
},

{
    "location": "#Gen.NewCallDiff",
    "page": "Gen Introduction",
    "title": "Gen.NewCallDiff",
    "category": "type",
    "text": "NewCallDiff()\n\nSingleton indicating that there was previously no call at this address.\n\n\n\n\n\n"
},

{
    "location": "#Gen.NoCallDiff",
    "page": "Gen Introduction",
    "title": "Gen.NoCallDiff",
    "category": "type",
    "text": "NoCallDiff()\n\nSingleton indicating that the return value of the call has not changed.\n\n\n\n\n\n"
},

{
    "location": "#Gen.UnknownCallDiff",
    "page": "Gen Introduction",
    "title": "Gen.UnknownCallDiff",
    "category": "type",
    "text": "UnknownCallDiff()\n\nSingleton indicating that there was a previous call at this address, but that no information is known about the change to the return value.\n\n\n\n\n\n"
},

{
    "location": "#Gen.CustomCallDiff",
    "page": "Gen Introduction",
    "title": "Gen.CustomCallDiff",
    "category": "type",
    "text": "CustomCallDiff(retdiff)\n\nWrapper around a retdiff value, indicating that there was a previous call at this address, and that isnodiff(retdiff) = false (otherwise NoCallDiff() would have been returned).\n\n\n\n\n\n"
},

{
    "location": "#Custom-incremental-computation-in-embedded-modeling-DSL-1",
    "page": "Gen Introduction",
    "title": "Custom incremental computation in embedded modeling DSL",
    "category": "section",
    "text": "<!– TODO: This section is a bit confusing, in particular the macros @choicediff and @calldiff appear suddenly and are not clear or well-motivated. Should introduce them earlier and explain more.–>We now show how argdiffs and retdiffs and can be used for incremental computation in the embedded modeling DSL. For generative functions expressed in the embedded modeling DSL, retdiff values are computed by user diff code that is placed inline in the body of the generative function definition. Diff code consists of Julia statements that can depend on non-diff code, but non-diff code cannot depend on the diff code. To distinguish diff code  from regular code in the generative function, the @diff macro is placed in front of the statement, e.g.:x = y + 1\n@diff foo = 2\ny = xDiff code is only executed during update methods such as update, fix_update, or extend methods. In other methods that are not associated with an update to a trace (e.g. generate, simulate, assess), the diff code is removed from the body of the generative function. Therefore, the body of the generative function with the diff code removed must still be a valid generative function.Unlike non-diff code, diff code has access to the argdiff value (using @argdiff()), and may invoke @retdiff(<value>), which sets the retdiff value. Diff code also has access to information about the change to the values of random choices and the change to the return values of calls to other generative functions. Changes to the return values of random choices are queried using @choicediff(<addr>), which must be invoked after the @addr expression for that address, and returns one of the following values:NewChoiceDiff\nNoChoiceDiff\nPrevChoiceDiffDiff code also has access to the retdiff values associated with calls it makes to generative functions, using @calldiff(<addr>), which returns a value of one of the following types:NewCallDiff\nNoCallDiff\nUnknownCallDiff\nCustomCallDiffDiff code can also pass argdiff values to generative functions that it calls, using the third argument in an @addr expression, which is always interpreted as diff code (depsite the absence of a @diff keyword).@diff my_argdiff = @argdiff()\n@diff argdiff_for_foo = ..\n@addr(foo(arg1, arg2), addr, argdiff_for_foo)\n@diff retdiff_from_foo = @calldiff(addr)\n@diff @retdiff(..)"
},

{
    "location": "#Higher-Order-Probabilistic-Modules-1",
    "page": "Gen Introduction",
    "title": "Higher-Order Probabilistic Modules",
    "category": "section",
    "text": ""
},

{
    "location": "#Using-metaprogramming-to-implement-new-inference-algorithms-1",
    "page": "Gen Introduction",
    "title": "Using metaprogramming to implement new inference algorithms",
    "category": "section",
    "text": "Many Monte Carlo inference algorithms, like Hamiltonian Monte Carlo (HMC) and Metropolis-Adjusted Langevin Algorithm (MALA) are instances of general inference algorithm templates like Metropolis-Hastings, with specialized proposal distributions. These algorithms can therefore be implemented with high-performance for a model if a compiled generative function defining the proposal is constructed manually. However, it is also possible to write a generic implementation that automatically generates the generative function for the proposal using Julia\'s metaprogramming capabilities. This section shows a simple example of writing a procedure that generates the code needed for a MALA update applied to an arbitrary set of top-level static addresses.First, we write out a non-generic implementation of MALA. MALA uses a proposal that tends to propose values in the direction of the gradient. The procedure will be hardcoded to act on a specific set of addresses for a model called model.set = DynamicAddressSet()\nfor addr in [:slope, :intercept, :inlier_std, :outlier_std]\n    Gen.push_leaf_node!(set, addr)\nend\nmala_selection = StaticAddressSet(set)\n\n@compiled @gen function mala_proposal(prev, tau)\n    std::Float64 = sqrt(2*tau)\n    gradients::StaticChoiceTrie = backprop_trace(model, prev, mala_selection, nothing)[3]\n    @addr(normal(get_assmt(prev)[:slope] + tau * gradients[:slope], std), :slope)\n    @addr(normal(get_assmt(prev)[:intercept] + tau * gradients[:intercept], std), :intercept)\n    @addr(normal(get_assmt(prev)[:inlier_std] + tau * gradients[:inlier_std], std), :inlier_std)\n    @addr(normal(get_assmt(prev)[:outlier_std] + tau * gradients[:outlier_std], std), :outlier_std)\nend\n\nmala_move(trace, tau::Float64) = mh(model, mala_proposal, (tau,), trace)Next, we write a generic version that takes a set of addresses and generates the implementation for that set. This version only works on a set of static top-level addresses.function generate_mala_move(model, addrs)\n\n    # create selection\n    set = DynamicAddressSet()\n    for addr in addrs\n        Gen.push_leaf_node!(set, addr)\n    end\n    selection = StaticAddressSet(set)\n\n    # generate proposal function\n    stmts = Expr[]\n    for addr in addrs\n        quote_addr = QuoteNode(addr)\n        push!(stmts, :(\n            @addr(normal(get_assmt(prev)[$quote_addr] + tau * gradients[$quote_addr], std),\n                  $quote_addr)\n        ))\n    end\n    mala_proposal_name = gensym(\"mala_proposal\")\n    mala_proposal = eval(quote\n        @compiled @gen function $mala_proposal_name(prev, tau)\n            gradients::StaticChoiceTrie = backprop_trace(\n                model, prev, $(QuoteNode(selection)), nothing)[3]\n            std::Float64 = sqrt(2*tau)\n            $(stmts...)\n        end\n    end)\n\n    return (trace, tau::Float64) -> mh(model, mala_proposal, (tau,), trace)\nend\nWe can then use the generate the MALA move for a speciic model and specific addresses using:mala_move = generate_mala_move(model, [:slope, :intercept, :inlier_std, :outlier_std])"
},

{
    "location": "documentation/#",
    "page": "Gen Documentation",
    "title": "Gen Documentation",
    "category": "page",
    "text": ""
},

{
    "location": "documentation/#Gen-Documentation-1",
    "page": "Gen Documentation",
    "title": "Gen Documentation",
    "category": "section",
    "text": ""
},

{
    "location": "documentation/#Gen.has_value",
    "page": "Gen Documentation",
    "title": "Gen.has_value",
    "category": "function",
    "text": "has_value(assmt::Assignment, addr)\n\nReturn true if there is a value at the given address.\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.get_value",
    "page": "Gen Documentation",
    "title": "Gen.get_value",
    "category": "function",
    "text": "value = get_value(assmt::Assignment, addr)\n\nReturn the value at the given address in the assignment, or throw a KeyError if no value exists. A syntactic sugar is Base.getindex:\n\nvalue = assmt[addr]\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.get_subassmt",
    "page": "Gen Documentation",
    "title": "Gen.get_subassmt",
    "category": "function",
    "text": "subassmt = get_subassmt(assmt::Assignment, addr)\n\nReturn the sub-assignment containing all choices whose address is prefixed by addr.\n\nIt is an error if the assignment contains a value at the given address. If there are no choices whose address is prefixed by addr then return an EmptyAssignment.\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.get_values_shallow",
    "page": "Gen Documentation",
    "title": "Gen.get_values_shallow",
    "category": "function",
    "text": "key_subassmt_iterable = get_values_shallow(assmt::Assignment)\n\nReturn an iterable collection of tuples (key, subassmt::Assignment) for each top-level key associated with a value.\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.get_subassmts_shallow",
    "page": "Gen Documentation",
    "title": "Gen.get_subassmts_shallow",
    "category": "function",
    "text": "key_subassmt_iterable = get_subassmts_shallow(assmt::Assignment)\n\nReturn an iterable collection of tuples (key, subassmt::Assignment) for each top-level key that has a non-empty sub-assignment.\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.to_array",
    "page": "Gen Documentation",
    "title": "Gen.to_array",
    "category": "function",
    "text": "arr::Vector{T} = to_array(assmt::Assignment, ::Type{T}) where {T}\n\nPopulate an array with values of choices in the given assignment.\n\nIt is an error if each of the values cannot be coerced into a value of the given type.\n\nImplementation\n\nTo support to_array, a concrete subtype T <: Assignment should implement the following method:\n\nn::Int = _fill_array!(assmt::T, arr::Vector{V}, start_idx::Int) where {V}\n\nPopulate arr with values from the given assignment, starting at start_idx, and return the number of elements in arr that were populated.\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.from_array",
    "page": "Gen Documentation",
    "title": "Gen.from_array",
    "category": "function",
    "text": "assmt::Assignment = from_array(proto_assmt::Assignment, arr::Vector)\n\nReturn an assignment with the same address structure as a prototype assignment, but with values read off from the given array.\n\nThe order in which addresses are populated is determined by the prototype assignment. It is an error if the number of choices in the prototype assignment is not equal to the length the array.\n\nImplementation\n\nTo support from_array, a concrete subtype T <: Assignment should implement the following method:\n\n(n::Int, assmt::T) = _from_array(proto_assmt::T, arr::Vector{V}, start_idx::Int) where {V}\n\nReturn an assignment with the same address structure as a prototype assignment, but with values read off from arr, starting at position start_idx, and the number of elements read from arr.\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.pair",
    "page": "Gen Documentation",
    "title": "Gen.pair",
    "category": "function",
    "text": "assmt = pair(assmt1::Assignment, assmt2::Assignment, key1::Symbol, key2::Symbol)\n\nReturn an assignment that contains assmt1 as a sub-assignment under key1 and assmt2 as a sub-assignment under key2.\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.unpair",
    "page": "Gen Documentation",
    "title": "Gen.unpair",
    "category": "function",
    "text": "(assmt1, assmt2) = unpair(assmt::Assignment, key1::Symbol, key2::Symbol)\n\nReturn the two sub-assignments at key1 and key2, one or both of which may be empty.\n\nIt is an error if there are any top-level values, or any non-empty top-level sub-assignments at keys other than key1 and key2.\n\n\n\n\n\n"
},

{
    "location": "documentation/#Base.merge",
    "page": "Gen Documentation",
    "title": "Base.merge",
    "category": "function",
    "text": "assmt = Base.merge(assmt1::Assignment, assmt2::Assignment)\n\nMerge two assignments.\n\nIt is an error if the assignments both have values at the same address, or if one assignment has a value at an address that is the prefix of the address of a value in the other assignment.\n\n\n\n\n\n"
},

{
    "location": "documentation/#Assignments-1",
    "page": "Gen Documentation",
    "title": "Assignments",
    "category": "section",
    "text": "An assignment is a map from addresses of random choices to their values. Assignments are represented using the abstract type Assignment. Assignments have the following methods:has_value\nget_value\nget_subassmt\nget_values_shallow\nget_subassmts_shallow\nto_array\nfrom_array\npair\nunpair\nBase.mergeTODO: change get_assmt to assmt TODO: simplify other method names"
},

{
    "location": "documentation/#Gen.get_address_schema",
    "page": "Gen Documentation",
    "title": "Gen.get_address_schema",
    "category": "function",
    "text": "schema = get_address_schema(::Type{T}) where {T <: Assignment}\n\nReturn the (top-level) address schema for the given assignment.\n\n\n\n\n\n"
},

{
    "location": "documentation/#Address-Schemata-1",
    "page": "Gen Documentation",
    "title": "Address Schemata",
    "category": "section",
    "text": "An address schema provides information about the set of addresses of random choices in an assignment.Address schemata are associated with the type of an assignment:get_address_schemaThe remainder if this section describes some concrete types that subtype Assignment."
},

{
    "location": "documentation/#Dynamic-Assignment-1",
    "page": "Gen Documentation",
    "title": "Dynamic Assignment",
    "category": "section",
    "text": "A DynamicAssignment is mutable, and can contain arbitrary values for its keys.DynamicAssignment()\nset_value! (with syntactic sugar Base.setindex!), will cause any previous value or sub-assignment at this addr to be deleted. it is an error if there is already a value present at some prefix of addr.\nset_subassmt!, will cause any previous value or sub-assignment at this addr to be deleted. it is an error if there is already a value present at some prefix of addr."
},

{
    "location": "documentation/#Static-Assignment-1",
    "page": "Gen Documentation",
    "title": "Static Assignment",
    "category": "section",
    "text": "A StaticAssignment is a immutable and contains only symbols as its keys for leaf nodes and for internal nodes. A StaticAssignment has type parametersR and T that are tuples of Symbols that are the keys of the leaf nodes and internal nodes respectively, so that code can be generated that is specialized to the particular set of keys in the trie:struct StaticAssignment{R,S,T,U} <: Assignment\n    leaf_nodes::NamedTuple{R,S}\n    internal_nodes::NamedTuple{T,U}\nend A StaticAssignment with leaf symbols :a and :b and internal key :c can be constructed using syntax like:trie = StaticAssignment((a=1, b=2), (c=inner_trie,))TODO: use generated functions in a lot more places, e.g. get_subassmtTODO: document static variants of getters:static_get_subassmt(assmt, ::Val{key}): throws a key error if the key isn\'t in the static address schema (get_subassmt would return an EmptyAssignment)\nNOTE: static_has_value(assmt, ::Val{key}) appears in the Static IR, but this an internal implementation detail, and not part of the \'static assignment interface\'."
},

{
    "location": "documentation/#Other-Concrete-Assignment-Types-1",
    "page": "Gen Documentation",
    "title": "Other Concrete Assignment Types",
    "category": "section",
    "text": "EmptyAssignment\nInternalVectorAssignment (TODO rename to DeepVectorAssignment)\nShallowVectorAssignment (TODO not yet implemented)\nAssignments produced from GFTraces\nAssignments produced "
},

{
    "location": "documentation/#Address-Selections-1",
    "page": "Gen Documentation",
    "title": "Address Selections",
    "category": "section",
    "text": "AddressSetTODO: document AddressSet API TODO: consider changing names of method in AddressSet APIAddressSchema\nDynamicAddressSet\nStaticAddressSet"
},

{
    "location": "documentation/#Gen.initialize",
    "page": "Gen Documentation",
    "title": "Gen.initialize",
    "category": "function",
    "text": "(trace::U, weight) = initialize(gen_fn::GenerativeFunction{T,U}, args::Tuple,\n                                assmt::Assignment)\n\nReturn a trace of a generative function that is consistent with the given assignment.\n\nBasic case\n\nGiven arguments x (args) and assignment u (assmt), sample t sim Q(cdot u x) and return the trace (x t) (trace).  Also return the weight (weight):\n\nfracP(t x)Q(t u x)\n\nGeneral case\n\nIdentical to the basic case, except that we also sample r sim Q(cdot x t), the trace is (x t r) and the weight is:\n\nfracP(t x)Q(t u x)\ncdot fracP(r x t)Q(r x t)\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.project",
    "page": "Gen Documentation",
    "title": "Gen.project",
    "category": "function",
    "text": "weight = project(trace::U, selection::AddressSet)\n\nEstimate the probability that the selected choices take the values they do in a trace. \n\nBasic case\n\nGiven a trace (x t) (trace) and a set of addresses A (selection), let u denote the restriction of t to A. Return the weight (weight):\n\nfracP(t x)Q(t u x)\n\nGeneral case\n\nIdentical to the basic case except that the previous trace is (x t r) and the weight is:\n\nfracP(t x)Q(t u x)\ncdot fracP(r x t)Q(r x t)\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.propose",
    "page": "Gen Documentation",
    "title": "Gen.propose",
    "category": "function",
    "text": "(assmt, weight, retval) = propose(gen_fn::GenerativeFunction, args::Tuple)\n\nSample an assignment and compute the probability of proposing that assignment.\n\nBasic case\n\nGiven arguments (args), sample t sim P(cdot x), and return t (assmt) and the weight (weight) P(t x).\n\nGeneral case\n\nIdentical to the basic case, except that we also sample r sim P(cdot x t), and the weight is:\n\nP(t x)\ncdot fracP(r x t)Q(r x t)\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.assess",
    "page": "Gen Documentation",
    "title": "Gen.assess",
    "category": "function",
    "text": "(weight, retval) = assess(gen_fn::GenerativeFunction, args::Tuple, assmt::Assignment)\n\nReturn the probability of proposing an assignment\n\nBasic case\n\nGiven arguments x (args) and an assignment t (assmt) such that P(t x)  0, return the weight (weight) P(t x).  It is an error if P(t x) = 0.\n\nGeneral case\n\nIdentical to the basic case except that we also sample r sim Q(cdot x t), and the weight is:\n\nP(t x)\ncdot fracP(r x t)Q(r x t)\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.force_update",
    "page": "Gen Documentation",
    "title": "Gen.force_update",
    "category": "function",
    "text": "(new_trace, weight, discard, retdiff) = force_update(args::Tuple, argdiff, trace,\n                                                     assmt::Assignment)\n\nUpdate a trace by changing the arguments and/or providing new values for some existing random choice(s) and values for any newly introduced random choice(s).\n\nBasic case\n\nGiven a previous trace (x t) (trace), new arguments x (args), and an assignment u (assmt), return a new trace (x t) (new_trace) that is consistent with u.  The values of choices in t are deterministically copied either from t or from u (with u taking precedence).  All choices in u must appear in t.  Also return an assignment v (discard) containing the choices in t that were overwritten by values from u, and any choices in t whose address does not appear in t.  Also return the weight (weight):\n\nfracP(t x)P(t x)\n\nGeneral case\n\nIdentical to the basic case except that the previous trace is (x t r), the new trace is (x t r) where r sim Q(cdot x t), and the weight is:\n\nfracP(t x)P(t x)\ncdot fracP(r x t) Q(r x t)P(r x t) Q(r x t)\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.fix_update",
    "page": "Gen Documentation",
    "title": "Gen.fix_update",
    "category": "function",
    "text": "(new_trace, weight, discard, retdiff) = fix_update(args::Tuple, argdiff, trace,\n                                                   assmt::Assignment)\n\nUpdate a trace, by changing the arguments and/or providing new values for some existing random choice(s).\n\nBasic case\n\nGiven a previous trace (x t) (trace), new arguments x (args), and an assignment u (assmt), return a new trace (x t) (new_trace) that is consistent with u.  Let u + t denote the merge of u and t (with u taking precedence).  Sample t sim Q(cdot u + t x). All addresses in u must appear in t and in t.  Also return an assignment v (discard) containing the values from t for addresses in u.  Also return the weight (weight):\n\nfracP(t x)P(t x) cdot fracQ(t v + t x)Q(t u + t x)\n\nGeneral case\n\nIdentical to the basic case except that the previous trace is (x t r), the new trace is (x t r) where r sim Q(cdot x t), and the weight is:\n\nfracP(t x)P(t x)\ncdot fracQ(t v + t x)Q(t u + t x)\ncdot fracP(r x t) Q(r x t)P(r x t) Q(r x t)\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.free_update",
    "page": "Gen Documentation",
    "title": "Gen.free_update",
    "category": "function",
    "text": "(new_trace, weight, retdiff) = free_update(args::Tuple, argdiff, trace,\n                                           selection::AddressSet)\n\nUpdate a trace by changing the arguments and/or randomly sampling new values for selected random choices.\n\nBasic case\n\nGiven a previous trace (x t) (trace), new arguments x (args), and a set of addresses A (selection), return a new trace (x t) (new_trace) such that t agrees with t on all addresses not in A (t and t may have different sets of addresses).  Let u denote the restriction of t to the complement of A.  Sample t sim Q(cdot u x).  Return the new trace (x t) (new_trace) and the weight (weight):\n\nfracP(t x)P(t x)\ncdot fracQ(t u x)Q(t u x)\n\nwhere u is the restriction of t to the complement of A.\n\nGeneral case\n\nIdentical to the basic case except that the previous trace is (x t r), the new trace is (x t r) where r sim Q(cdot x t), and the weight is:\n\nfracP(t x)P(t x)\ncdot fracQ(t u x)Q(t u x)\ncdot fracP(r x t) Q(r x t)P(r x t) Q(r x t)\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.extend",
    "page": "Gen Documentation",
    "title": "Gen.extend",
    "category": "function",
    "text": "(new_trace, weight, retdiff) = extend(args::Tuple, argdiff, trace, assmt::Assignment)\n\nExtend a trace with new random choices by changing the arguments.\n\nBasic case\n\nGiven a previous trace (x t) (trace), new arguments x (args), and an assignment u (assmt) that shares no addresses with t, return a new trace (x t) (new_trace) such that t agrees with t on all addresses in t and t agrees with u on all addresses in u. Sample t sim Q(cdot t + u x). Also return the weight (weight):\n\nfracP(t x)P(t x) Q(t t + u x)\n\nGeneral case\n\nIdentical to the basic case except that the previous trace is (x t r), and we also sample r sim Q(cdot t x), the new trace is (x t r), and the weight is:\n\nfracP(t x)P(t x) Q(t t + u x)\ncdot fracP(r x t) Q(r x t)P(r x t) Q(r x t)\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.backprop_params",
    "page": "Gen Documentation",
    "title": "Gen.backprop_params",
    "category": "function",
    "text": "arg_grads = backprop_params(trace, retgrad)\n\nIncrement gradient accumulators for parameters by the gradient of the log-probability of the trace.\n\nBasic case\n\nGiven a previous trace (x t) (trace) and a gradient with respect to the return value _y J (retgrad), return the following gradient (arg_grads) with respect to the arguments x:\n\n_x left( log P(t x) + J right)\n\nAlso increment the gradient accumulators for the static parameters Θ of the function by:\n\n_Θ left( log P(t x) + J right)\n\nGeneral case\n\nNot yet formalized.\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.backprop_trace",
    "page": "Gen Documentation",
    "title": "Gen.backprop_trace",
    "category": "function",
    "text": "(arg_grads, choice_values, choice_grads) = backprop_trace(trace, selection::AddressSet,\n                                                          retgrad)\n\nBasic case\n\nGiven a previous trace (x t) (trace) and a gradient with respect to the return value _y J (retgrad), return the following gradient (arg_grads) with respect to the arguments x:\n\n_x left( log P(t x) + J right)\n\nAlso given a set of addresses A (selection) that are continuous-valued random choices, return the folowing gradient (choice_grads) with respect to the values of these choices:\n\n_A left( log P(t x) + J right)\n\nAlso return the assignment (choice_values) that is the restriction of t to A.\n\nGeneral case\n\nNot yet formalized.\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.get_assmt",
    "page": "Gen Documentation",
    "title": "Gen.get_assmt",
    "category": "function",
    "text": "get_assmt(trace)\n\nReturn a value implementing the assignment interface\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.get_args",
    "page": "Gen Documentation",
    "title": "Gen.get_args",
    "category": "function",
    "text": "get_args(trace)\n\nReturn the argument tuple for a given execution.\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.get_retval",
    "page": "Gen Documentation",
    "title": "Gen.get_retval",
    "category": "function",
    "text": "get_retval(trace)\n\nReturn the return value of the given execution.\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.get_score",
    "page": "Gen Documentation",
    "title": "Gen.get_score",
    "category": "function",
    "text": "get_score(trace)\n\nBasic case\n\nReturn P(t x)\n\nGeneral case\n\nReturn P(r t x)  Q(r tx t)\n\n\n\n\n\n"
},

{
    "location": "documentation/#Traces-and-Generative-Functions-1",
    "page": "Gen Documentation",
    "title": "Traces and Generative Functions",
    "category": "section",
    "text": "A trace is a record of an execution of a generative function. There is no abstract type representing all traces. Generative functions implement the generative function interface, which is a set of methods that involve the execution traces and probabilistic behavior of generative functions. In the mathematical description of the interface methods, we denote arguments to a function by x, complete assignments of values to addresses of random choices (containing all the random choices made during some execution) by t and partial assignments by either u or v. We denote a trace of a generative function by the tuple (x t). We say that two assignments u and t agree when they assign addresses that appear in both assignments to the same values (they can different or even disjoint sets of addresses and still agree). A generative function is associated with a family of probability distributions P(t x) on assignments t, parameterized by arguments x, and a second family of distributions Q(t u x) on assignments t parameterized by partial assignment u and arguments x. Q is called the internal proposal family of the generative function, and satisfies that if u and t agree then P(t x)  0 if and only if Q(t x u)  0, and that Q(t x u)  0 implies that u and t agree. See the Gen technical report for additional details.Generative functions may also use non-addressable random choices, denoted r. Unlike regular (addressable) random choices, non-addressable random choices do not have addresses, and the value of non-addressable random choices is not exposed through the generative function interface. However, the state of non-addressable random choices is maintained in the trace. A trace that contains non-addressable random choices is denoted (x t r). Non-addressable random choices manifest to the user of the interface as stochasticity in weights returned by generative function interface methods. The behavior of non-addressable random choices is defined by an additional pair of families of distributions associated with the generative function, denoted Q(r x t) and P(r x t), which are defined for P(t x)  0, and which satisfy Q(r x t)  0 if and only if P(r x t)  0. For each generative function below, we describe its semantics first in the basic setting where there is no non-addressable random choices, and then in the more general setting that may include non-addressable random choices.initialize\nproject\npropose\nassess\nforce_update\nfix_update\nfree_update\nextend\nbackprop_params\nbackprop_trace\nget_assmt\nget_args\nget_retval\nget_scoreTODO: document has_argument_grads"
},

{
    "location": "documentation/#Distributions-1",
    "page": "Gen Documentation",
    "title": "Distributions",
    "category": "section",
    "text": "Probability distributions are singleton types whose supertype is Distribution{T}, where T indicates the data type of the random sample.abstract type Distribution{T} endBy convention, distributions have a global constant lower-case name for the singleton value. For example:struct Bernoulli <: Distribution{Bool} end\nconst bernoulli = Bernoulli()Distributions must implement two methods, random and logpdf.random returns a random sample from the distribution:x::Bool = random(bernoulli, 0.5)\nx::Bool = random(Bernoulli(), 0.5)logpdf returns the log probability (density) of the distribution at a given value:logpdf(bernoulli, false, 0.5)\nlogpdf(Bernoulli(), false, 0.5)Distribution values are also callable, which is a syntactic sugar with the same behavior of calling random:bernoulli(0.5) # identical to random(bernoulli, 0.5) and random(Bernoulli(), 0.5)"
},

{
    "location": "documentation/#Gradients-of-Distributions-1",
    "page": "Gen Documentation",
    "title": "Gradients of Distributions",
    "category": "section",
    "text": "Distributions may also implement logpdf_grad, which returns the gradient of the log probability (density) with respect to the random sample and the parameters, as a tuple:(grad_sample, grad_mu, grad_std) = logpdf_grad(normal, 1.324, 0.0, 1.0)The partial derivative of the log probability (density) with respect to the random sample, or one of the parameters, might not always exist. Distributions indicate which partial derivatives exist using the methods has_output_grad and has_argument_grads:has_output_grad(::Normal) = true\nhas_argument_grads(::Normal) = (true, true)If a particular partial derivative does not exist, that field of the tuple returned by logpdf_grad should be nothing."
},

{
    "location": "documentation/#Gen.bernoulli",
    "page": "Gen Documentation",
    "title": "Gen.bernoulli",
    "category": "constant",
    "text": "bernoulli(prob_true::Real)\n\nSamples a Bool value which is true with given probability\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.normal",
    "page": "Gen Documentation",
    "title": "Gen.normal",
    "category": "constant",
    "text": "normal(mu::Real, std::Real)\n\nSamples a Float64 value from a normal distribution.\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.mvnormal",
    "page": "Gen Documentation",
    "title": "Gen.mvnormal",
    "category": "constant",
    "text": "mvnormal(mu::AbstractVector{T}, cov::AbstractMatrix{U}} where {T<:Real,U<:Real}\n\nSamples a Vector{Float64} value from a multivariate normal distribution.\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.gamma",
    "page": "Gen Documentation",
    "title": "Gen.gamma",
    "category": "constant",
    "text": "gamma(shape::Real, scale::Real)\n\nSample a Float64 from a gamma distribution.\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.inv_gamma",
    "page": "Gen Documentation",
    "title": "Gen.inv_gamma",
    "category": "constant",
    "text": "inv_gamma(shape::Real, scale::Real)\n\nSample a Float64 from a inverse gamma distribution.\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.beta",
    "page": "Gen Documentation",
    "title": "Gen.beta",
    "category": "constant",
    "text": "beta(alpha::Real, beta::Real)\n\nSample a Float64 from a beta distribution.\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.categorical",
    "page": "Gen Documentation",
    "title": "Gen.categorical",
    "category": "constant",
    "text": "categorical(probs::AbstractArray{U, 1}) where {U <: Real}\n\nGiven a vector of probabilities probs where sum(probs) = 1, sample an Int i from the set {1, 2, .., length(probs)} with probability probs[i].\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.uniform",
    "page": "Gen Documentation",
    "title": "Gen.uniform",
    "category": "constant",
    "text": "uniform(low::Real, high::Real)\n\nSample a Float64 from the uniform distribution on the interval [low, high].\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.uniform_discrete",
    "page": "Gen Documentation",
    "title": "Gen.uniform_discrete",
    "category": "constant",
    "text": "uniform_discrete(low::Integer, high::Integer)\n\nSample an Int from the uniform distribution on the set {low, low + 1, ..., high-1, high}.\n\n\n\n\n\n"
},

{
    "location": "documentation/#Gen.poisson",
    "page": "Gen Documentation",
    "title": "Gen.poisson",
    "category": "constant",
    "text": "poisson(lambda::Real)\n\nSample an Int from the Poisson distribution with rate lambda.\n\n\n\n\n\n"
},

{
    "location": "documentation/#Built-In-Distributions-1",
    "page": "Gen Documentation",
    "title": "Built-In Distributions",
    "category": "section",
    "text": "bernoulli\nnormal\nmvnormal\ngamma\ninv_gamma\nbeta\ncategorical\nuniform\nuniform_discrete\npoisson"
},

{
    "location": "documentation/#Trie-1",
    "page": "Gen Documentation",
    "title": "Trie",
    "category": "section",
    "text": ""
},

{
    "location": "documentation/#Modeling-DSLs-1",
    "page": "Gen Documentation",
    "title": "Modeling DSLs",
    "category": "section",
    "text": ""
},

{
    "location": "documentation/#Dynamic-DSL-1",
    "page": "Gen Documentation",
    "title": "Dynamic DSL",
    "category": "section",
    "text": "TODO: remove the @ad return value differentiation flag"
},

]}
