var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#Gen.jl-1",
    "page": "Home",
    "title": "Gen.jl",
    "category": "section",
    "text": "A general-purpose probabilistic programming system with programmable inference, embedded in JuliaPages = [\n    \"getting_started.md\",\n    \"tutorials.md\",\n    \"guide.md\",\n]\nDepth = 2ReferencePages = [\n    \"ref/modeling.md\",\n    \"ref/combinators.md\",\n    \"ref/assignments.md\",\n    \"ref/selections.md\",\n    \"ref/parameter_optimization.md\",\n    \"ref/inference.md\",\n    \"ref/gfi.md\",\n    \"ref/distributions.md\"\n    \"ref/extending.md\",\n]\nDepth = 2"
},

{
    "location": "getting_started/#",
    "page": "Getting Started",
    "title": "Getting Started",
    "category": "page",
    "text": ""
},

{
    "location": "getting_started/#Getting-Started-1",
    "page": "Getting Started",
    "title": "Getting Started",
    "category": "section",
    "text": ""
},

{
    "location": "getting_started/#Installation-1",
    "page": "Getting Started",
    "title": "Installation",
    "category": "section",
    "text": "First, obtain Julia 1.3 or later, available here.The Gen package can be installed with the Julia package manager. From the Julia REPL, type ] to enter the Pkg REPL mode and then run:pkg> add GenTo test the installation locally, you can run the tests with:using Pkg; Pkg.test(\"Gen\")"
},

{
    "location": "getting_started/#Example-1",
    "page": "Getting Started",
    "title": "Example",
    "category": "section",
    "text": "Let\'s write a short Gen program that does Bayesian linear regression: given a set of points in the (x, y) plane, we want to find a line that fits them well.There are three main components to a typical Gen program.First, we define a generative model: a Julia function, extended with some extra syntax, that, conceptually, simulates a fake dataset. The model below samples slope and intercept parameters, and then for each of the x-coordinates that it accepts as input, samples a corresponding y-coordinate. We name the random choices we make with @trace, so we can refer to them in our inference program.using Gen\n\n@gen function my_model(xs::Vector{Float64})\n    slope = @trace(normal(0, 2), :slope)\n    intercept = @trace(normal(0, 10), :intercept)\n    for (i, x) in enumerate(xs)\n        @trace(normal(slope * x + intercept, 1), \"y-$i\")\n    end\nendSecond, we write an inference program that implements an algorithm for manipulating the execution traces of the model. Inference programs are regular Julia code, and make use of Gen\'s standard inference library.The inference program below takes in a data set, and runs an iterative MCMC algorithm to fit slope and intercept parameters:function my_inference_program(xs::Vector{Float64}, ys::Vector{Float64}, num_iters::Int)\n    # Create a set of constraints fixing the \n    # y coordinates to the observed y values\n    constraints = choicemap()\n    for (i, y) in enumerate(ys)\n        constraints[\"y-$i\"] = y\n    end\n    \n    # Run the model, constrained by `constraints`,\n    # to get an initial execution trace\n    (trace, _) = generate(my_model, (xs,), constraints)\n    \n    # Iteratively update the slope then the intercept,\n    # using Gen\'s metropolis_hastings operator.\n    for iter=1:num_iters\n        (trace, _) = metropolis_hastings(trace, select(:slope))\n        (trace, _) = metropolis_hastings(trace, select(:intercept))\n    end\n    \n    # From the final trace, read out the slope and\n    # the intercept.\n    choices = get_choices(trace)\n    return (choices[:slope], choices[:intercept])\nendFinally, we run the inference program on some data, and get the results:xs = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]\nys = [8.23, 5.87, 3.99, 2.59, 0.23, -0.66, -3.53, -6.91, -7.24, -9.90]\n(slope, intercept) = my_inference_program(xs, ys, 1000)\nprintln(\"slope: $slope, intercept: $intercept\")"
},

{
    "location": "tutorials/#",
    "page": "Tutorials",
    "title": "Tutorials",
    "category": "page",
    "text": ""
},

{
    "location": "tutorials/#Tutorials-and-Case-Studies-1",
    "page": "Tutorials",
    "title": "Tutorials and Case Studies",
    "category": "section",
    "text": "See Gen Quickstart repository for tutorials and case studiesAdditional examples are available in the GenExamples.jl repository."
},

{
    "location": "ref/gfi/#",
    "page": "Generative Functions",
    "title": "Generative Functions",
    "category": "page",
    "text": ""
},

{
    "location": "ref/gfi/#Generative-Functions-1",
    "page": "Generative Functions",
    "title": "Generative Functions",
    "category": "section",
    "text": "One of the core abstractions in Gen is the generative function. Generative functions are used to represent a variety of different types of probabilistic computations including generative models, inference models, custom proposal distributions, and variational approximations."
},

{
    "location": "ref/gfi/#Gen.GenerativeFunction",
    "page": "Generative Functions",
    "title": "Gen.GenerativeFunction",
    "category": "type",
    "text": "GenerativeFunction{T,U <: Trace}\n\nAbstract type for a generative function with return value type T and trace type U.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Introduction-1",
    "page": "Generative Functions",
    "title": "Introduction",
    "category": "section",
    "text": "Generative functions are represented by the following abstact type:GenerativeFunctionThere are various kinds of generative functions, which are represented by concrete subtypes of GenerativeFunction. For example, the Built-in Modeling Language allows generative functions to be constructed using Julia function definition syntax:@gen function foo(a, b=0)\n    if @trace(bernoulli(0.5), :z)\n        return a + b + 1\n    else\n        return a + b\n    end\nendUsers can also extend Gen by implementing their own Custom generative function types, which can be new modeling languages, or just specialized optimized implementations of a fragment of a specific model.Generative functions behave like Julia functions in some respects. For example, we can call a generative function foo on arguments and get an output value using regular Julia call syntax:julia> foo(2, 4)\n7However, generative functions are distinct from Julia functions because they support additional behaviors, described in the remainder of this section."
},

{
    "location": "ref/gfi/#Mathematical-concepts-1",
    "page": "Generative Functions",
    "title": "Mathematical concepts",
    "category": "section",
    "text": "Generative functions represent computations that accept some arguments, may use randomness internally, return an output, and cannot mutate externally observable state. We represent the randomness used during an execution of a generative function as a choice map from unique addresses to values of random choices, denoted t  A to V where A is a finite (but not a priori bounded) address set and V is a set of possible values that random choices can take. In this section, we assume that random choices are discrete to simplify notation. We say that two choice maps t and s agree if they assign the same value for any address that is in both of their domains.Generative functions may also use non-addressable randomness, which is not included in the map t. We denote non-addressable randomness by r. Untraced randomness is useful for example, when calling black box Julia code that implements a randomized algorithm.The observable behavior of every generative function is defined by the following mathematical objects:"
},

{
    "location": "ref/gfi/#Input-type-1",
    "page": "Generative Functions",
    "title": "Input type",
    "category": "section",
    "text": "The set of valid argument tuples to the function, denoted X."
},

{
    "location": "ref/gfi/#Probability-distribution-family-1",
    "page": "Generative Functions",
    "title": "Probability distribution family",
    "category": "section",
    "text": "A family of probability distributions p(t r x) on maps t from random choice addresses to their values, and non-addressable randomness r, indexed by arguments x, for all x in X. Note that the distribution must be normalized:sum_t r p(t r x) = 1  mboxfor all  x in XThis corresponds to a requirement that the function terminate with probabability 1 for all valid arguments. We use p(t x) to denote the marginal distribution on the map t:p(t x) = sum_r p(t r x)And we denote the conditional distribution on non-addressable randomness r, given the map t, as:p(r x t) = p(t r x)  p(t x)"
},

{
    "location": "ref/gfi/#Return-value-function-1",
    "page": "Generative Functions",
    "title": "Return value function",
    "category": "section",
    "text": "A (deterministic) function f that maps the tuple (x t) of the arguments and the choice map to the return value of the function (which we denote by y). Note that the return value cannot depend on the non-addressable randomness."
},

{
    "location": "ref/gfi/#Auxiliary-state-1",
    "page": "Generative Functions",
    "title": "Auxiliary state",
    "category": "section",
    "text": "Generative functions may expose additional auxiliary state associated with an execution, besides the choice map and the return value. This auxiliary state is a function z = h(x t r) of the arguments, choice map, and non-addressable randomness. Like the choice map, the auxiliary state is indexed by addresses. We require that the addresses of auxiliary state are disjoint from the addresses in the choice map. Note that when a generative function is called within a model, the auxiliary state is not available to the caller. It is typically used by inference programs, for logging and for caching the results of deterministic computations that would otherwise need to be reconstructed."
},

{
    "location": "ref/gfi/#Internal-proposal-distribution-family-1",
    "page": "Generative Functions",
    "title": "Internal proposal distribution family",
    "category": "section",
    "text": "A family of probability distributions q(t x u) on maps t from random choice addresses to their values, indexed by tuples (x u) where u is a map from random choice addresses to values, and where x are the arguments to the function. It must satisfy the following conditions:sum_t q(t x u) = 1  mboxfor all  x in X up(t x)  0 mbox if and only if  q(t x u)  0 mbox for all  u mbox where  u mbox and  t mbox agree q(t x u)  0 mbox implies that  u mbox and  t mbox agree There is also a family of probability distributions q(r x t) on non-addressable randomness, that satisfies:q(r x t)  0 mbox if and only if  p(r x t)  0"
},

{
    "location": "ref/gfi/#Gen.Trace",
    "page": "Generative Functions",
    "title": "Gen.Trace",
    "category": "type",
    "text": "Trace\n\nAbstract type for a trace of a generative function.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Traces-1",
    "page": "Generative Functions",
    "title": "Traces",
    "category": "section",
    "text": "An execution trace (or just trace) is a record of an execution of a generative function. Traces are the primary data structures manipulated by Gen inference programs. There are various methods for producing, updating, and inspecting traces. Traces contain:the arguments to the generative function\nthe choice map\nthe return value\nauxiliary state\nother implementation-specific state that is not exposed to the caller or user of the generative function, but is used internally to facilitate e.g. incremental updates to executions and automatic differentiation\nany necessary record of the non-addressable randomnessDifferent concrete types of generative functions use different data structures and different Julia types for their traces, but traces are subtypes of Trace.TraceThe concrete trace type that a generative function uses is the second type parameter of the GenerativeFunction abstract type. For example, the trace type of DynamicDSLFunction is DynamicDSLTrace.A generative function can be executed to produce a trace of the execution using simulate:trace = simulate(foo, (a, b))A traced execution that satisfies constraints on the choice map can be generated using generate:trace, weight = generate(foo, (a, b), choicemap((:z, false)))There are various methods for inspecting traces, including:get_args (returns the arguments to the function)\nget_retval (returns the return value of the function)\nget_choices (returns the choice map)\nget_score (returns the log probability that the random choices took the values they did)\nget_gen_fn (returns a reference to the generative function)You can also access the values in the choice map and the auxiliary state of the trace by passing the address to Base.getindex. For example, to retrieve the value of random choice at address :z:z = trace[:z]When a generative function has default values specified for trailing arguments, those arguments can be left out when calling simulate, generate, and other functions provided by the generative function interface. The default values will automatically be filled in:julia> trace = simulate(foo, (2,));\njulia> get_args(trace)\n(2, 0)"
},

{
    "location": "ref/gfi/#Updating-traces-1",
    "page": "Generative Functions",
    "title": "Updating traces",
    "category": "section",
    "text": "It is often important to incrementally modify the trace of a generative function (e.g. within MCMC, numerical optimization, sequential Monte Carlo, etc.). In Gen, traces are functional data structures, meaning they can be treated as immutable values. There are several methods that take a trace of a generative function as input and return a new trace of the generative function based on adjustments to the execution history of the function. We will illustrate these methods using the following generative function:@gen function bar()\n    val = @trace(bernoulli(0.3), :a)\n    if @trace(bernoulli(0.4), :b)\n        val = @trace(bernoulli(0.6), :c) && val\n    else\n        val = @trace(bernoulli(0.1), :d) && val\n    end\n    val = @trace(bernoulli(0.7), :e) && val\n    return val\nendSuppose we have a trace (trace) of bar with initial choices:│\n├── :a : false\n│\n├── :b : true\n│\n├── :c : false\n│\n└── :e : trueNote that address :d is not present because the branch in which :d is sampled was not taken because random choice :b had value true."
},

{
    "location": "ref/gfi/#Update-1",
    "page": "Generative Functions",
    "title": "Update",
    "category": "section",
    "text": "The update method takes a trace and generates an adjusted trace that is consistent with given changes to the arguments to the function, and changes to the values of random choices made.Example. Suppose we run update on the example trace, with the following constraints:│\n├── :b : false\n│\n└── :d : trueconstraints = choicemap((:b, false), (:d, true))\n(new_trace, w, _, discard) = update(trace, (), (), constraints)Then get_choices(new_trace) will be:│\n├── :a : false\n│\n├── :b : false\n│\n├── :d : true\n│\n└── :e : trueand discard will be:│\n├── :b : true\n│\n└── :c : falseNote that the discard contains both the previous values of addresses that were overwritten, and the values for addresses that were in the previous trace but are no longer in the new trace. The weight (w) is computed as:p(t x) = 07  04  04  07 = 00784\np(t x) = 07  06  01  07 = 00294\nw = log p(t x)p(t x) = log 0029400784 = log 0375Example. Suppose we run update on the example trace, with the following constraints, which do not contain a value for :d:│\n└── :b : falseconstraints = choicemap((:b, false))\n(new_trace, w, _, discard) = update(trace, (), (), constraints)Then get_choices(new_trace) will be:│\n├── :a : false\n│\n├── :b : false\n│\n├── :d : true\n│\n└── :e : truewith probability 0.1, or:│\n├── :a : false\n│\n├── :b : false\n│\n├── :d : false\n│\n└── :e : truewith probability 0.9. Also, discard will be:│\n├── :b : true\n│\n└── :c : falseIf the former case occurs and :d is assigned to true, then the weight (w) is computed as:p(t x) = 07  04  04  07 = 00784\np(t x) = 07  06  01  07 = 00294\nq(t x t + u) = 01\nw = log p(t x)(p(t x) q(t x t + u)) = log 00294(00784 cdot 01) = log (375)"
},

{
    "location": "ref/gfi/#Regenerate-1",
    "page": "Generative Functions",
    "title": "Regenerate",
    "category": "section",
    "text": "The regenerate method takes a trace and generates an adjusted trace that is consistent with a change to the arguments to the function, and also generates new values for selected random choices.Example. Suppose we run regenerate on the example trace, with selection :a and :b:(new_trace, w, _) = regenerate(trace, (), (), select(:a, :b))Then, a new value for :a will be sampled from bernoulli(0.3), and a new value for :b will be sampled from bernoulli(0.4). If the new value for :b is true, then the previous value for :c (false) will be retained. If the new value for :b is false, then a new value for :d will be sampled from bernoulli(0.7). The previous value for :c will always be retained. Suppose the new value for :a is true, and the new value for :b is true. Then get_choices(new_trace) will be:│\n├── :a : true\n│\n├── :b : true\n│\n├── :c : false\n│\n└── :e : trueThe weight (w) is log 1 = 0."
},

{
    "location": "ref/gfi/#Gen.NoChange",
    "page": "Generative Functions",
    "title": "Gen.NoChange",
    "category": "type",
    "text": "NoChange\n\nThe value did not change.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.UnknownChange",
    "page": "Generative Functions",
    "title": "Gen.UnknownChange",
    "category": "type",
    "text": "UnknownChange\n\nNo information is provided about the change to the value.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Argdiffs-1",
    "page": "Generative Functions",
    "title": "Argdiffs",
    "category": "section",
    "text": "In addition to the input trace, and other arguments that indicate how to adjust the trace, each of these methods also accepts an args argument and an argdiffs argument, both of which are tuples. The args argument contains the new arguments to the generative function, which may differ from the previous arguments to the generative function (which can be retrieved by applying get_args to the previous trace). In many cases, the adjustment to the execution specified by the other arguments to these methods is \'small\' and only affects certain parts of the computation. Therefore, it is often possible to generate the new trace and the appropriate log probability ratios required for these methods without revisiting every state of the computation of the generative function.To enable this, the argdiffs argument provides additional information about the difference between each of the previous arguments to the generative function, and its new argument value. This argdiff information permits the implementation of the update method to avoid inspecting the entire argument data structure to identify which parts were updated. Note that the correctness of the argdiff is in general not verified by Gen–-passing incorrect argdiff information may result in incorrect behavior.The trace update methods for all generative functions above should accept at least the following types of argdiffs:NoChange\nUnknownChangeGenerative functions may also be able to process more specialized diff data types for each of their arguments, that allow more precise information about the different to be supplied."
},

{
    "location": "ref/gfi/#Retdiffs-1",
    "page": "Generative Functions",
    "title": "Retdiffs",
    "category": "section",
    "text": "To enable generative functions that invoke other functions to efficiently make use of incremental computation, the trace update methods of generative functions also return a retdiff value, which provides information about the difference in the return value of the previous trace an the return value of the new trace."
},

{
    "location": "ref/gfi/#Differentiable-programming-1",
    "page": "Generative Functions",
    "title": "Differentiable programming",
    "category": "section",
    "text": "The trace of a generative function may support computation of gradients of its log probability with respect to some subset of (i) its arguments, (ii) values of random choice, and (iii) any of its trainable parameters (see below).To compute gradients with respect to the arguments as well as certain selected random choices, use:choice_gradientsTo compute gradients with respect to the arguments, and to increment a stateful gradient accumulator for the trainable parameters of the generative function, use:accumulate_param_gradients!A generative function statically reports whether or not it is able to compute gradients with respect to each of its arguments, through the function has_argument_grads."
},

{
    "location": "ref/gfi/#Trainable-parameters-1",
    "page": "Generative Functions",
    "title": "Trainable parameters",
    "category": "section",
    "text": "The trainable parameters of a generative function are (unlike arguments and random choices) state of the generative function itself, and are not contained in the trace. Generative functions that have trainable parameters maintain gradient accumulators for these parameters, which get incremented by the gradient induced by the given trace by a call to accumulate_param_gradients!. Users then use these accumulated gradients to update to the values of the trainable parameters."
},

{
    "location": "ref/gfi/#Return-value-gradient-1",
    "page": "Generative Functions",
    "title": "Return value gradient",
    "category": "section",
    "text": "The set of elements (either arguments, random choices, or trainable parameters) for which gradients are available is called the gradient source set. If the return value of the function is conditionally dependent on any element in the gradient source set given the arguments and values of all other random choices, for all possible traces of the function, then the generative function requires a return value gradient to compute gradients with respect to elements of the gradient source set. This static property of the generative function is reported by accepts_output_grad."
},

{
    "location": "ref/gfi/#Gen.simulate",
    "page": "Generative Functions",
    "title": "Gen.simulate",
    "category": "function",
    "text": "trace = simulate(gen_fn, args)\n\nExecute the generative function and return the trace.\n\nGiven arguments (args), sample t sim p(cdot x) and r sim p(cdot x t), and return a trace with choice map t.\n\nIf gen_fn has optional trailing arguments (i.e., default values are provided), the optional arguments can be omitted from the args tuple. The generated trace  will have default values filled in.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.generate",
    "page": "Generative Functions",
    "title": "Gen.generate",
    "category": "function",
    "text": "(trace::U, weight) = generate(gen_fn::GenerativeFunction{T,U}, args::Tuple)\n\nReturn a trace of a generative function.\n\n(trace::U, weight) = generate(gen_fn::GenerativeFunction{T,U}, args::Tuple,\n                                constraints::ChoiceMap)\n\nReturn a trace of a generative function that is consistent with the given constraints on the random choices.\n\nGiven arguments x (args) and assignment u (constraints) (which is empty for the first form), sample t sim q(cdot u x) and r sim q(cdot x t), and return the trace (x t r) (trace). Also return the weight (weight):\n\nlog fracp(t r x)q(t u x) q(r x t)\n\nIf gen_fn has optional trailing arguments (i.e., default values are provided), the optional arguments can be omitted from the args tuple. The generated trace  will have default values filled in.\n\nExample without constraints:\n\n(trace, weight) = generate(foo, (2, 4))\n\nExample with constraint that address :z takes value true.\n\n(trace, weight) = generate(foo, (2, 4), choicemap((:z, true))\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.update",
    "page": "Generative Functions",
    "title": "Gen.update",
    "category": "function",
    "text": "(new_trace, weight, retdiff, discard) = update(trace, args::Tuple, argdiffs::Tuple,\n                                               constraints::ChoiceMap)\n\nUpdate a trace by changing the arguments and/or providing new values for some existing random choice(s) and values for some newly introduced random choice(s).\n\nGiven a previous trace (x t r) (trace), new arguments x (args), and a map u (constraints), return a new trace (x t r) (new_trace) that is consistent with u.  The values of choices in t are either copied from t or from u (with u taking precedence) or are sampled from the internal proposal distribution.  All choices in u must appear in t.  Also return an assignment v (discard) containing the choices in t that were overwritten by values from u, and any choices in t whose address does not appear in t. Sample t sim q(cdot x t + u), and r sim q(cdot x t), where t + u is the choice map obtained by merging t and u with u taking precedence for overlapping addresses.  Also return a weight (weight):\n\nlog fracp(r t x) q(r x t)p(r t x) q(r x t) q(t x t + u)\n\nNote that argdiffs is expected to be the same length as args. If the function that generated trace supports default values for trailing arguments, then these arguments can be omitted from args and argdiffs. Note that if the original trace was generated using non-default argument values, then for each optional argument that is omitted, the old value will be over-written by the default argument value in the updated trace.\n\n\n\n\n\n(new_trace, weight, retdiff, discard) = update(trace, constraints::ChoiceMap)\n\nShorthand variant of update which assumes the arguments are unchanged.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.regenerate",
    "page": "Generative Functions",
    "title": "Gen.regenerate",
    "category": "function",
    "text": "(new_trace, weight, retdiff) = regenerate(trace, args::Tuple, argdiffs::Tuple,\n                                          selection::Selection)\n\nUpdate a trace by changing the arguments and/or randomly sampling new values for selected random choices using the internal proposal distribution family.\n\nGiven a previous trace (x t r) (trace), new arguments x (args), and a set of addresses A (selection), return a new trace (x t) (new_trace) such that t agrees with t on all addresses not in A (t and t may have different sets of addresses).  Let u denote the restriction of t to the complement of A.  Sample t sim Q(cdot u x) and sample r sim Q(cdot x t). Return the new trace (x t r) (new_trace) and the weight (weight):\n\nlog fracp(r t x) q(t u x) q(r x t)p(r t x) q(t u x) q(r x t)\n\nwhere u is the restriction of t to the complement of A.\n\nNote that argdiffs is expected to be the same length as args. If the function that generated trace supports default values for trailing arguments, then these arguments can be omitted from args and argdiffs. Note that if the original trace was generated using non-default argument values, then for each optional argument that is omitted, the old value will be over-written by the default argument value in the regenerated trace.\n\n\n\n\n\n(new_trace, weight, retdiff) = regenerate(trace, selection::Selection)\n\nShorthand variant of regenerate which assumes the arguments are unchanged.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.get_args",
    "page": "Generative Functions",
    "title": "Gen.get_args",
    "category": "function",
    "text": "get_args(trace)\n\nReturn the argument tuple for a given execution.\n\nExample:\n\nargs::Tuple = get_args(trace)\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.get_retval",
    "page": "Generative Functions",
    "title": "Gen.get_retval",
    "category": "function",
    "text": "get_retval(trace)\n\nReturn the return value of the given execution.\n\nExample for generative function with return type T:\n\nretval::T = get_retval(trace)\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.get_choices",
    "page": "Generative Functions",
    "title": "Gen.get_choices",
    "category": "function",
    "text": "get_choices(trace)\n\nReturn a value implementing the assignment interface\n\nNote that the value of any non-addressed randomness is not externally accessible.\n\nExample:\n\nchoices::ChoiceMap = get_choices(trace)\nz_val = choices[:z]\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.get_score",
    "page": "Generative Functions",
    "title": "Gen.get_score",
    "category": "function",
    "text": "get_score(trace)\n\nReturn:\n\nlog fracp(t r x)q(r x t)\n\nWhen there is no non-addressed randomness, this simplifies to the log probability log p(t x).\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.get_gen_fn",
    "page": "Generative Functions",
    "title": "Gen.get_gen_fn",
    "category": "function",
    "text": "gen_fn::GenerativeFunction = get_gen_fn(trace)\n\nReturn the generative function that produced the given trace.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Base.getindex",
    "page": "Generative Functions",
    "title": "Base.getindex",
    "category": "function",
    "text": "value = getindex(trace::Trace, addr)\n\nGet the value of the random choice, or auxiliary state (e.g. return value of inner function call), at address addr.\n\n\n\n\n\nretval = getindex(trace::Trace)\nretval = trace[]\n\nSynonym for get_retval.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.project",
    "page": "Generative Functions",
    "title": "Gen.project",
    "category": "function",
    "text": "weight = project(trace::U, selection::Selection)\n\nEstimate the probability that the selected choices take the values they do in a trace.\n\nGiven a trace (x t r) (trace) and a set of addresses A (selection), let u denote the restriction of t to A. Return the weight (weight):\n\nlog fracp(r t x)q(t u x) q(r x t)\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.propose",
    "page": "Generative Functions",
    "title": "Gen.propose",
    "category": "function",
    "text": "(choices, weight, retval) = propose(gen_fn::GenerativeFunction, args::Tuple)\n\nSample an assignment and compute the probability of proposing that assignment.\n\nGiven arguments (args), sample t sim p(cdot x) and r sim p(cdot x t), and return t (choices) and the weight (weight):\n\nlog fracp(r t x)q(r x t)\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.assess",
    "page": "Generative Functions",
    "title": "Gen.assess",
    "category": "function",
    "text": "(weight, retval) = assess(gen_fn::GenerativeFunction, args::Tuple, choices::ChoiceMap)\n\nReturn the probability of proposing an assignment\n\nGiven arguments x (args) and an assignment t (choices) such that p(t x)  0, sample r sim q(cdot x t) and return the weight (weight):\n\nlog fracp(r t x)q(r x t)\n\nIt is an error if p(t x) = 0.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.has_argument_grads",
    "page": "Generative Functions",
    "title": "Gen.has_argument_grads",
    "category": "function",
    "text": "bools::Tuple = has_argument_grads(gen_fn::Union{GenerativeFunction,Distribution})\n\nReturn a tuple of booleans indicating whether a gradient is available for each of its arguments.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.accepts_output_grad",
    "page": "Generative Functions",
    "title": "Gen.accepts_output_grad",
    "category": "function",
    "text": "req::Bool = accepts_output_grad(gen_fn::GenerativeFunction)\n\nReturn a boolean indicating whether the return value is dependent on any of the gradient source elements for any trace.\n\nThe gradient source elements are:\n\nAny argument whose position is true in has_argument_grads\nAny trainable parameter\nRandom choices made at a set of addresses that are selectable by choice_gradients.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.accumulate_param_gradients!",
    "page": "Generative Functions",
    "title": "Gen.accumulate_param_gradients!",
    "category": "function",
    "text": "arg_grads = accumulate_param_gradients!(trace, retgrad=nothing, scale_factor=1.)\n\nIncrement gradient accumulators for parameters by the gradient of the log-probability of the trace, optionally scaled, and return the gradient with respect to the arguments (not scaled).\n\nGiven a previous trace (x t) (trace) and a gradient with respect to the return value _y J (retgrad), return the following gradient (arg_grads) with respect to the arguments x:\n\n_x left( log P(t x) + J right)\n\nThe length of arg_grads will be equal to the number of arguments to the function that generated trace (including any optional trailing arguments). If an argument is not annotated with (grad), the corresponding value in arg_grads will be nothing.\n\nAlso increment the gradient accumulators for the trainable parameters Θ of the function by:\n\n_Θ left( log P(t x) + J right)\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.choice_gradients",
    "page": "Generative Functions",
    "title": "Gen.choice_gradients",
    "category": "function",
    "text": "(arg_grads, choice_values, choice_grads) = choice_gradients(\n    trace, selection=EmptySelection(), retgrad=nothing)\n\nGiven a previous trace (x t) (trace) and a gradient with respect to the return value _y J (retgrad), return the following gradient (arg_grads) with respect to the arguments x:\n\n_x left( log P(t x) + J right)\n\nThe length of arg_grads will be equal to the number of arguments to the function that generated trace (including any optional trailing arguments). If an argument is not annotated with (grad), the corresponding value in arg_grads will be nothing.\n\nAlso given a set of addresses A (selection) that are continuous-valued random choices, return the folowing gradient (choice_grads) with respect to the values of these choices:\n\n_A left( log P(t x) + J right)\n\nThe gradient is represented as a choicemap whose value at (hierarchical) address addr is Jttextttaddr.\n\nAlso return the choicemap (choice_values) that is the restriction of t to A.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.get_params",
    "page": "Generative Functions",
    "title": "Gen.get_params",
    "category": "function",
    "text": "get_params(gen_fn::GenerativeFunction)\n\nReturn an iterable over the trainable parameters of the generative function.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Generative-function-interface-1",
    "page": "Generative Functions",
    "title": "Generative function interface",
    "category": "section",
    "text": "The complete set of methods in the generative function interface (GFI) is:simulate\ngenerate\nupdate\nregenerate\nget_args\nget_retval\nget_choices\nget_score\nget_gen_fn\nBase.getindex\nproject\npropose\nassess\nhas_argument_grads\naccepts_output_grad\naccumulate_param_gradients!\nchoice_gradients\nget_params"
},

{
    "location": "ref/distributions/#",
    "page": "Probability Distributions",
    "title": "Probability Distributions",
    "category": "page",
    "text": ""
},

{
    "location": "ref/distributions/#Probability-Distributions-1",
    "page": "Probability Distributions",
    "title": "Probability Distributions",
    "category": "section",
    "text": "Gen provides a library of built-in probability distributions, and three ways of defining custom distributions, each of which are explained below:The @dist constructor, for a distribution that can be expressed as a simple deterministic transformation (technically, a pushforward) of an existing distribution.\nThe HeterogeneousMixture and HomogeneousMixture constructors for distributions that are mixtures of other distributions.\nAn API for defining arbitrary custom distributions in plain Julia code."
},

{
    "location": "ref/distributions/#Gen.bernoulli",
    "page": "Probability Distributions",
    "title": "Gen.bernoulli",
    "category": "constant",
    "text": "bernoulli(prob_true::Real)\n\nSamples a Bool value which is true with given probability\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.beta",
    "page": "Probability Distributions",
    "title": "Gen.beta",
    "category": "constant",
    "text": "beta(alpha::Real, beta::Real)\n\nSample a Float64 from a beta distribution.\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.beta_uniform",
    "page": "Probability Distributions",
    "title": "Gen.beta_uniform",
    "category": "constant",
    "text": "beta_uniform(theta::Real, alpha::Real, beta::Real)\n\nSamples a Float64 value from a mixture of a uniform distribution on [0, 1] with probability 1-theta and a beta distribution with parameters alpha and beta with probability theta.\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.binom",
    "page": "Probability Distributions",
    "title": "Gen.binom",
    "category": "constant",
    "text": "binom(n::Integer, p::Real)\n\nSample an Int from the Binomial distribution with parameters n (number of trials) and p (probability of success in each trial).\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.categorical",
    "page": "Probability Distributions",
    "title": "Gen.categorical",
    "category": "constant",
    "text": "categorical(probs::AbstractArray{U, 1}) where {U <: Real}\n\nGiven a vector of probabilities probs where sum(probs) = 1, sample an Int i from the set {1, 2, .., length(probs)} with probability probs[i].\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.cauchy",
    "page": "Probability Distributions",
    "title": "Gen.cauchy",
    "category": "constant",
    "text": "cauchy(x0::Real, gamma::Real)\n\nSample a Float64 value from a Cauchy distribution with location x0 and scale gamma.\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.exponential",
    "page": "Probability Distributions",
    "title": "Gen.exponential",
    "category": "constant",
    "text": "exponential(rate::Real)\n\nSample a Float64 from the exponential distribution with rate parameter rate.\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.gamma",
    "page": "Probability Distributions",
    "title": "Gen.gamma",
    "category": "constant",
    "text": "gamma(shape::Real, scale::Real)\n\nSample a Float64 from a gamma distribution.\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.geometric",
    "page": "Probability Distributions",
    "title": "Gen.geometric",
    "category": "constant",
    "text": "geometric(p::Real)\n\nSample an Int from the Geometric distribution with parameter p.\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.inv_gamma",
    "page": "Probability Distributions",
    "title": "Gen.inv_gamma",
    "category": "constant",
    "text": "inv_gamma(shape::Real, scale::Real)\n\nSample a Float64 from a inverse gamma distribution.\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.laplace",
    "page": "Probability Distributions",
    "title": "Gen.laplace",
    "category": "constant",
    "text": "laplce(loc::Real, scale::Real)\n\nSample a Float64 from a laplace distribution.\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.mvnormal",
    "page": "Probability Distributions",
    "title": "Gen.mvnormal",
    "category": "constant",
    "text": "mvnormal(mu::AbstractVector{T}, cov::AbstractMatrix{U}} where {T<:Real,U<:Real}\n\nSamples a Vector{Float64} value from a multivariate normal distribution.\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.neg_binom",
    "page": "Probability Distributions",
    "title": "Gen.neg_binom",
    "category": "constant",
    "text": "neg_binom(r::Real, p::Real)\n\nSample an Int from a Negative Binomial distribution. Returns the number of failures before the rth success in a sequence of independent Bernoulli trials. r is the number of successes (which may be fractional) and p is the probability of success per trial.\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.normal",
    "page": "Probability Distributions",
    "title": "Gen.normal",
    "category": "constant",
    "text": "normal(mu::Real, std::Real)\n\nSamples a Float64 value from a normal distribution.\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.piecewise_uniform",
    "page": "Probability Distributions",
    "title": "Gen.piecewise_uniform",
    "category": "constant",
    "text": "piecewise_uniform(bounds, probs)\n\nSamples a Float64 value from a piecewise uniform continuous distribution.\n\nThere are n bins where n = length(probs) and n + 1 = length(bounds). Bounds must satisfy bounds[i] < bounds[i+1] for all i. The probability density at x is zero if x <= bounds[1] or x >= bounds[end] and is otherwise probs[bin] / (bounds[bin] - bounds[bin+1]) where bounds[bin] < x <= bounds[bin+1].\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.poisson",
    "page": "Probability Distributions",
    "title": "Gen.poisson",
    "category": "constant",
    "text": "poisson(lambda::Real)\n\nSample an Int from the Poisson distribution with rate lambda.\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.uniform",
    "page": "Probability Distributions",
    "title": "Gen.uniform",
    "category": "constant",
    "text": "uniform(low::Real, high::Real)\n\nSample a Float64 from the uniform distribution on the interval [low, high].\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.uniform_discrete",
    "page": "Probability Distributions",
    "title": "Gen.uniform_discrete",
    "category": "constant",
    "text": "uniform_discrete(low::Integer, high::Integer)\n\nSample an Int from the uniform distribution on the set {low, low + 1, ..., high-1, high}.\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Built-In-Distributions-1",
    "page": "Probability Distributions",
    "title": "Built-In Distributions",
    "category": "section",
    "text": "bernoulli\nbeta\nbeta_uniform\nbinom\ncategorical\ncauchy\nexponential\ngamma\ngeometric\ninv_gamma\nlaplace\nmvnormal\nneg_binom\nnormal\npiecewise_uniform\npoisson\nuniform\nuniform_discrete"
},

{
    "location": "ref/distributions/#dist_dsl-1",
    "page": "Probability Distributions",
    "title": "Defining New Distributions Inline with the @dist DSL",
    "category": "section",
    "text": "The @dist DSL allows the user to concisely define a distribution, as long as that distribution can be expressed as a certain type of deterministic transformation of an existing distribution.  The syntax of the @dist DSL, as well as the class of permitted deterministic transformations, are explained below.@dist name(arg1, arg2, ..., argN) = bodyor@dist function name(arg1, arg2, ..., argN)\n    body\nendHere body is ordinary Julia code, with the constraint that body must contain exactly one random choice.  The value of the @dist expression is then a Gen.Distribution object called name, parameterized by arg1, ..., argN, representing the distribution over return values of body.This DSL is designed to address the issue that sometimes, values stored in the trace do not correspond to the most natural physical elements of the model state space, making inference programming and querying more taxing than necessary. For example, suppose we have a model of classes at a school, where the number of students is random, with mean 10, but always at least 3. Rather than writing the model as@gen function class_model()\n   n_students = @trace(poisson(7), :n_students_minus_3) + 3\n   ...\nendand thinking about the random variable :n_students_minus_3, you can use the @dist DSL to instead write@dist student_distr(mean, min) = poisson(mean-min) + min\n\n@gen function class_model()\n   n_students = @trace(student_distr(10, 3), :n_students)\n   ...\nendand think about the more natural random variable :n_students.  This leads to more natural inference programs, which can constrain and propose directly to the :n_students trace address."
},

{
    "location": "ref/distributions/#Permitted-constructs-for-the-body-of-a-@dist-1",
    "page": "Probability Distributions",
    "title": "Permitted constructs for the body of a @dist",
    "category": "section",
    "text": "It is not possible for @dist to work on any arbitrary body.  We now describe which constructs are permitted inside the body of a @dist expression.We can think of the body of an @dist function as containing ordinary Julia code, except that in addition to being described by their ordinary Julia types, each expression also belongs to one of three \"type spaces.\" These are:CONST: Constants, whose value is known at the time this @dist expression is evaluated.\nARG: Arguments and (deterministic, differentiable) functions of arguments. All expressions representing non-random values that depend on distribution arguments are ARG expressions.\nRND: Random variables. All expressions whose runtime values may differ across multiple calls to this distribution (with the same arguments) are RND expressions.Importantly, Julia control flow constructs generally expect CONST values: the condition of an if or the range of a for loop cannot be ARG or RND.The body expression as a whole must be a RND expression, representing a random variable. The behavior of the @dist definition is then to define a new distribution (with name name) that samples and evaluates the logpdf of the random variable represented by the body expression.Expressions are typed compositionally, with the following typing rules:Literals and free variables are CONSTs. Literals and symbols that appear free in the @dist body are of type CONST.\nArguments are ARGs. Symbols bound as arguments in the @dist declaration have type ARG in its body.\nDrawing from a distribution gives RND. If d is a distribution, and x_i are of type ARG or CONST, d(x_1, x_2, ...) is of type RND.\nFunctions of CONSTs are CONSTs. If f is a deterministic function and x_i are all of type CONST, f(x_1, x_2, ...) is of type CONST.\nFunctions of CONSTs and ARGs are ARGs. If f is a differentiable function, and each x_i is either a CONST or a scalar ARG (with at least one x_i being an ARG), then f(x_1, x_2, ...) is of type ARG.\nFunctions of CONSTs, ARGs, and RNDs are RNDs. If f is one of a special set of deterministic functions we\'ve defined (+, -, *, /, exp, log, getindex), and exactly one of its arguments x_i is of type RND, then f(x_1, x_2, ...) is of type RND.One way to think about this, without all the rules, is that CONST values are \"contaminated\" by interaction with ARG values (becoming ARGs themselves), and both CONST and ARG are \"contaminated\" by interaction with RND. Thinking of the body as an AST, the journey from leaf node to root node always involves transitions in the direction of CONST -> ARG -> RND, never in reverse."
},

{
    "location": "ref/distributions/#Restrictions-1",
    "page": "Probability Distributions",
    "title": "Restrictions",
    "category": "section",
    "text": "Users may not reassign to arguments (like x in the above example), and may not apply functions with side effects. Names bound to expressions of type RND must be used only once. e.g., let x = normal(0, 1) in x + x is not allowed."
},

{
    "location": "ref/distributions/#Examples-1",
    "page": "Probability Distributions",
    "title": "Examples",
    "category": "section",
    "text": "Let\'s walk through some examples.@dist f(x) = exp(normal(x, 1))We can annotate with types:1 :: CONST		  (by rule 1)\nx :: ARG 		  (by rule 2)\nnormal(x, 1) :: RND 	  (by rule 3)\nexp(normal(x, 1)) :: RND  (by rule 6)Here\'s another:@dist function labeled_cat(labels, probs)\n	index = categorical(probs)\n	labels[index]\nendAnd the types:probs :: ARG 			(by rule 2)\ncategorical(probs) :: RND 	(by rule 3)\nindex :: RND 			(Julia assignment)\nlabels :: ARG 			(by rule 2)\nlabels[index] :: RND 		(by rule 6, f == getindex)Note that getindex is designed to work on anything indexible, not just vectors. So, for example, it also works with Dicts.Another one (not as realistic, but it uses all the rules):@dist function weird(x)\n  log(normal(exp(x), exp(x))) + (x * (2 + 3))\nendAnd the types:2, 3 :: CONST 						(by rule 1)\n2 + 3 :: CONST 						(by rule 4)\nx :: ARG 						(by rule 2)\nx * (2 + 3) :: ARG 					(by rule 5)\nexp(x) :: ARG 						(by rule 5)\nnormal(exp(x), exp(x)) :: RND 				(by rule 3)\nlog(normal(exp(x), exp(x))) :: RND 			(by rule 6)\nlog(normal(exp(x), exp(x))) + (x * (2 + 3)) :: RND 	(by rule 6)"
},

{
    "location": "ref/distributions/#Gen.HomogeneousMixture",
    "page": "Probability Distributions",
    "title": "Gen.HomogeneousMixture",
    "category": "type",
    "text": "HomogeneousMixture(distribution::Distribution, dims::Vector{Int})\n\nDefine a new distribution that is a mixture of some number of instances of single base distributions.\n\nThe first argument defines the base distribution of each component in the mixture.\n\nThe second argument must have length equal to the number of arguments taken by the base distribution.  A value of 0 at a position in the vector an indicates that the corresponding argument to the base distribution is a scalar, and integer values of i for i >= 1 indicate that the corresponding argument is an i-dimensional array.\n\nExample:\n\nmixture_of_normals = HomogeneousMixture(normal, [0, 0])\n\nThe resulting distribution (e.g. mixture_of_normals above) can then be used like the built-in distribution values like normal. The distribution takes n+1 arguments where n is the number of arguments taken by the base distribution. The first argument to the distribution is a vector of non-negative mixture weights, which must sum to 1.0. The remaining arguments to the distribution correspond to the arguments of the base distribution, but have a different type: If an argument to the base distribution is a scalar of type T, then the corresponding argument to the mixture distribution is a Vector{T}, where each element of this vector is the argument to the corresponding mixture component. If an argument to the base distribution is an Array{T,N} for some N, then the corresponding argument to the mixture distribution is of the form arr::Array{T,N+1}, where each slice of the array of the form arr[:,:,...,i] is the argument for the ith mixture component.\n\nExample:\n\nmixture_of_normals = HomogeneousMixture(normal, [0, 0])\nmixture_of_mvnormals = HomogeneousMixture(mvnormal, [1, 2])\n\n@gen function foo()\n    # mixture of two normal distributions\n    # with means -1.0 and 1.0\n    # and standard deviations 0.1 and 10.0\n    # the first normal distribution has weight 0.4; the second has weight 0.6\n    x ~ mixture_of_normals([0.4, 0.6], [-1.0, 1.0], [0.1, 10.0])\n\n    # mixture of two multivariate normal distributions\n    # with means: [0.0, 0.0] and [1.0, 1.0]\n    # and covariance matrices: [1.0 0.0; 0.0 1.0] and [10.0 0.0; 0.0 10.0]\n    # the first multivariate normal distribution has weight 0.4;\n    # the second has weight 0.6\n    means = [0.0 1.0; 0.0 1.0] # or, cat([0.0, 0.0], [1.0, 1.0], dims=2)\n    covs = cat([1.0 0.0; 0.0 1.0], [10.0 0.0; 0.0 10.0], dims=3)\n    y ~ mixture_of_mvnormals([0.4, 0.6], means, covs)\nend\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.HeterogeneousMixture",
    "page": "Probability Distributions",
    "title": "Gen.HeterogeneousMixture",
    "category": "type",
    "text": "HeterogeneousMixture(distributions::Vector{Distribution{T}}) where {T}\n\nDefine a new distribution that is a mixture of a given list of base distributions.\n\nThe argument is the vector of base distributions, one for each mixture component.\n\nNote that the base distributions must have the same output type.\n\nExample:\n\nuniform_beta_mixture = HeterogeneousMixture([uniform, beta])\n\nThe resulting mixture distribution takes n+1 arguments, where n is the sum of the number of arguments taken by each distribution in the list. The first argument to the mixture distribution is a vector of non-negative mixture weights, which must sum to 1.0. The remaining arguments are the arguments to each mixture component distribution, in order in which the distributions are passed into the constructor.\n\nExample:\n\n@gen function foo()\n    # mixure of a uniform distribution on the interval [`lower`, `upper`]\n    # and a beta distribution with alpha parameter `a` and beta parameter `b`\n    # the uniform as weight 0.4 and the beta has weight 0.6\n    x ~ uniform_beta_mixture([0.4, 0.6], lower, upper, a, b)\nend\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Mixture-Distribution-Constructors-1",
    "page": "Probability Distributions",
    "title": "Mixture Distribution Constructors",
    "category": "section",
    "text": "There are two built-in constructors for defining mixture distributions:HomogeneousMixture\nHeterogeneousMixture"
},

{
    "location": "ref/distributions/#Defining-New-Distributions-From-Scratch-1",
    "page": "Probability Distributions",
    "title": "Defining New Distributions From Scratch",
    "category": "section",
    "text": "For distributions that cannot be expressed in the @dist DSL, users can define a custom distribution by defining an (ordinary Julia) subtype of Gen.Distribution and implementing the methods of the Distribution API.  This method requires more custom code than using the @dist DSL, but also affords more flexibility: arbitrary user-defined logic for sampling, PDF evaluation, etc."
},

{
    "location": "ref/modeling/#",
    "page": "Built-in Modeling Language",
    "title": "Built-in Modeling Language",
    "category": "page",
    "text": ""
},

{
    "location": "ref/modeling/#Gen.DynamicDSLFunction",
    "page": "Built-in Modeling Language",
    "title": "Gen.DynamicDSLFunction",
    "category": "type",
    "text": "DynamicDSLFunction{T} <: GenerativeFunction{T,DynamicDSLTrace}\n\nA generative function based on a shallowly embedding modeling language based on Julia functions.\n\nConstructed using the @gen keyword. Most methods in the generative function interface involve a end-to-end execution of the function.\n\n\n\n\n\n"
},

{
    "location": "ref/modeling/#Built-in-Modeling-Language-1",
    "page": "Built-in Modeling Language",
    "title": "Built-in Modeling Language",
    "category": "section",
    "text": "Gen provides a built-in embedded modeling language for defining generative functions. The language uses a syntax that extends Julia\'s syntax for defining regular Julia functions, and is also referred to as the Dynamic Modeling Language.Generative functions in the modeling language are identified using the @gen keyword in front of a Julia function definition. Here is an example @gen function that samples two random choices:@gen function foo(prob::Float64=0.1)\n    z1 = @trace(bernoulli(prob), :a)\n    z2 = @trace(bernoulli(prob), :b)\n    return z1 || z2\nendAfter running this code, foo is a Julia value of type DynamicDSLFunction:DynamicDSLFunctionNote that it is possible to provide default values for trailing positional arguments. However, keyword arguments are currently not supported.We can call the resulting generative function like we would a regular Julia function:retval::Bool = foo(0.5)We can also trace its execution:(trace, _) = generate(foo, (0.5,))Optional arguments can be left out of the above operations, and default values will be filled in automatically:julia> (trace, _) = generate(foo, (,));\njulia> get_args(trace)\n(0.1,)See Generative Functions for the full set of operations supported by a generative function. Note that the built-in modeling language described in this section is only one of many ways of defining a generative function – generative functions can also be constructed using other embedded languages, or by directly implementing the methods of the generative function interface. However, the built-in modeling language is intended to being flexible enough cover a wide range of use cases. In the remainder of this section, we refer to generative functions defined using the built-in modeling language as @gen functions. Details about the implementation of @gen functions can be found in the Modeling Language Implementation section."
},

{
    "location": "ref/modeling/#Annotations-1",
    "page": "Built-in Modeling Language",
    "title": "Annotations",
    "category": "section",
    "text": "Annotations are a syntactic construct in the built-in modeling language that allows users to provide additional information about how @gen functions should be interpreted. Annotations are optional, and not necessary to understand the basics of Gen. There are two types of annotations – argument annotations and function annotations.Argument annotations. In addition to type declarations on arguments like regular Julia functions, @gen functions also support additional annotations on arguments. Each argument can have the following different syntactic forms:y: No type declaration; no annotations.\ny::Float64: Type declaration; but no annotations.\n(grad)(y): No type declaration provided;, annotated with grad.\n(grad)(y::Float64): Type declaration provided; and annotated with grad.Currently, the possible argument annotations are:grad (see Differentiable programming).Function annotations. The @gen function itself can also be optionally associated with zero or more annotations, which are separate from the per-argument annotations. Function-level annotations use the following different syntactic forms:@gen function foo(<args>) <body> end: No function annotations.\n@gen (grad) function foo(<args>) <body> end: The function has the grad annotation.\n@gen (grad,static) function foo(<args>) <body> end: The function has both the grad and static annotations.Currently the possible function annotations are:grad (see Differentiable programming).\nstatic (see Static Modeling Language).\nnojuliacache (see Static Modeling Language)."
},

{
    "location": "ref/modeling/#Making-random-choices-1",
    "page": "Built-in Modeling Language",
    "title": "Making random choices",
    "category": "section",
    "text": "Random choices are made by calling a probability distribution on some arguments:val::Bool = bernoulli(0.5)See Probability Distributions for the set of built-in probability distributions, and for information on implementing new probability distributions.In the body of a @gen function, wrapping a call to a random choice with an @trace expression associates the random choice with an address, and evaluates to the value of the random choice. The syntax is:@trace(<distribution>(<args>), <addr>)Addresses can be any Julia value. Here, we give the Julia symbol address :z to a Bernoulli random choice.val::Bool = @trace(bernoulli(0.5), :z)Not all random choices need to be given addresses. An address is required if the random choice will be observed, or will be referenced by a custom inference algorithm (e.g. if it will be proposed to by a custom proposal distribution)."
},

{
    "location": "ref/modeling/#Sample-space-and-support-of-random-choices-1",
    "page": "Built-in Modeling Language",
    "title": "Sample space and support of random choices",
    "category": "section",
    "text": "Different probability distributions produce different types of values for their random choices. For example, the bernoulli distribution results in Bool values (either true or false), the normal distribution results in Real values that may be positive or negative, and the beta distribution result in Real values that are always in the unit interval (0, 1).Each Distribution is associated with two sets of values:The sample space of the distribution, which does not depend on the arguments.\nThe support of the distribution, which may depend on the arguments, and is the set of values that has nonzero probability (or probability density). It may be the entire sample space, or it may be a subset of the sample space.For example, the sample space of bernoulli is Bool and its support is either {true}, {false}, or {true, false}. The sample space of normal is Real and its support is the set of all values on the real line. The sample space of beta is Real and its support is the set of values in the interval (0, 1).Gen\'s built in modeling languages require that a address is associated with a fixed sample space. For example, it is not permitted to use a bernoulli distribution to sample at addresss :a in one execution, and a normal distribution to sample at address :a in a different execution, because their sample spaces differ (Bool vs Real):@gen function foo()\n    if @trace(bernoulli(0.5), :branch)\n        @trace(bernoulli(0.5), :x)\n    else\n        @trace(normal(0, 1), :x)\n    end\nendA generative function can be disciplined or not. In a disciplined generative function, the support of random choices at each address must be fixed. That is, for each address a there must exist a set S that is a subset of the sample space such that for all executions of the generative function, if a occurs as the address of a choice in the execution, then the support of that choice is exactly S. Violating this discipline will cause NaNs, errors, or undefined behavior in some inference programs. However, in many cases it is convenient to write an inference program that operates correctly and efficiently on some specialized class of undisciplined models. In these cases, authors who want their inference code to be reusable should consider documenting which kinds of undisciplined models their inference algorithms allow or expect to see.If the support of a random choice needs to change, a disciplined generative function can represent this by using a different address for each distinct value of the support. For example, consider the following generative function:@gen function foo()\n    n = @trace(categorical([0.5, 0.5]), :n) + 1\n    @trace(categorical(ones(n) / n), :x)\nendThe support of the random choice with address :x is either the set 1 2 or 1 2 3. Therefore, this random choice does not have constant support, and the generative function foo is not \'disciplined\'. Specifically, this could result in undefined behavior for the following inference program:tr, _ = importance_resampling(foo, (), choicemap((:x, 3)))It is recommended to write disciplined generative functions when possible."
},

{
    "location": "ref/modeling/#Calling-generative-functions-1",
    "page": "Built-in Modeling Language",
    "title": "Calling generative functions",
    "category": "section",
    "text": "@gen functions can invoke other generative functions in three ways:Untraced call: If foo is a generative function, we can invoke foo from within the body of a @gen function using regular call syntax. The random choices made within the call are not given addresses in our trace, and are therefore untraced random choices (see Generative Function Interface for details on untraced random choices).val = foo(0.5)Traced call with a nested address namespace: We can include the traced random choices made by foo in the caller\'s trace, under a namespace, using @trace:val = @trace(foo(0.5), :x)Now, all random choices made by foo are included in our trace, under the namespace :x. For example, if foo makes random choices at addresses :a and :b, these choices will have addresses :x => :a and :x => :b in the caller\'s trace.Traced call with shared address namespace: We can include the traced random choices made by foo in the caller\'s trace using @trace:val = @trace(foo(0.5))Now, all random choices made by foo are included in our trace. The caller must guarantee that there are no address collisions. NOTE: This type of call can only be used when calling other @gen functions. Other types of generative functions cannot be called in this way."
},

{
    "location": "ref/modeling/#Composite-addresses-1",
    "page": "Built-in Modeling Language",
    "title": "Composite addresses",
    "category": "section",
    "text": "In Julia, Pair values can be constructed using the => operator. For example, :a => :b is equivalent to Pair(:a, :b) and :a => :b => :c is equivalent to Pair(:a, Pair(:b, :c)). A Pair value (e.g. :a => :b => :c) can be passed as the address field in an @trace expression, provided that there is not also a random choice or generative function called with @trace at any prefix of the address.Consider the following examples.This example is invalid because :a => :b is a prefix of :a => :b => :c:@trace(normal(0, 1), :a => :b => :c)\n@trace(normal(0, 1), :a => :b)This example is invalid because :a is a prefix of :a => :b => :c:@trace(normal(0, 1), :a => :b => :c)\n@trace(normal(0, 1), :a)This example is invalid because :a => :b is a prefix of :a => :b => :c:@trace(normal(0, 1), :a => :b => :c)\n@trace(foo(0.5), :a => :b)This example is invalid because :a is a prefix of :a => :b:@trace(normal(0, 1), :a)\n@trace(foo(0.5), :a => :b)This example is valid because :a => :b and :a => :c are not prefixes of one another:@trace(normal(0, 1), :a => :b)\n@trace(normal(0, 1), :a => :c)This example is valid because :a => :b and :a => :c are not prefixes of one another:@trace(normal(0, 1), :a => :b)\n@trace(foo(0.5), :a => :c)"
},

{
    "location": "ref/modeling/#Tilde-syntax-1",
    "page": "Built-in Modeling Language",
    "title": "Tilde syntax",
    "category": "section",
    "text": "As a short-hand for @trace expressions, the tilde operator ~ can also be used to make random choices and traced calls to generative functions. For example, the expression{:x} ~ normal(0, 1)is equivalent to:@trace(normal(0, 1), :x)One can also conveniently assign random values to variables using the syntax:x ~ normal(0, 1)which is equivalent to:x = @trace(normal(0, 1), :x)Finally, one can make traced calls using a shared address namespace with the syntax:{*} ~ foo(0.5)which is equivalent to:@trace(foo(0.5))Note that ~ is also defined in Base as a unary operator that performs the bitwise-not operation (see Base.:~). This use of ~ is also supported within @gen functions. However, uses of ~ as a binary infix operator within an @gen function will always be treated as equivalent to an @trace expression. If your module contains its own two-argument definition YourModule.:~(a, b) of the ~ function, calls to that function within @gen functions have to be in qualified prefix form, i.e., you have to write YourModule.:~(a, b) instead of a ~ b."
},

{
    "location": "ref/modeling/#Return-value-1",
    "page": "Built-in Modeling Language",
    "title": "Return value",
    "category": "section",
    "text": "Like regular Julia functions, @gen functions return either the expression used in a return keyword, or by evaluating the last expression in the function body. Note that the return value of a @gen function is different from a trace of @gen function, which contains the return value associated with an execution as well as the assignment to each random choice made during the execution. See Generative Function Interface for more information about traces."
},

{
    "location": "ref/modeling/#Gen.init_param!",
    "page": "Built-in Modeling Language",
    "title": "Gen.init_param!",
    "category": "function",
    "text": "init_param!(gen_fn::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol, value)\n\nInitialize the the value of a named trainable parameter of a generative function.\n\nAlso generates the gradient accumulator for that parameter to zero(value).\n\nExample:\n\ninit_param!(foo, :theta, 0.6)\n\n\n\n\n\n"
},

{
    "location": "ref/modeling/#Gen.get_param",
    "page": "Built-in Modeling Language",
    "title": "Gen.get_param",
    "category": "function",
    "text": "value = get_param(gen_fn::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol)\n\nGet the current value of a trainable parameter of the generative function.\n\n\n\n\n\n"
},

{
    "location": "ref/modeling/#Gen.get_param_grad",
    "page": "Built-in Modeling Language",
    "title": "Gen.get_param_grad",
    "category": "function",
    "text": "value = get_param_grad(gen_fn::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol)\n\nGet the current value of the gradient accumulator for a trainable parameter of the generative function.\n\n\n\n\n\n"
},

{
    "location": "ref/modeling/#Gen.set_param!",
    "page": "Built-in Modeling Language",
    "title": "Gen.set_param!",
    "category": "function",
    "text": "set_param!(gen_fn::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol, value)\n\nSet the value of a trainable parameter of the generative function.\n\nNOTE: Does not update the gradient accumulator value.\n\n\n\n\n\n"
},

{
    "location": "ref/modeling/#Gen.zero_param_grad!",
    "page": "Built-in Modeling Language",
    "title": "Gen.zero_param_grad!",
    "category": "function",
    "text": "zero_param_grad!(gen_fn::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol)\n\nReset the gradient accumlator for a trainable parameter of the generative function to all zeros.\n\n\n\n\n\n"
},

{
    "location": "ref/modeling/#Trainable-parameters-1",
    "page": "Built-in Modeling Language",
    "title": "Trainable parameters",
    "category": "section",
    "text": "A @gen function may begin with an optional block of trainable parameter declarations. The block consists of a sequence of statements, beginning with @param, that declare the name and Julia type for each trainable parameter. The function below has a single trainable parameter theta with type Float64:@gen function foo(prob::Float64)\n    @param theta::Float64\n    z1 = @trace(bernoulli(prob), :a)\n    z2 = @trace(bernoulli(theta), :b)\n    return z1 || z2\nendTrainable parameters obey the same scoping rules as Julia local variables defined at the beginning of the function body. The value of a trainable parameter is undefined until it is initialized using init_param!. In addition to the current value, each trainable parameter has a current gradient accumulator value. The gradient accumulator value has the same shape (e.g. array dimension) as the parameter value. It is initialized to all zeros, and is incremented by accumulate_param_gradients!.The following methods are exported for the trainable parameters of @gen functions:init_param!\nget_param\nget_param_grad\nset_param!\nzero_param_grad!Trainable parameters are designed to be trained using gradient-based methods. This is discussed in the next section."
},

{
    "location": "ref/modeling/#Differentiable-programming-1",
    "page": "Built-in Modeling Language",
    "title": "Differentiable programming",
    "category": "section",
    "text": "Given a trace of a @gen function, Gen supports automatic differentiation of the log probability (density) of all of the random choices made in the trace with respect to the following types of inputs:all or a subset of the arguments to the function.\nthe values of all or a subset of random choices.\nall or a subset of trainable parameters of the @gen function.We first discuss the semantics of these gradient computations, and then discuss what how to write and use Julia code in the body of a @gen function so that it can be automatically differentiated by the gradient computation."
},

{
    "location": "ref/modeling/#Supported-gradient-computations-1",
    "page": "Built-in Modeling Language",
    "title": "Supported gradient computations",
    "category": "section",
    "text": "Gradients with respect to arguments. A @gen function may have a fixed set of its arguments annotated with grad, which indicates that gradients with respect to that argument should be supported. For example, in the function below, we indicate that we want to support differentiation with respect to the y argument, but that we do not want to support differentiation with respect to the x argument.@gen function foo(x, (grad)(y))\n    if x > 5\n        @trace(normal(y, 1), :z)\n    else\n        @trace(normal(y, 10), :z)\n    end\nendFor the function foo above, when x > 5, the gradient with respect to y is the gradient of the log probability density of a normal distribution with standard deviation 1, with respect to its mean, evaluated at mean y. When x <= 5, we instead differentiate the log density of a normal distribution with standard deviation 10, relative to its mean.Gradients with respect to values of random choices. The author of a @gen function also identifies a set of addresses of random choices with respect to which they wish to support gradients of the log probability (density). Gradients of the log probability (density) with respect to the values of random choices are used in gradient-based numerical optimization of random choices, as well as certain MCMC updates that require gradient information.Gradients with respect to trainable parameters. The gradient of the log probability (density) with respect to the trainable parameters can also be computed using automatic differentiation. Currently, the log probability (density) must be a differentiable function of all trainable parameters.Gradients of a function of the return value. Differentiable programming in Gen composes across function calls. If the return value of the @gen function is conditionally dependent on source elements including (i) any arguments annotated with grad or (ii) any random choices for which gradients are supported, or (ii) any trainable parameters, then the gradient computation requires a gradient of the an external function with respect to the return value in order to the compute the correct gradients. Thus, the function being differentiated always includes a term representing the log probability (density) of all random choices made by the function, but can be extended with a term that depends on the return value of the function. The author of a @gen function can indicate that the return value depends on the source elements (causing the gradient with respect to the return value is required for all gradient computations) by adding the grad annotation to the @gen function itself. For example, in the function below, the return value is conditionally dependent (and actually identical to) on the random value at address :z:@gen function foo(x, (grad)(y))\n    if x > 5\n        return @trace(normal(y, 1), :z)\n    else\n        return @trace(normal(y, 10), :z)\n    end\nendIf the author of foo wished to support the computation of gradients with respect to the value of :z, they would need to add the grad annotation to foo using the following syntax:@gen (grad) function foo(x, (grad)(y))\n    if x > 5\n        return @trace(normal(y, 1), :z)\n    else\n        return @trace(normal(y, 10), :z)\n    end\nend"
},

{
    "location": "ref/modeling/#Writing-differentiable-code-1",
    "page": "Built-in Modeling Language",
    "title": "Writing differentiable code",
    "category": "section",
    "text": "In order to compute the gradients described above, the code in the body of the @gen function needs to be differentiable. Code in the body of a @gen function consists of:Julia code\nMaking random choices\nCalling generative functionsWe now discuss how to ensure that code of each of these forms is differentiable. Note that the procedures for differentiation of code described below are only performed during certain operations on @gen functions (choice_gradients and accumulate_param_gradients!).Julia code. Julia code used within a body of a @gen function is made differentiable using the ReverseDiff package, which implements  reverse-mode automatic differentiation. Specifically, values whose gradient is required (either values of arguments, random choices, or trainable parameters) are \'tracked\' by boxing them into special values and storing the tracked value on a \'tape\'. For example a Float64 value is boxed into a ReverseDiff.TrackedReal value. Methods (including e.g. arithmetic operators) are defined that operate on these tracked values and produce other tracked values as a result. As the computation proceeds all the values are placed onto the tape, with back-references to the parent operation and operands. Arithmetic operators, array and linear algebra functions, and common special numerical functions, as well as broadcasting, are automatically supported. See ReverseDiff for more details.Making random choices. When making a random choice, each argument is either a tracked value or not. If the argument is a tracked value, then the probability distribution must support differentiation of the log probability (density) with respect to that argument. Otherwise, an error is thrown. The has_argument_grads function indicates which arguments support differentiation for a given distribution (see Probability Distributions). If the gradient is required for the value of a random choice, the distribution must support differentiation of the log probability (density) with respect to the value. This is indicated by the has_output_grad function.Calling generative functions. Like distributions, generative functions indicate which of their arguments support differentiation, using the has_argument_grads function. It is an error if a tracked value is passed as an argument of a generative function, when differentiation is not supported by the generative function for that argument. If a generative function gen_fn has accepts_output_grad(gen_fn) = true, then the return value of the generative function call will be tracked and will propagate further through the caller @gen function\'s computation."
},

{
    "location": "ref/modeling/#Static-Modeling-Language-1",
    "page": "Built-in Modeling Language",
    "title": "Static Modeling Language",
    "category": "section",
    "text": "The static modeling language is a restricted variant of the built-in modeling language. Models written in the static modeling language can result in better inference performance (more inference operations per second and less memory consumption), than the full built-in modeling language, especially for models used with iterative inference algorithms like Markov chain Monte Carlo.A function is identified as using the static modeling language by adding the static annotation to the function. For example:@gen (static) function foo(prob::Float64)\n    z1 = @trace(bernoulli(prob), :a)\n    z2 = @trace(bernoulli(prob), :b)\n    z3 = z1 || z2\n    z4 = !z3\n    return z4\nendAfter running this code, foo is a Julia value whose type is a subtype of StaticIRGenerativeFunction, which is a subtype of GenerativeFunction."
},

{
    "location": "ref/modeling/#Static-computation-graph-1",
    "page": "Built-in Modeling Language",
    "title": "Static computation graph",
    "category": "section",
    "text": "Using the static annotation instructs Gen to statically construct a directed acyclic graph for the computation represented by the body of the function. For the function foo above, the static graph looks like:<div style=\"text-align:center\">\n    <img src=\"../../images/static_graph.png\" alt=\"example static computation graph\" width=\"50%\"/>\n</div>In this graph, oval nodes represent random choices, square nodes represent Julia computations, and diamond nodes represent arguments. The light blue shaded node is the return value of the function. Having access to the static graph allows Gen to generate specialized code for Updating traces that skips unecessary parts of the computation. Specifically, when applying an update operation, the graph is analyzed, and each value in the graph identified as having possibly changed, or not. Nodes in the graph do not need to be re-executed if none of their input values could have possibly changed. Also, even if some inputs to a generative function node may have changed, knowledge that some of the inputs have not changed often allows the generative function being called to more efficiently perform its update operation. This is the case for functions produced by Generative Function Combinators.You can plot the graph for a function with the static annotation if you have PyCall installed, and a Python environment that contains the graphviz Python package, using, e.g.:using PyCall\n@pyimport graphviz\nusing Gen: draw_graph\ndraw_graph(foo, graphviz, \"test\")This will produce a file test.pdf in the current working directory containing the rendered graph."
},

{
    "location": "ref/modeling/#Restrictions-1",
    "page": "Built-in Modeling Language",
    "title": "Restrictions",
    "category": "section",
    "text": "First, the definition of a (static) generative function is always expected to occur as a top-level definition (aka global variable); usage in non–top-level scopes is unsupported and may result in incorrect behavior. Recall also that the macro @load_generated_functions is expected to be called as a top-level expression only.Next, in order to be able to construct the static graph, Gen restricts the permitted syntax that can be used in functions annotated with static. In particular, each statement in the body must be one of the following:A @param statement specifying any Trainable parameters, e.g.:@param theta::Float64An assignment, with a symbol or tuple of symbols on the left-hand side, and a Julia expression on the right-hand side, which may include @trace expressions, e.g.:mu, sigma = @trace(bernoulli(p), :x) ? (mu1, sigma1) : (mu2, sigma2)A top-level @trace expression, e.g.:@trace(bernoulli(1-prob_tails), :flip)All @trace expressions must use a literal Julia symbol for the first component in the address. Unlike the full built-in modeling-language, the address is not optional.A return statement, with a Julia expression on the right-hand side, e.g.:return @trace(geometric(prob), :n_flips) + 1The functions are also subject to the following restrictions:Default argument values are not supported.\nJulia closures are not allowed.\nList comprehensions with internal @trace calls are not allowed.\nSplatting within @trace calls is not supported\nGenerative functions that are passed in as arguments cannot be traced.\nFor composite addresses (e.g. :a => 2 => :c) the first component of the address must be a literal symbol, and there may only be one statement in the function body that uses this symbol for the first component of its address.\nJulia control flow constructs (e.g. if, for, while) cannot be used as top-level statements in the function body. Control flow should be implemented inside either Julia functions that are called, or other generative functions.Certain loop constructs can be implemented using Generative Function Combinators instead. For example, the following loop:for (i, prob) in enumerate(probs)\n    @trace(foo(prob), :foo => i)\nendcan instead be implemented as:@trace(Map(foo)(probs), :foo)"
},

{
    "location": "ref/modeling/#Gen.@load_generated_functions",
    "page": "Built-in Modeling Language",
    "title": "Gen.@load_generated_functions",
    "category": "macro",
    "text": "@load_generated_functions\n\nPermit use of generative functions written in the static modeling language up to this point. Functions are loaded into the calling module.\n\nThis macro is intended to be called as a top-level expression only; use in non–top-level scopes may result in incorrect behavior.\n\n\n\n\n\n"
},

{
    "location": "ref/modeling/#Loading-generated-functions-1",
    "page": "Built-in Modeling Language",
    "title": "Loading generated functions",
    "category": "section",
    "text": "Before a function with a static annotation can be used, the @load_generated_functions macro must be called:@load_generated_functionsTypically, one call to this function, at the top level of a script, separates the definition of generative functions from the execution of inference code, e.g.:using Gen: @load_generated_functions\n\n# define generative functions and inference code\n..\n\n# allow static generative functions defined above to be used\n@load_generated_functions()\n\n# run inference code\n..When static generative functions are defined in a Julia module, @load_generated_functions should be called after all static functions are defined:module MyModule\nusing Gen\n# Include code that defines static generative functions\ninclude(\"my_static_gen_functions.jl\")\n# Load generated functions defined in this module\n@load_generated_functions()\nendAny script that imports or uses MyModule will then no longer need to call @load_generated_functions in order to use the static generative functions defined in that module:using Gen\nusing MyModule: my_static_gen_fn\ntrace = simulate(my_static_gen_fn, ())"
},

{
    "location": "ref/modeling/#Performance-tips-1",
    "page": "Built-in Modeling Language",
    "title": "Performance tips",
    "category": "section",
    "text": "For better performance when the arguments are simple data types like Float64, annotate the arguments with the concrete type. This permits a more optimized trace data structure to be generated for the generative function."
},

{
    "location": "ref/modeling/#Caching-Julia-values-1",
    "page": "Built-in Modeling Language",
    "title": "Caching Julia values",
    "category": "section",
    "text": "By default, the values of Julia computations (all calls that are not random choices or calls to generative functions) are cached as part of the trace, so that Updating traces can avoid unecessary re-execution of Julia code. However, this cache may grow the memory footprint of a trace. To disable caching of Julia values, use the function annotation nojuliacache (this annotation is ignored unless the static function annotation is also used)."
},

{
    "location": "ref/combinators/#",
    "page": "Generative Function Combinators",
    "title": "Generative Function Combinators",
    "category": "page",
    "text": ""
},

{
    "location": "ref/combinators/#Generative-Function-Combinators-1",
    "page": "Generative Function Combinators",
    "title": "Generative Function Combinators",
    "category": "section",
    "text": "Generative function combinators are Julia functions that take one or more generative functions as input and return a new generative function. Generative function combinators are used to express patterns of repeated computation that appear frequently in generative models. Some generative function combinators are similar to higher order functions from functional programming languages. However, generative function combinators are not \'higher order generative functions\', because they are not themselves generative functions (they are regular Julia functions)."
},

{
    "location": "ref/combinators/#Gen.Map",
    "page": "Generative Function Combinators",
    "title": "Gen.Map",
    "category": "type",
    "text": "gen_fn = Map(kernel::GenerativeFunction)\n\nReturn a new generative function that applies the kernel independently for a vector of inputs.\n\nThe returned generative function has one argument with type Vector{X} for each argument of the input generative function with type X. The length of each argument, which must be the same for each argument, determines the number of times the input generative function is called (N). Each call to the input function is made under address namespace i for i=1..N. The return value of the returned function has type FunctionalCollections.PersistentVector{Y} where Y is the type of the return value of the input function. The map combinator is similar to the \'map\' higher order function in functional programming, except that the map combinator returns a new generative function that must then be separately applied.\n\nIf kernel has optional trailing arguments, the corresponding Vector arguments can be omitted from calls to Map(kernel).\n\n\n\n\n\n"
},

{
    "location": "ref/combinators/#Map-combinator-1",
    "page": "Generative Function Combinators",
    "title": "Map combinator",
    "category": "section",
    "text": "MapIn the schematic below, the kernel is denoted mathcalG_mathrmk.<div style=\"text-align:center\">\n    <img src=\"../../images/map_combinator.png\" alt=\"schematic of map combinator\" width=\"50%\"/>\n</div>For example, consider the following generative function, which makes one random choice at address :z:@gen function foo(x1::Float64, x2::Float64)\n    y = @trace(normal(x1 + x2, 1.0), :z)\n    return y\nendWe apply the map combinator to produce a new generative function bar:bar = Map(foo)We can then obtain a trace of bar:(trace, _) = generate(bar, ([0.0, 0.5], [0.5, 1.0]))This causes foo to be invoked twice, once with arguments (0.0, 0.5) in address namespace 1 and once with arguments (0.5, 1.0) in address namespace 2. If the resulting trace has random choices:│\n├── 1\n│   │\n│   └── :z : -0.5757913836706721\n│\n└── 2\n    │\n    └── :z : 0.7357177113395333then the return value is:FunctionalCollections.PersistentVector{Any}[-0.575791, 0.735718]"
},

{
    "location": "ref/combinators/#Gen.Unfold",
    "page": "Generative Function Combinators",
    "title": "Gen.Unfold",
    "category": "type",
    "text": "gen_fn = Unfold(kernel::GenerativeFunction)\n\nReturn a new generative function that applies the kernel in sequence, passing the return value of one application as an input to the next.\n\nThe kernel accepts the following arguments:\n\nThe first argument is the Int index indicating the position in the sequence (starting from 1).\nThe second argument is the state.\nThe kernel may have additional arguments after the state.\n\nThe return type of the kernel must be the same type as the state.\n\nThe returned generative function accepts the following arguments:\n\nThe number of times (N) to apply the kernel.\nThe initial state.\nThe rest of the arguments (not including the state) that will be passed to each kernel application.\n\nThe return type of the returned generative function is FunctionalCollections.PersistentVector{T} where T is the return type of the kernel.\n\nIf kernel has optional trailing arguments, the corresponding arguments can be omitted from calls to Unfold(kernel).\n\n\n\n\n\n"
},

{
    "location": "ref/combinators/#Unfold-combinator-1",
    "page": "Generative Function Combinators",
    "title": "Unfold combinator",
    "category": "section",
    "text": "UnfoldIn the schematic below, the kernel is denoted mathcalG_mathrmk. The initial state is denoted y_0, the number of applications is n, and the remaining arguments to the kernel not including the state, are z.<div style=\"text-align:center\">\n    <img src=\"../../images/unfold_combinator.png\" alt=\"schematic of unfold combinator\" width=\"70%\"/>\n</div>For example, consider the following kernel, with state type Bool, which makes one random choice at address :z:@gen function foo(t::Int, y_prev::Bool, z1::Float64, z2::Float64)\n    y = @trace(bernoulli(y_prev ? z1 : z2), :y)\n    return y\nendWe apply the map combinator to produce a new generative function bar:bar = Unfold(foo)We can then obtain a trace of bar:(trace, _) = generate(bar, (5, false, 0.05, 0.95))This causes foo to be invoked five times. The resulting trace may contain the following random choices:│\n├── 1\n│   │\n│   └── :y : true\n│\n├── 2\n│   │\n│   └── :y : false\n│\n├── 3\n│   │\n│   └── :y : true\n│\n├── 4\n│   │\n│   └── :y : false\n│\n└── 5\n    │\n    └── :y : true\nthen the return value is:FunctionalCollections.PersistentVector{Any}[true, false, true, false, true]"
},

{
    "location": "ref/combinators/#Recurse-combinator-1",
    "page": "Generative Function Combinators",
    "title": "Recurse combinator",
    "category": "section",
    "text": "TODO: document me<div style=\"text-align:center\">\n    <img src=\"../../images/recurse_combinator.png\" alt=\"schematic of recurse combinatokr\" width=\"70%\"/>\n</div>"
},

{
    "location": "ref/combinators/#Gen.Switch",
    "page": "Generative Function Combinators",
    "title": "Gen.Switch",
    "category": "type",
    "text": "gen_fn = Switch(gen_fns::GenerativeFunction...)\n\nReturns a new generative function that accepts an argument tuple of type Tuple{Int, ...} where the first index indicates which branch to call.\n\ngen_fn = Switch(d::Dict{T, Int}, gen_fns::GenerativeFunction...) where T\n\nReturns a new generative function that accepts an argument tuple of type Tuple{Int, ...} or an argument tuple of type Tuple{T, ...} where the first index either indicates which branch to call, or indicates an index into d which maps to the selected branch. This form is meant for convenience - it allows the programmer to use d like if-else or case statements.\n\nSwitch is designed to allow for the expression of patterns of if-else control flow. gen_fns must satisfy a few requirements:\n\nEach gen_fn in gen_fns must accept the same argument types.\nEach gen_fn in gen_fns must return the same return type.\n\nOtherwise, each gen_fn can come from different modeling languages, possess different traces, etc.\n\n\n\n\n\n"
},

{
    "location": "ref/combinators/#Switch-combinator-1",
    "page": "Generative Function Combinators",
    "title": "Switch combinator",
    "category": "section",
    "text": "Switch<div style=\"text-align:center\">\n    <img src=\"../../images/switch_combinator.png\" alt=\"schematic of switch combinator\" width=\"100%\"/>\n</div>Consider the following constructions:@gen function bang((grad)(x::Float64), (grad)(y::Float64))\n    std::Float64 = 3.0\n    z = @trace(normal(x + y, std), :z)\n    return z\nend\n\n@gen function fuzz((grad)(x::Float64), (grad)(y::Float64))\n    std::Float64 = 3.0\n    z = @trace(normal(x + 2 * y, std), :z)\n    return z\nend\n\nsc = Switch(bang, fuzz)This creates a new generative function sc. We can then obtain the trace of sc:(trace, _) = simulate(sc, (2, 5.0, 3.0))The resulting trace contains the subtrace from the branch with index 2 - in this case, a call to fuzz:│\n└── :z : 13.552870875213735"
},

{
    "location": "ref/choice_maps/#",
    "page": "Choice Maps",
    "title": "Choice Maps",
    "category": "page",
    "text": ""
},

{
    "location": "ref/choice_maps/#Gen.ChoiceMap",
    "page": "Choice Maps",
    "title": "Gen.ChoiceMap",
    "category": "type",
    "text": "abstract type ChoiceMap end\n\nAbstract type for maps from hierarchical addresses to values.\n\n\n\n\n\n"
},

{
    "location": "ref/choice_maps/#Gen.has_value",
    "page": "Choice Maps",
    "title": "Gen.has_value",
    "category": "function",
    "text": "has_value(choices::ChoiceMap, addr)\n\nReturn true if there is a value at the given address.\n\n\n\n\n\n"
},

{
    "location": "ref/choice_maps/#Gen.get_value",
    "page": "Choice Maps",
    "title": "Gen.get_value",
    "category": "function",
    "text": "value = get_value(choices::ChoiceMap, addr)\n\nReturn the value at the given address in the assignment, or throw a KeyError if no value exists. A syntactic sugar is Base.getindex:\n\nvalue = choices[addr]\n\n\n\n\n\n"
},

{
    "location": "ref/choice_maps/#Gen.get_submap",
    "page": "Choice Maps",
    "title": "Gen.get_submap",
    "category": "function",
    "text": "submap = get_submap(choices::ChoiceMap, addr)\n\nReturn the sub-assignment containing all choices whose address is prefixed by addr.\n\nIt is an error if the assignment contains a value at the given address. If there are no choices whose address is prefixed by addr then return an EmptyChoiceMap.\n\n\n\n\n\n"
},

{
    "location": "ref/choice_maps/#Gen.get_values_shallow",
    "page": "Choice Maps",
    "title": "Gen.get_values_shallow",
    "category": "function",
    "text": "key_submap_iterable = get_values_shallow(choices::ChoiceMap)\n\nReturn an iterator over tuples of the form (key, value) for each top-level key associated with a value.\n\n\n\n\n\n"
},

{
    "location": "ref/choice_maps/#Gen.get_submaps_shallow",
    "page": "Choice Maps",
    "title": "Gen.get_submaps_shallow",
    "category": "function",
    "text": "key_submap_iterable = get_submaps_shallow(choices::ChoiceMap)\n\nReturn an iterator over tuples of the form (key, submap::ChoiceMap) for each top-level key that has a non-empty sub-assignment.\n\n\n\n\n\n"
},

{
    "location": "ref/choice_maps/#Gen.to_array",
    "page": "Choice Maps",
    "title": "Gen.to_array",
    "category": "function",
    "text": "arr::Vector{T} = to_array(choices::ChoiceMap, ::Type{T}) where {T}\n\nPopulate an array with values of choices in the given assignment.\n\nIt is an error if each of the values cannot be coerced into a value of the given type.\n\nImplementation\n\nTo support to_array, a concrete subtype T <: ChoiceMap should implement the following method:\n\nn::Int = _fill_array!(choices::T, arr::Vector{V}, start_idx::Int) where {V}\n\nPopulate arr with values from the given assignment, starting at start_idx, and return the number of elements in arr that were populated.\n\n\n\n\n\n"
},

{
    "location": "ref/choice_maps/#Gen.from_array",
    "page": "Choice Maps",
    "title": "Gen.from_array",
    "category": "function",
    "text": "choices::ChoiceMap = from_array(proto_choices::ChoiceMap, arr::Vector)\n\nReturn an assignment with the same address structure as a prototype assignment, but with values read off from the given array.\n\nThe order in which addresses are populated is determined by the prototype assignment. It is an error if the number of choices in the prototype assignment is not equal to the length the array.\n\nImplementation\n\nTo support from_array, a concrete subtype T <: ChoiceMap should implement the following method:\n\n(n::Int, choices::T) = _from_array(proto_choices::T, arr::Vector{V}, start_idx::Int) where {V}\n\nReturn an assignment with the same address structure as a prototype assignment, but with values read off from arr, starting at position start_idx, and the number of elements read from arr.\n\n\n\n\n\n"
},

{
    "location": "ref/choice_maps/#Gen.get_selected",
    "page": "Choice Maps",
    "title": "Gen.get_selected",
    "category": "function",
    "text": "selected_choices = get_selected(choices::ChoiceMap, selection::Selection)\n\nFilter the choice map to include only choices in the given selection.\n\nReturns a new choice map.\n\n\n\n\n\n"
},

{
    "location": "ref/choice_maps/#Choice-Maps-1",
    "page": "Choice Maps",
    "title": "Choice Maps",
    "category": "section",
    "text": "Maps from the addresses of random choices to their values are stored in associative tree-structured data structures that have the following abstract type:ChoiceMapChoice maps are constructed by users to express observations and/or constraints on the traces of generative functions. Choice maps are also returned by certain Gen inference methods, and are used internally by various Gen inference methods.Choice maps provide the following methods:has_value\nget_value\nget_submap\nget_values_shallow\nget_submaps_shallow\nto_array\nfrom_array\nget_selectedNote that none of these methods mutate the choice map.Choice maps also implement:Base.isempty, which tests of there are no random choices in the choice map\nBase.merge, which takes two choice maps, and returns a new choice map containing all random choices in either choice map. It is an error if the choice maps both have values at the same address, or if one choice map has a value at an address that is the prefix of the address of a value in the other choice map.\n==, which tests if two choice maps have the same addresses and values at those addresses."
},

{
    "location": "ref/choice_maps/#Gen.choicemap",
    "page": "Choice Maps",
    "title": "Gen.choicemap",
    "category": "function",
    "text": "choices = choicemap()\n\nConstruct an empty mutable choice map.\n\n\n\n\n\nchoices = choicemap(tuples...)\n\nConstruct a mutable choice map initialized with given address, value tuples.\n\n\n\n\n\n"
},

{
    "location": "ref/choice_maps/#Gen.set_value!",
    "page": "Choice Maps",
    "title": "Gen.set_value!",
    "category": "function",
    "text": "set_value!(choices::DynamicChoiceMap, addr, value)\n\nSet the given value for the given address.\n\nWill cause any previous value or sub-assignment at this address to be deleted. It is an error if there is already a value present at some prefix of the given address.\n\nThe following syntactic sugar is provided:\n\nchoices[addr] = value\n\n\n\n\n\n"
},

{
    "location": "ref/choice_maps/#Gen.set_submap!",
    "page": "Choice Maps",
    "title": "Gen.set_submap!",
    "category": "function",
    "text": "set_submap!(choices::DynamicChoiceMap, addr, submap::ChoiceMap)\n\nReplace the sub-assignment rooted at the given address with the given sub-assignment. Set the given value for the given address.\n\nWill cause any previous value or sub-assignment at the given address to be deleted. It is an error if there is already a value present at some prefix of address.\n\n\n\n\n\n"
},

{
    "location": "ref/choice_maps/#Mutable-Choice-Maps-1",
    "page": "Choice Maps",
    "title": "Mutable Choice Maps",
    "category": "section",
    "text": "A mutable choice map can be constructed with choicemap, and then populated:choices = choicemap()\nchoices[:x] = true\nchoices[\"foo\"] = 1.25\nchoices[:y => 1 => :z] = -6.3There is also a constructor that takes initial (address, value) pairs:choices = choicemap((:x, true), (\"foo\", 1.25), (:y => 1 => :z, -6.3))choicemap\nset_value!\nset_submap!"
},

{
    "location": "ref/selections/#",
    "page": "Selections",
    "title": "Selections",
    "category": "page",
    "text": ""
},

{
    "location": "ref/selections/#Gen.Selection",
    "page": "Selections",
    "title": "Gen.Selection",
    "category": "type",
    "text": "abstract type Selection end\n\nAbstract type for selections of addresses.\n\nAll selections implement the following methods:\n\nBase.in(addr, selection)\n\nIs the address selected?\n\nBase.getindex(selection, addr)\n\nGet the subselection at the given address.\n\nBase.isempty(selection)\n\nIs the selection guaranteed to be empty?\n\nget_address_schema(T)\n\nReturn a shallow, compile-time address schema, where T is the concrete type of the selection.\n\n\n\n\n\n"
},

{
    "location": "ref/selections/#Gen.select",
    "page": "Selections",
    "title": "Gen.select",
    "category": "function",
    "text": "selection = select(addrs...)\n\nReturn a selection containing a given set of addresses.\n\nExamples:\n\nselection = select(:x, \"foo\", :y => 1 => :z)\nselection = select()\nselection = select(:x => 1, :x => 2)\n\n\n\n\n\n"
},

{
    "location": "ref/selections/#Gen.selectall",
    "page": "Selections",
    "title": "Gen.selectall",
    "category": "function",
    "text": "selection = selectall()\n\nConstruct a selection that includes all random choices.\n\n\n\n\n\n"
},

{
    "location": "ref/selections/#Gen.complement",
    "page": "Selections",
    "title": "Gen.complement",
    "category": "function",
    "text": "comp_selection = complement(selection::Selection)\n\nReturn a selection that is the complement of the given selection.\n\nAn address is in the selection if it is not in the complement selection.\n\n\n\n\n\n"
},

{
    "location": "ref/selections/#Selections-1",
    "page": "Selections",
    "title": "Selections",
    "category": "section",
    "text": "A selection represents a set of addresses of random choices. Selections allow users to specify to which subset of the random choices in a trace a given inference operation should apply.An address that is added to a selection indicates that either the random choice at that address should be included in the selection, or that all random choices made by a generative function traced at that address should be included. For example, consider the following selection:selection = select(:x, :y)If we use this selection in the context of a trace of the function baz below, we are selecting two random choices, at addresses :x and :y:@gen function baz()\n    @trace(bernoulli(0.5), :x)\n    @trace(bernoulli(0.5), :y)\nendIf we use this selection in the context of a trace of the function bar below, we are actually selecting three random choices–-the one random choice made by bar at address :x and the two random choices made by foo at addresses :y => :z and :y => :w`:@gen function foo()\n    @trace(normal(0, 1), :z)\n    @trace(normal(0, 1), :w)\nend\nend\n\n@gen function bar()\n    @trace(bernoulli(0.5), :x)\n    @trace(foo(), :y)\nendThere is an abstract type for selections:SelectionThere are various concrete types for selections, each of which is a subtype of Selection. Users can construct selections with the following methods:select\nselectall\ncomplementThe select method returns a selection with concrete type DynamicSelection. The selectall method returns a selection with concrete type AllSelection. The full list of concrete types of selections is shown below. Most users need not worry about these types. Note that only selections of type DynamicSelection are mutable (using push! and set_subselection!).EmptySelection\nAllSelection\nHierarchicalSelection\nDynamicSelection\nStaticSelection\nComplementSelection"
},

{
    "location": "ref/parameter_optimization/#",
    "page": "Optimizing Trainable Parameters",
    "title": "Optimizing Trainable Parameters",
    "category": "page",
    "text": ""
},

{
    "location": "ref/parameter_optimization/#Optimizing-Trainable-Parameters-1",
    "page": "Optimizing Trainable Parameters",
    "title": "Optimizing Trainable Parameters",
    "category": "section",
    "text": "Trainable parameters of generative functions are initialized differently depending on the type of generative function. Trainable parameters of the built-in modeling language are initialized with init_param!.Gradient-based optimization of the trainable parameters of generative functions is based on interleaving two steps:Incrementing gradient accumulators for trainable parameters by calling accumulate_param_gradients! on one or more traces.\nUpdating the value of trainable parameters and resetting the gradient accumulators to zero, by calling apply! on a parameter update, as described below."
},

{
    "location": "ref/parameter_optimization/#Gen.ParamUpdate",
    "page": "Optimizing Trainable Parameters",
    "title": "Gen.ParamUpdate",
    "category": "type",
    "text": "update = ParamUpdate(conf, param_lists...)\n\nReturn an update configured by conf that applies to set of parameters defined by param_lists.\n\nEach element in param_lists value is is pair of a generative function and a vector of its parameter references.\n\nExample. To construct an update that applies a gradient descent update to the parameters :a and :b of generative function foo and the parameter :theta of generative function :bar:\n\nupdate = ParamUpdate(GradientDescent(0.001, 100), foo => [:a, :b], bar => [:theta])\n\n\n\nSyntactic sugar for the constructor form above.\n\nupdate = ParamUpdate(conf, gen_fn::GenerativeFunction)\n\nReturn an update configured by conf that applies to all trainable parameters owned by the given generative function.\n\nNote that trainable parameters not owned by the given generative function will not be updated, even if they are used during execution of the function.\n\nExample. If generative function foo has parameters :a and :b, to construct an update that applies a gradient descent update to the parameters :a and :b:\n\nupdate = ParamUpdate(GradientDescent(0.001, 100), foo)\n\n\n\n\n\n"
},

{
    "location": "ref/parameter_optimization/#Gen.apply!",
    "page": "Optimizing Trainable Parameters",
    "title": "Gen.apply!",
    "category": "function",
    "text": "apply!(update::ParamUpdate)\n\nPerform one step of the update.\n\n\n\n\n\n"
},

{
    "location": "ref/parameter_optimization/#Parameter-update-1",
    "page": "Optimizing Trainable Parameters",
    "title": "Parameter update",
    "category": "section",
    "text": "A parameter update reads from the gradient accumulators for certain trainable parameters, updates the values of those parameters, and resets the gradient accumulators to zero. A paramter update is constructed by combining an update configuration with the set of trainable parameters to which the update should be applied:ParamUpdateThe set of possible update configurations is described in Update configurations. An update is applied with:apply!"
},

{
    "location": "ref/parameter_optimization/#Gen.FixedStepGradientDescent",
    "page": "Optimizing Trainable Parameters",
    "title": "Gen.FixedStepGradientDescent",
    "category": "type",
    "text": "conf = FixedStepGradientDescent(step_size)\n\nConfiguration for stochastic gradient descent update with fixed step size.\n\n\n\n\n\n"
},

{
    "location": "ref/parameter_optimization/#Gen.GradientDescent",
    "page": "Optimizing Trainable Parameters",
    "title": "Gen.GradientDescent",
    "category": "type",
    "text": "conf = GradientDescent(step_size_init, step_size_beta)\n\nConfiguration for stochastic gradient descent update with step size given by (t::Int) -> step_size_init * (step_size_beta + 1) / (step_size_beta + t) where t is the iteration number.\n\n\n\n\n\n"
},

{
    "location": "ref/parameter_optimization/#Gen.ADAM",
    "page": "Optimizing Trainable Parameters",
    "title": "Gen.ADAM",
    "category": "type",
    "text": "conf = ADAM(learning_rate, beta1, beta2, epsilon)\n\nConfiguration for ADAM update.\n\n\n\n\n\n"
},

{
    "location": "ref/parameter_optimization/#Update-configurations-1",
    "page": "Optimizing Trainable Parameters",
    "title": "Update configurations",
    "category": "section",
    "text": "Gen has built-in support for the following types of update configurations.FixedStepGradientDescent\nGradientDescent\nADAMFor adding new types of update configurations, see Optimizing Trainable Parameters (Internal)."
},

{
    "location": "ref/trace_translators/#",
    "page": "Trace Translators",
    "title": "Trace Translators",
    "category": "page",
    "text": ""
},

{
    "location": "ref/trace_translators/#Trace-Translators-1",
    "page": "Trace Translators",
    "title": "Trace Translators",
    "category": "section",
    "text": "While Generative Functions define probability distributions on traces, Trace Translators convert from one space of traces to another space of traces. Trace translators are building blocks of inference programs that utilize multiple model representations, like Involutive MCMC.Trace translators are significantly more general than Bijectors. Trace translators can (i) convert between spaces of traces that include mixed numeric discrete random choices, as well as stochastic control flow, and (ii) convert between spaces for which there is no one-to-one correspondence (e.g. between models of different dimensionality, or between discrete and continuous models). Bijectors are limited to deterministic transformations between real-valued vectors of constant dimension."
},

{
    "location": "ref/trace_translators/#Deterministic-Trace-Translators-1",
    "page": "Trace Translators",
    "title": "Deterministic Trace Translators",
    "category": "section",
    "text": "Inference programs manipulate traces, but they also keep track of probabilities and probability densities associated with these traces. Suppose we have two generative functions p1 and p2. Given a trace t2 of p2 we can easily compute the probability (or probability density) that the trace would have been generated by p2 using get_score(t2). But suppose we want to construct the trace of p2 first sampling a trace t1 of p1 and then applying a deterministic transformation to that trace to obtain t2. How can we compute the probability that a trace t2 would have been produced by this process? This probability is needed if, for example, p2 defines a probabilistic model and want to use p1 as a proposal distribution within importance sampling. If we produce t2 via an arbitrary deterministic transformation of the random choices in t1, then computing the necessary probability is difficult.If we restrict ourselves to deterministic transformations that are bijections (one-to-one correspondences) from the set of traces of p1 to the set of traces of p2, then the problem is much simplified. If the transformation is a bijection this means that (i) each trace of p1 gets mapped to a different trace of p2, and (ii) for every trace of p2 there is some trace of p1 that maps to it. Bijective transformations between traces are useful components of inference programs because the probability that a given trace t2 of p2 would have been produced by first sampling from p1 and then applying the transform can be computed simply as the probability that p1 would produce  the (unique) trace t1 that gets mapped to the given trace by the transform. Conceptually, bijective trace transforms preserve probability. When trace transforms operate on traces with continuous random choices, computing probability densities of the transformed traces requires computing a Jacobian associated with the continuous part of the transformation.Gen provides a DSL for expressing bijections between spaces of traces, called the Trace Transform DSL. We introduce this DSL via an example. Below are two generative functions. The first samples polar coordinates and the second uses cartesian coordinates.@gen function p1()\n    r ~ inv_gamma(1, 1)\n    theta ~ uniform(-pi/2, pi/2)\nend@gen function p2()\n    x ~ normal(0, 1)\n    y ~ normal(0, 1)\nend"
},

{
    "location": "ref/trace_translators/#Defining-a-trace-transform-with-the-Trace-Transform-DSL-1",
    "page": "Trace Translators",
    "title": "Defining a trace transform with the Trace Transform DSL",
    "category": "section",
    "text": "The following trace transform DSL program defines a transformation (called f) that transforms traces of p1 into traces of p2:@transform f (t1) to (t2) begin\n    r = @read(t1[:r], continuous)\n    theta = @read(t1[theta], continuous)\n    @write(t2[:x], r * cos(theta), continuous)\n    @write(t2[:y], r * sin(theta), continuous)\nendThis transform reads values of random choices in the input trace (t1) at specific addresses (indicated by the syntax t1[addr]) using @read and writes values to the output trace (t2) using @write. Each read and write operation is labeled with whether the random choice is discrete or continuous. The section Trace Transform DSL defines the DSL in more detail.It is usually a good idea to write the inverse of the bijection. The inverse can provide a dynamic check that the transform truly is a bijection. The inverse of the above transformation is:@transform finv (t2) to (t1) begin\n    x = @read(t2[:x], continuous)\n    y = @read(t2[:y], continuous)\n    r = sqrt(x^2 + y^2)\n    @write(t1[:r], sqrt(x^2 + y^2), continuous)\n    @write(t1[:theta], atan(y, x), continuous)\nendWe can inform Gen that two transforms are inverses of one another using pair_bijections!:pair_bijections!(f, finv)"
},

{
    "location": "ref/trace_translators/#Wrapping-a-trace-transform-in-a-trace-translator-1",
    "page": "Trace Translators",
    "title": "Wrapping a trace transform in a trace translator",
    "category": "section",
    "text": "Note that the transform DSL code does not specify what the two generative functions are, or what the arguments to these generative functions are. This information will be required for computing probabilities and probability densities of traces. We provide this information by constructing a Trace Translator that wraps the transform along with this transformation:translator = DeterministicTraceTranslator(p2, (), f)We then can then apply the translator to a trace of p1 using function call syntax. The translator returns a trace of p2 and a log-weight that we can use to compute the probability (density) of the resulting trace:t2, log_weight = translator(t1)Specifically, the log probability (density) that the trace t2 was produced by first sampling t1 = simulate(p1, ()) and then applying the trace translator, is:get_score(t1) + log_weightLet\'s unpack the previous few code blocks in more detail. First, note that we did not pass in the source generative function (p1) or its arguments (()) when we constructed the trace translator. This is because this information will be attached to the input trace t1 itself. We did need to pass in the target generative function (p2) and its arguments (()) however, because this information is not included in t1.In this case, because continuous random choices are involved, the probabilities are probability densities, and the trace translator used automatic differentiation of the code in the transform f to compute a change-of-variables Jacobian that is necessary to compute the correct probability density of the new trace t2."
},

{
    "location": "ref/trace_translators/#Observations-1",
    "page": "Trace Translators",
    "title": "Observations",
    "category": "section",
    "text": "Typically, there are observations associated with one or both of the generative functions involved, and we have values for these in a choice map, so we do not want the trace translator to be responsible for populating the values of these observed random choices. For example, suppose we want to condition p2 on an observed random choice z:@gen function p2()\n    x ~ normal(0, 1)\n    y ~ normal(0, 1)\n    z ~ normal(x + y, 0.1)\nend\nobservations = choicemap()\nobservations[:z] = 2.3When p2 has observations, these can be passed in as an additional argument to the trace translator constructor:translator = DeterministicTraceTranslator(p2, (), observations, f)"
},

{
    "location": "ref/trace_translators/#Discrete-random-choices-and-stochastic-control-flow-1",
    "page": "Trace Translators",
    "title": "Discrete random choices and stochastic control flow",
    "category": "section",
    "text": "Trace transforms and trace translators interoperate seamlessly with discrete random choices and stochastic control flow. Both the input and the output traces can contain a mix of discrete and continuous choices, and arbitrary stochastic control flow. Consider the following simple example, where we use two different discrete representations to represent probability distributions the integers 0-7:@gen function p1()\n    bit1 ~ bernoulli(0.5)\n    bit2 ~ bernoulli(0.5)\n    bit3 ~ bernoulli(0.5)\nend@gen function p2()\n    n ~ categorical([0.1, 0.1, 0.1, 0.2, 0.2, 0.15, 0.15])\nendWe define the forward and inverse transforms:@transform f (t1) to (t2) begin\n    n = (\n        @read(t1[:bit1], :discrete) * 1 +\n        @read(t1[:bit2], :discrete) * 2 +\n        @read(t1[:bit3], :discrete) * 4)\n    @write(t2[:n], n, :discrete)\nend@transform finv (t2) to (t1) begin\n    bits = digits(@read(t2[:n], :discrete), base=2)\n    @write(t1[:bit1], bits[1], :discrete)\n    @write(t1[:bit2], bits[2], :discrete)\n    @write(t1[:bit3], bits[3], :discrete)\nendHere is an example that includes discrete random choices, stochastic control flow, and continuous random choices.@gen function p1()\n    if ({:branch} ~ bernoulli(0.5))\n        x ~ normal(0, 1)\n    else\n        other ~ categorical([0.3, 0.7])\n    end\nend@gen function p2()\n    k ~ uniform_discrete(1, 4)\n    if k <= 2\n        y ~ gamma(1, 1)\n    end\nendNote that transformations between spaces of traces need not be intuitive (although they probably should)! Try to convince yourself that the functions below are indeed a pair of bijections between the traces of these two generative functions.@transform f (t1) to (t2) begin\n    if @read(t1[:branch], :discrete)\n        x = @read(t1[:x], :continuous)\n        if x > 0\n            @write(t2[:k], 2, :discrete)\n        else\n            @write(t2[:k], 1, :discrete)\n        end\n        @write(t2[:y], abs(x), :continuous)\n    else\n        other = @read(t1[:other], :discrete)\n        @write(t2[:k], (other == 1) ? 3 : 4, :discrete)\n    end\nend@transform finv (t2) to (t1) begin\n    k = @read(t2[:k], :discrete)\n    if k <= 2\n        y = @read(t2[:y], :continuous)\n        @write(t2[:x], (k == 1) ? -y : y, :continuous)\n    else\n        @write(t1[:other], (k == 3) ? 1 : 2, :discrete)\n    end\nend"
},

{
    "location": "ref/trace_translators/#General-Trace-Translators-1",
    "page": "Trace Translators",
    "title": "General Trace Translators",
    "category": "section",
    "text": "Note that for two arbitrary generative functions p1 and p2 there may not exist any one-to-one correspondence between their spaces of traces. For example, consider a generative function p1 that samples points within the unit square 0 1^2@gen function p1()\n    x ~ uniform(0, 1)\n    y ~ uniform(0, 1)\nendand another generative function p2 that samples one of 100 possible discrete values, each value representing one cell of the unit square:@gen function p2()\n    i ~ uniform_discrete(1, 10) # interval [(i-1)/10, i/10]\n    j ~ uniform_discrete(1, 10) # interval [(j-1)/10, j/10]\nendThere is no one-to-one correspondence between the spaces of traces of these two generative functions: The first is an uncountably infinite set, and the other is a finite set with 100 elements in it.However, there is an intuitive notion of correspondence that we would like to be able to encode. Each discrete cell (i j) corresponds to a subset of the unit square (i - 1)10 i10 times (j-1)10 j10.We can express this correspondence (and any correspondence between two arbitrary generative functions) by introducing two auxiliary generative functions q1 and q2. The first function q1 will take a trace of p1 as input, and the second function q2 will take a trace of p2 as input. Then, instead of a transfomation between traces of p1 and traces of p2 our trace transform will transform between (i) the space of pairs of traces of p1 and q1 and (ii) the space of pairs of traces of p2 and q2. We construct q1 and q2 so that the two spaces have the same size, and a one-to-one correspondence is possible.For our example above, we construct q2 to sample the coordinate (0 01^2) relative to the cell. We construct q1 to be empty–there is already a mapping from each trace of p1 to each trace of p2 that simply identifies what cell (i j) a given point in 0 1^2 is in, so no extra random choices are needed.@gen function q1()\nend\n\n@gen function q2(p2_trace)\n    i = p2_trace[:i]\n    j = p2_trace[:j]\n    dx ~ uniform(0.0, 0.1)\n    dy ~ uniform(0.0, 0.1)\nend"
},

{
    "location": "ref/trace_translators/#Trace-transforms-between-pairs-of-traces-1",
    "page": "Trace Translators",
    "title": "Trace transforms between pairs of traces",
    "category": "section",
    "text": "To handle general trace translators that require auxiliary probability distributions, the trace trace DSL supports defining transformations between pairs of traces. For example, the following defines a trace transform that maps from pairs of traces of p1 and q1 to pairs of traces of p2 and q2:@transform f (p1_trace, q1_trace) to (p2_trace, q2_trace)\n    x = @read(p1_trace[:x], :continuous)\n    y = @read(p1_trace[:y], :continuous)\n    i = ceil(x * 10)\n    j = ceil(y * 10)\n    @write(p2_trace[:i], i, :discrete)\n    @write(p2_trace[:j], j, :discrete)\n    @write(q2_trace[:dx], x / 10, :continuous)\n    @write(q2_trace[:dy], y / 10, :continuous)\nendand the inverse transform:@transform f_inv (p2_trace, q2_trace) to (p1_trace, q1_trace)\n    i = @read(p2_trace[:i], :discrete)\n    j = @read(p2_trace[:j], :discrete)\n    dx = @read(q2_trace[:dx], :continuous)\n    dy = @read(q2_trace[:dy], :continuous)\n    x = (i-1)/10 + dx\n    y = (j-1)/10 + dy\n    @write(p1_trace[:x], x, :continuous)\n    @write(p1_trace[:y], x, :continuous)\nendwhich we associate as inverses:pair_bijections!(f, f_inv)"
},

{
    "location": "ref/trace_translators/#Constructing-a-general-trace-translator-1",
    "page": "Trace Translators",
    "title": "Constructing a general trace translator",
    "category": "section",
    "text": "We now wrap the transform above into a general trace translator, by providing the three probabilistic programs p2, q1, q2 that it uses (a reference to p1 will included in the input trace), and the arguments to these functions.translator = GeneralTraceTranslator(\n    p_new=p2,\n    p_new_args=(),\n    new_observations=choicemap(),\n    q_forward=q1,\n    q_forward_args=(),\n    q_backward=q2,\n    q_backward_args=(),\n    f=f)Then, we can apply the trace translator to a trace (t1) of p1 and get a trace (t2) of p2 and a log-weight:(t2, log_weight) = translator(t1)"
},

{
    "location": "ref/trace_translators/#Symmetric-Trace-Translators-1",
    "page": "Trace Translators",
    "title": "Symmetric Trace Translators",
    "category": "section",
    "text": "When the previous and new generative functions (e.g. p1 and p2 in the previous example) are the same, and their arguments are the same, and q_forward and q_backward (and their arguments) are also identical, we call this the trace translator a Symmetric Trace Translator. Symmetric trace translators are important because they form the basis of Involutive MCMC. Instead of translating a trace of one generative function to the trace of another generative function, they translate a trace of a generative function to another trace of the same generative function.Symmetric trace translators have the interesting property that the function f is an involution, or a function that is its own inverse. To indicate that a trace transform is an involution, use is_involution!.Because symmetric trace translators translate within the same generative function, their implementation uses update to incrementally modify the trace from the previous to the new trace. This has two benefits when the previous and new traces have random choices that aren\'t modified between them: (i) the incremental modification may be more efficient than writing the new trace entirely from scratch, and (ii) the transform DSL program does not need to specify a value for addresses whose value is not changed from the previous trace."
},

{
    "location": "ref/trace_translators/#Simple-Extending-Trace-Translators-1",
    "page": "Trace Translators",
    "title": "Simple Extending Trace Translators",
    "category": "section",
    "text": "TODO Document"
},

{
    "location": "ref/trace_translators/#Trace-Transform-DSL-1",
    "page": "Trace Translators",
    "title": "Trace Transform DSL",
    "category": "section",
    "text": "The Trace Transform DSL is a differentiable programming language for writing deterministic transformations of traces. Programs written in this DSL are called transforms. Transforms read the value of random choices from input trace(s) and write values of random choices to output trace(s). These programs are not typically executed directly by users, but are instead wrapped into one of the several forms of trace translators listed above (GeneralTraceTranslator, and SymmetricTraceTranslator).A transform is identified with the @transform macro, and uses one of the following four syntactic forms for the signature (the name of the transform, and the names of the input and output traces are all user-defined varibles; the only keywords are @transform, to, begin, and end):A transform from one trace to another, without extra parameters@transform f t1 to t2 begin\n    ...\nendA transform from one trace to another, with extra parameters@transform f(x, y, ..) t1 to t2 begin\n    ...\nendA transform from pairs of traces to pairs of traces, without extra parameters@transform f (t1, t2) to (t3, t4) begin\n    ...\nendA transform from one trace to another, with extra parameters@transform f(x, y, ..) (t1, t2) to (t3, t4) begin\n    ...\nendThe extra parameters are optional, and can be used to pass arguments to a transform function that is invoked, from another transform function, using the @tcall macro. For example:@transform g(x) t1 to t2 begin\n    ...\nend\n@transform f t1 to t2 begin\n    x = ..\n    @tcall(g(x))\nendThe callee transform function (g above) reads and writes to the same input and output traces as the caller transform function (f above). Note that the input and output traces can be assigned different names in the caller and the callee.The body of a transform reads the values of random choices at addresses in the input trace(s), performs computation using regular Julia code (provided this code can be differentiated with ForwardDiff.jl) and writes valeus of random choices at addresses in the output trace(s). In the body @read expressions read a value from a particular address of an input trace:val = @read(<source>, <type-label>)where <source> is an expression of the form <trace>[<addr>] where <trace> must be the name of an input trace in the transform\'s signature. The <type-label> is either :continuous or :discrete, and indicates whether the random choice is discrete or continuous (in measure-theoretic terms, whether it uses the counting measure, or a Lebesgue-measure a Euclidean space of some dimension). Similarly, @write expressions write a value to a particular address in an output trace:@write(<destination>, <value>, <type-label>)Sometimes trace transforms need to directly copy the value from one address in an input trace to one address in an output trace. In these cases, it is recommended to use the explicit @copy expression:@copy(<source>, <destination>)where <source> and <destination> are of the form <trace>[<addr>] as above. Note you can also copy collections of multiple random choices under an address namespace in an input trace to an address namespace in an output trace. For example,@copy(trace1[:foo], trace2[:bar])will copy every random choice in trace1 with an address of the form :foo => <rest> to trace2 at address :bar => <rest>.It is also possible to read the return value from an input trace using the following syntax, but this value must be discrete (in the local neighborhood of traces, the return value must be constant as a function of all continuous random choices in input traces):val = @read(<trace>[], :discrete)This feature is useful when the generative function precomputes a quantity as part of its return value, and we would like to reuse this value, instead of having to recompute it as part of the transform. The `discrete\' requirement is needed because the transform DSL does not currently backpropagate through the return value (this feature could be added in the future).Tips for defining valid transforms:If you find yourself copying the same continuous source address to multiple locations, it probably means your transform is not valid (the Jacobian matrix will have rows that are identical, and so the Jacobian determinant will be zero).\nYou can gain some confidence that your transform is valid by enabling dynamic checks (check=true) in the trace translator that uses it."
},

{
    "location": "ref/trace_translators/#Gen.@transform",
    "page": "Trace Translators",
    "title": "Gen.@transform",
    "category": "macro",
    "text": "@transform f[(params...)] (in1 [,in2]) to (out1 [,out2])\n    ..\nend\n\nWrite a program in the Trace Transform DSL.\n\n\n\n\n\n"
},

{
    "location": "ref/trace_translators/#Gen.@read",
    "page": "Trace Translators",
    "title": "Gen.@read",
    "category": "macro",
    "text": "@read(<source>, <annotation>)\n\nMacro for reading the value of a random choice from an input trace in the Trace Transform DSL.\n\n<source> is of the form <trace>[<addr>] where <trace> is an input trace, and <annotation> is either :discrete or :continuous.\n\n\n\n\n\n"
},

{
    "location": "ref/trace_translators/#Gen.@write",
    "page": "Trace Translators",
    "title": "Gen.@write",
    "category": "macro",
    "text": "@write(<destination>, <value>, <annotation>)\n\nMacro for writing the value of a random choice to an output trace in the Trace Transform DSL.\n\n<destination> is of the form <trace>[<addr>] where <trace> is an input trace, and <annotation> is either :discrete or :continuous.\n\n\n\n\n\n"
},

{
    "location": "ref/trace_translators/#Gen.@copy",
    "page": "Trace Translators",
    "title": "Gen.@copy",
    "category": "macro",
    "text": "@copy(<source>, <destination>)\n\nMacro for copying the value of a random choice (or a whole namespace of random choices) from an input trace to an output trace in the Trace Transform DSL.\n\n<destination> is of the form <trace>[<addr>] where <trace> is an input trace, and <annotation> is either :discrete or :continuous.\n\n\n\n\n\n"
},

{
    "location": "ref/trace_translators/#Gen.pair_bijections!",
    "page": "Trace Translators",
    "title": "Gen.pair_bijections!",
    "category": "function",
    "text": "pair_bijections!(f1::TraceTransformDSLProgram, f2::TraceTransformDSLProgram)\n\nAssert that a pair of bijections contsructed using the Trace Transform DSL are inverses of one another.\n\n\n\n\n\n"
},

{
    "location": "ref/trace_translators/#Gen.is_involution!",
    "page": "Trace Translators",
    "title": "Gen.is_involution!",
    "category": "function",
    "text": "is_involution!(f::TraceTransformDSLProgram)\n\nAssert that a bijection constructed with the Trace Transform DSL is its own inverse.\n\n\n\n\n\n"
},

{
    "location": "ref/trace_translators/#Gen.inverse",
    "page": "Trace Translators",
    "title": "Gen.inverse",
    "category": "function",
    "text": "b::TraceTransformDSLProgram = inverse(a::TraceTransformDSLProgram)\n\nObtain the inverse of a bijection that was constructed with the Trace Transform DSL.\n\nThe inverse must have been associated with the bijection either via pair_bijections! or [is_involution!])(@ref).\n\n\n\n\n\n"
},

{
    "location": "ref/trace_translators/#Gen.DeterministicTraceTranslator",
    "page": "Trace Translators",
    "title": "Gen.DeterministicTraceTranslator",
    "category": "type",
    "text": "translator = DeterministicTraceTranslator(;\n    p_new::GenerativeFunction, p_args::Tuple=();\n    new_observations::ChoiceMap=EmptyChoiceMap()\n    f::TraceTransformDSLProgram)\n\nConstructor for a deterministic trace translator.\n\nRun the translator with:\n\n(output_trace, log_weight) = translator(input_trace)\n\n\n\n\n\n"
},

{
    "location": "ref/trace_translators/#Gen.GeneralTraceTranslator",
    "page": "Trace Translators",
    "title": "Gen.GeneralTraceTranslator",
    "category": "type",
    "text": "translator = GeneralTraceTranslator(;\n    p_new::GenerativeFunction,\n    p_new_args::Tuple = (),\n    new_observations::ChoiceMap = EmptyChoiceMap(),\n    q_forward::GenerativeFunction,\n    q_forward_args::Tuple  = (),\n    q_backward::GenerativeFunction,\n    q_backward_args::Tuple  = (),\n    f::TraceTransformDSLProgram)\n\nConstructor for a general trace translator.\n\nRun the translator with:\n\n(output_trace, log_weight) = translator(input_trace; check=false, prev_observations=EmptyChoiceMap())\n\nUse check to enable a bijection check (this requires that the transform f has been paired with its inverse using `pair_bijections! or is_involution).\n\nIf check is enabled, then prev_observations is a choice map containing the observed random choices in the previous trace.\n\n\n\n\n\n"
},

{
    "location": "ref/trace_translators/#Gen.SimpleExtendingTraceTranslator",
    "page": "Trace Translators",
    "title": "Gen.SimpleExtendingTraceTranslator",
    "category": "type",
    "text": "translator = SimpleExtendingTraceTranslator(;\n    p_new_args::Tuple = (),\n    argdiffs::Tuple = (),\n    new_obs::ChoiceMap = EmptyChoiceMap(),\n    q_fwd::GenerativeFunction,\n    q_fwd_args::Tuple  = ())\n\nConstructor for a simple extending trace translator.\n\nRun the translator with:\n\n(output_trace, log_weight) = translator(input_trace)\n\n\n\n\n\n"
},

{
    "location": "ref/trace_translators/#Gen.SymmetricTraceTranslator",
    "page": "Trace Translators",
    "title": "Gen.SymmetricTraceTranslator",
    "category": "type",
    "text": "translator = SymmetricTraceTranslator(;\n    q::GenerativeFunction,\n    q_args::Tuple = (),\n    involution::Union{TraceTransformDSLProgram,Function})\n\nConstructor for a symmetric trace translator.\n\nThe involution is either constructed via the @transform macro (recommended), or can be provided as a Julia function.\n\nRun the translator with:\n\n(output_trace, log_weight) = translator(input_trace; check=false, observations=EmptyChoiceMap())\n\nUse check to enable the involution check (this requires that the transform f has been marked with is_involution).\n\nIf check is enabled, then observations is a choice map containing the observed random choices, and the check will additionally ensure they are not mutated by the involution.\n\n\n\n\n\n"
},

{
    "location": "ref/trace_translators/#API-1",
    "page": "Trace Translators",
    "title": "API",
    "category": "section",
    "text": "@transform\n@read\n@write\n@copy\npair_bijections!\nis_involution!\ninverse\nDeterministicTraceTranslator\nGeneralTraceTranslator\nSimpleExtendingTraceTranslator\nSymmetricTraceTranslator"
},

{
    "location": "ref/extending/#",
    "page": "Extending Gen",
    "title": "Extending Gen",
    "category": "page",
    "text": ""
},

{
    "location": "ref/extending/#Extending-Gen-1",
    "page": "Extending Gen",
    "title": "Extending Gen",
    "category": "section",
    "text": "Gen is designed for extensibility. To implement behaviors that are not directly supported by the existing modeling languages, users can implement `black-box\' generative functions directly, without using built-in modeling language. These generative functions can then be invoked by generative functions defined using the built-in modeling language. This includes several special cases:Extending Gen with custom gradient computations\nExtending Gen with custom incremental computation of return values\nExtending Gen with new modeling languages."
},

{
    "location": "ref/extending/#Gen.CustomGradientGF",
    "page": "Extending Gen",
    "title": "Gen.CustomGradientGF",
    "category": "type",
    "text": "CustomGradientGF{T}\n\nAbstract type for a generative function with a custom gradient computation, and default behaviors for all other generative function interface methods.\n\nT is the type of the return value.\n\n\n\n\n\n"
},

{
    "location": "ref/extending/#Gen.apply",
    "page": "Extending Gen",
    "title": "Gen.apply",
    "category": "function",
    "text": "retval = apply(gen_fn::CustomGradientGF, args)\n\nApply the function to the arguments.\n\n\n\n\n\n"
},

{
    "location": "ref/extending/#Gen.gradient",
    "page": "Extending Gen",
    "title": "Gen.gradient",
    "category": "function",
    "text": "arg_grads = gradient(gen_fn::CustomDetermGF, args, retval, retgrad)\n\nReturn the gradient tuple with respect to the arguments, where nothing is for argument(s) whose gradient is not available.\n\n\n\n\n\n"
},

{
    "location": "ref/extending/#Custom-gradients-1",
    "page": "Extending Gen",
    "title": "Custom gradients",
    "category": "section",
    "text": "To add a custom gradient for a differentiable deterministic computation, define a concrete subtype of CustomGradientGF with the following methods:apply\ngradient\nhas_argument_gradsFor example:struct MyPlus <: CustomGradientGF{Float64} end\n\nGen.apply(::MyPlus, args) = args[1] + args[2]\nGen.gradient(::MyPlus, args, retval, retgrad) = (retgrad, retgrad)\nGen.has_argument_grads(::MyPlus) = (true, true)CustomGradientGF\napply\ngradient"
},

{
    "location": "ref/extending/#Gen.CustomUpdateGF",
    "page": "Extending Gen",
    "title": "Gen.CustomUpdateGF",
    "category": "type",
    "text": "CustomUpdateGF{T,S}\n\nAbstract type for a generative function with a custom update computation, and default behaviors for all other generative function interface methods.\n\nT is the type of the return value and S is the type of state used internally for incremental computation.\n\n\n\n\n\n"
},

{
    "location": "ref/extending/#Gen.apply_with_state",
    "page": "Extending Gen",
    "title": "Gen.apply_with_state",
    "category": "function",
    "text": "retval, state = apply_with_state(gen_fn::CustomDetermGF, args)\n\nExecute the generative function and return the return value and the state.\n\n\n\n\n\n"
},

{
    "location": "ref/extending/#Gen.update_with_state",
    "page": "Extending Gen",
    "title": "Gen.update_with_state",
    "category": "function",
    "text": "state, retval, retdiff = update_with_state(gen_fn::CustomDetermGF, state, args, argdiffs)\n\nUpdate the arguments to the generative function and return new return value and state.\n\n\n\n\n\n"
},

{
    "location": "ref/extending/#Custom-incremental-computation-1",
    "page": "Extending Gen",
    "title": "Custom incremental computation",
    "category": "section",
    "text": "Iterative inference techniques like Markov chain Monte Carlo involve repeatedly updating the execution traces of generative models. In some cases, the output of a deterministic computation within the model can be incrementally computed during each of these updates, instead of being computed from scratch.To add a custom incremental computation for a deterministic computation, define a concrete subtype of CustomUpdateGF with the following methods:apply_with_state\nupdate_with_state\nhas_argument_gradsThe second type parameter of CustomUpdateGF is the type of the state that may be used internally to facilitate incremental computation within update_with_state.For example, we can implement a function for computing the sum of a vector that efficiently computes the new sum when a small fraction of the vector elements change:struct MyState\n    prev_arr::Vector{Float64}\n    sum::Float64\nend\n\nstruct MySum <: CustomUpdateGF{Float64,MyState} end\n\nfunction Gen.apply_with_state(::MySum, args)\n    arr = args[1]\n    s = sum(arr)\n    state = MyState(arr, s)\n    (s, state)\nend\n\nfunction Gen.update_with_state(::MySum, state, args, argdiffs::Tuple{VectorDiff})\n    arr = args[1]\n    prev_sum = state.sum\n    retval = prev_sum\n    for i in keys(argdiffs[1].updated)\n        retval += (arr[i] - state.prev_arr[i])\n    end\n    prev_length = length(state.prev_arr)\n    new_length = length(arr)\n    for i=prev_length+1:new_length\n        retval += arr[i]\n    end\n    for i=new_length+1:prev_length\n        retval -= arr[i]\n    end\n    state = MyState(arr, retval)\n    (state, retval, UnknownChange())\nend\n\nGen.num_args(::MySum) = 1CustomUpdateGF\napply_with_state\nupdate_with_state"
},

{
    "location": "ref/extending/#Gen.random",
    "page": "Extending Gen",
    "title": "Gen.random",
    "category": "function",
    "text": "val::T = random(dist::Distribution{T}, args...)\n\nSample a random choice from the given distribution with the given arguments.\n\n\n\n\n\n"
},

{
    "location": "ref/extending/#Gen.logpdf",
    "page": "Extending Gen",
    "title": "Gen.logpdf",
    "category": "function",
    "text": "lpdf = logpdf(dist::Distribution{T}, value::T, args...)\n\nEvaluate the log probability (density) of the value.\n\n\n\n\n\n"
},

{
    "location": "ref/extending/#Gen.has_output_grad",
    "page": "Extending Gen",
    "title": "Gen.has_output_grad",
    "category": "function",
    "text": "has::Bool = has_output_grad(dist::Distribution)\n\nReturn true of the gradient if the distribution computes the gradient of the logpdf with respect to the value of the random choice.\n\n\n\n\n\n"
},

{
    "location": "ref/extending/#Gen.logpdf_grad",
    "page": "Extending Gen",
    "title": "Gen.logpdf_grad",
    "category": "function",
    "text": "grads::Tuple = logpdf_grad(dist::Distribution{T}, value::T, args...)\n\nCompute the gradient of the logpdf with respect to the value, and each of the arguments.\n\nIf has_output_grad returns false, then the first element of the returned tuple is nothing. Otherwise, the first element of the tuple is the gradient with respect to the value. If the return value of has_argument_grads has a false value for at position i, then the i+1th element of the returned tuple has value nothing. Otherwise, this element contains the gradient with respect to the ith argument.\n\n\n\n\n\n"
},

{
    "location": "ref/extending/#custom_distributions-1",
    "page": "Extending Gen",
    "title": "Custom distributions",
    "category": "section",
    "text": "Users can extend Gen with new probability distributions, which can then be used to make random choices within generative functions. Simple transformations of existing distributions can be created using the @dist DSL. For arbitrary distributions, including distributions that cannot be expressed in the @dist DSL, users can define a custom distribution by implementing Gen\'s Distribution interface directly, as defined below.Probability distributions are singleton types whose supertype is Distribution{T}, where T indicates the data type of the random sample.abstract type Distribution{T} endA new Distribution type must implement the following methods:random\nlogpdf\nhas_output_grad\nlogpdf_grad\nhas_argument_gradsBy convention, distributions have a global constant lower-case name for the singleton value. For example:struct Bernoulli <: Distribution{Bool} end\nconst bernoulli = Bernoulli()Distribution values should also be callable, which is a syntactic sugar with the same behavior of calling random:bernoulli(0.5) # identical to random(bernoulli, 0.5) and random(Bernoulli(), 0.5)random\nlogpdf\nhas_output_grad\nlogpdf_grad"
},

{
    "location": "ref/extending/#Custom-generative-functions-1",
    "page": "Extending Gen",
    "title": "Custom generative functions",
    "category": "section",
    "text": "We recommend the following steps for implementing a new type of generative function, and also looking at the implementation for the DynamicDSLFunction type as an example."
},

{
    "location": "ref/extending/#Define-a-trace-data-type-1",
    "page": "Extending Gen",
    "title": "Define a trace data type",
    "category": "section",
    "text": "struct MyTraceType <: Trace\n    ..\nend"
},

{
    "location": "ref/extending/#Decide-the-return-type-for-the-generative-function-1",
    "page": "Extending Gen",
    "title": "Decide the return type for the generative function",
    "category": "section",
    "text": "Suppose our return type is Vector{Float64}."
},

{
    "location": "ref/extending/#Define-a-data-type-for-your-generative-function-1",
    "page": "Extending Gen",
    "title": "Define a data type for your generative function",
    "category": "section",
    "text": "This should be a subtype of GenerativeFunction, with the appropriate type parameters.struct MyGenerativeFunction <: GenerativeFunction{Vector{Float64},MyTraceType}\n..\nendNote that your generative function may not need to have any fields. You can create a constructor for it, e.g.:function MyGenerativeFunction(...)\n..\nend"
},

{
    "location": "ref/extending/#Decide-what-the-arguments-to-a-generative-function-should-be-1",
    "page": "Extending Gen",
    "title": "Decide what the arguments to a generative function should be",
    "category": "section",
    "text": "For example, our generative functions might take two arguments, a (of type Int) and b (of type Float64). Then, the argument tuple passed to e.g. generate will have two elements.NOTE: Be careful to distinguish between arguments to the generative function itself, and arguments to the constructor of the generative function. For example, if you have a generative function type that is parametrized by, for example, modeling DSL code, this DSL code would be a parameter of the generative function constructor."
},

{
    "location": "ref/extending/#Decide-what-the-traced-random-choices-(if-any)-will-be-1",
    "page": "Extending Gen",
    "title": "Decide what the traced random choices (if any) will be",
    "category": "section",
    "text": "Remember that each random choice is assigned a unique address in (possibly) hierarchical address space. You are free to design this address space as you wish, although you should document it for users of your generative function type."
},

{
    "location": "ref/extending/#Implement-methods-of-the-Generative-Function-Interface-1",
    "page": "Extending Gen",
    "title": "Implement methods of the Generative Function Interface",
    "category": "section",
    "text": "At minimum, you need to implement the following methods:simulate\nhas_argument_grads\naccepts_output_grad\nget_args\nget_retval\nget_choices\nget_score\nget_gen_fn\nprojectIf you want to use the generative function within models, you should implement:generateIf you want to use MCMC on models that call your generative function, then implement:update\nregenerateIf you want to use gradient-based inference techniques on models that call your generative function, then implement:choice_gradients\nupdateIf your generative function has trainable parameters, then implement:accumulate_param_gradients!"
},

{
    "location": "ref/extending/#Custom-modeling-languages-1",
    "page": "Extending Gen",
    "title": "Custom modeling languages",
    "category": "section",
    "text": "Gen can be extended with new modeling languages by implementing new generative function types, and constructors for these types that take models as input. This typically requires implementing the entire generative function interface, and is advanced usage of Gen."
},

{
    "location": "ref/importance/#",
    "page": "Importance Sampling",
    "title": "Importance Sampling",
    "category": "page",
    "text": ""
},

{
    "location": "ref/importance/#Gen.importance_sampling",
    "page": "Importance Sampling",
    "title": "Gen.importance_sampling",
    "category": "function",
    "text": "(traces, log_norm_weights, lml_est) = importance_sampling(model::GenerativeFunction,\n    model_args::Tuple, observations::ChoiceMap, num_samples::Int, verbose=false)\n\n(traces, log_norm_weights, lml_est) = importance_sampling(model::GenerativeFunction,\n    model_args::Tuple, observations::ChoiceMap,\n    proposal::GenerativeFunction, proposal_args::Tuple,\n    num_samples::Int, verbose=false)\n\nRun importance sampling, returning a vector of traces with associated log weights.\n\nThe log-weights are normalized. Also return the estimate of the marginal likelihood of the observations (lml_est). The observations are addresses that must be sampled by the model in the given model arguments. The first variant uses the internal proposal distribution of the model. The second variant uses a custom proposal distribution defined by the given generative function. All addresses of random choices sampled by the proposal should also be sampled by the model function. Setting verbose=true prints a progress message every sample.\n\n\n\n\n\n"
},

{
    "location": "ref/importance/#Gen.importance_resampling",
    "page": "Importance Sampling",
    "title": "Gen.importance_resampling",
    "category": "function",
    "text": "(trace, lml_est) = importance_resampling(model::GenerativeFunction,\n    model_args::Tuple, observations::ChoiceMap, num_samples::Int,\n    verbose=false)\n\n(traces, lml_est) = importance_resampling(model::GenerativeFunction,\n    model_args::Tuple, observations::ChoiceMap,\n    proposal::GenerativeFunction, proposal_args::Tuple,\n    num_samples::Int, verbose=false)\n\nRun sampling importance resampling, returning a single trace.\n\nUnlike importance_sampling, the memory used constant in the number of samples.\n\nSetting verbose=true prints a progress message every sample.\n\n\n\n\n\n"
},

{
    "location": "ref/importance/#Importance-Sampling-1",
    "page": "Importance Sampling",
    "title": "Importance Sampling",
    "category": "section",
    "text": "importance_sampling\nimportance_resampling"
},

{
    "location": "ref/map/#",
    "page": "MAP Optimization",
    "title": "MAP Optimization",
    "category": "page",
    "text": ""
},

{
    "location": "ref/map/#Gen.map_optimize",
    "page": "MAP Optimization",
    "title": "Gen.map_optimize",
    "category": "function",
    "text": "new_trace = map_optimize(trace, selection::Selection,\n    max_step_size=0.1, tau=0.5, min_step_size=1e-16, verbose=false)\n\nPerform backtracking gradient ascent to optimize the log probability of the trace over selected continuous choices.\n\nSelected random choices must have support on the entire real line.\n\n\n\n\n\n"
},

{
    "location": "ref/map/#MAP-Optimization-1",
    "page": "MAP Optimization",
    "title": "MAP Optimization",
    "category": "section",
    "text": "map_optimize"
},

{
    "location": "ref/mcmc/#",
    "page": "Markov chain Monte Carlo",
    "title": "Markov chain Monte Carlo",
    "category": "page",
    "text": ""
},

{
    "location": "ref/mcmc/#Markov-chain-Monte-Carlo-(MCMC)-1",
    "page": "Markov chain Monte Carlo",
    "title": "Markov chain Monte Carlo (MCMC)",
    "category": "section",
    "text": "Markov chain Monte Carlo (MCMC) is an approach to inference which involves initializing a hypothesis and then repeatedly sampling a new hypotheses given the previous hypothesis by making a change to the previous hypothesis. The function that samples the new hypothesis given the previous hypothesis is called the MCMC kernel (or `kernel\' for short). If we design the kernel appropriately, then the distribution of the hypotheses will converge to the conditional (i.e. posterior) distribution as we increase the number of times we apply the kernel.Gen includes primitives for constructing MCMC kernels and composing them into MCMC algorithms. Although Gen encourages you to write MCMC algorithms that converge to the conditional distribution, Gen does not enforce this requirement. You may use Gen\'s MCMC primitives in other ways, including for stochastic optimization.For background on MCMC see [1].[1] Andrieu, Christophe, et al. \"An introduction to MCMC for machine learning.\" Machine learning 50.1-2 (2003): 5-43. Link."
},

{
    "location": "ref/mcmc/#MCMC-in-Gen-1",
    "page": "Markov chain Monte Carlo",
    "title": "MCMC in Gen",
    "category": "section",
    "text": "Suppose we are doing inference in the following toy model:@gen function model()\n    x = @trace(bernoulli(0.5), :x) # a latent variable\n    @trace(normal(x ? -1. : 1., 1.), :y) # the variable that will be observed\nendTo do MCMC, we first need to obtain an initial trace of the model. Recall that a trace encodes both the observed data and hypothesized values of latent variables. We can obtain an initial trace that encodes the observed data, and contains a randomly initialized hypothesis, using generate, e.g.:observations = choicemap((:y, 1.23))\ntrace, = generate(model, (), observations)Then, an MCMC algorithm is Gen is implemented simply by writing Julia for loop, which repeatedly applies a kernel, which is a regular Julia function:for i=1:100\n    trace = kernel(trace)\nend"
},

{
    "location": "ref/mcmc/#Built-in-Stationary-Kernels-1",
    "page": "Markov chain Monte Carlo",
    "title": "Built-in Stationary Kernels",
    "category": "section",
    "text": "However, we don\'t expect to be able to use any function for kernel and expect to converge to the conditional distribution. To converge to the conditional distribution, the kernels must satisfy some properties. One of these properties is that the kernel is stationary with respect to the conditional distribution. Gen\'s inference library contains a number of functions for constructing stationary kernels:metropolis_hastings with alias mh, which has three variants with differing tradeoffs between ease-of-use and efficiency. The simplest variant simply requires you to select the set of random choices to be updated, without specifying how. The middle variant allows you to use custom proposals that encode problem-specific heuristics, or custom proposals based on neural networks that are trained via amortized inference. The most sophisticated variant allows you to specify any kernel in the reversible jump MCMC framework.\nmala, which performs a Metropolis Adjusted Langevin algorithm update on a set of selected random choices.\nhmc, which performs a Hamiltonian Monte Carlo update on a set of selected random choices.\nelliptical_slice, which performs an elliptical slice sampling update on a selected multivariate normal random choice.For example, here is an MCMC inference algorithm that uses mh:function do_inference(y, num_iters)\n    trace, = generate(model, (), choicemap((:y, y)))\n    xs = Float64[]\n    for i=1:num_iters\n        trace, = mh(trace, select(:x))\n        push!(xs, trace[:x])\n    end\n    xs\nendNote that each of the kernel functions listed above stationary with respect to the joint distribution on traces of the model, but may not be stationary with respect to the intended conditional distribution, which is determined by the set of addresses that consititute the observed data. If a kernel modifies the values of any of the observed data, then the kernel is not stationary with respect to the conditional distribution. Therefore, you should ensure that your MCMC kernels never propose to the addresses of the observations.Note that stationarity with respect to the conditional distribution alone is not sufficient for a kernel to converge to the posterior with infinite iterations. Other requirements include that the chain is irreducible (it is possible to get from any state to any other state in a finite number of steps), and aperiodicity, which is a more complex requirement that is satisfied when kernels have some probability of staying in the same state, which most of the primitive kernels above satisfy. We refer interested readers to [1] for additional details on MCMC convergence."
},

{
    "location": "ref/mcmc/#Enabling-Dynamic-Checks-1",
    "page": "Markov chain Monte Carlo",
    "title": "Enabling Dynamic Checks",
    "category": "section",
    "text": "Gen does not statically guarantee that kernels (either ones built-in or composed with the Composite Kernel DSL) are stationary. However, you can enable dynamic checks that will detect common bugs that break stationarity. To enable the dynamic checks we pass a keyword argument beyond those of the kernel itself:new_trace = k(trace, 2, check=true)Note that these checks aim to detect when a kernel is not stationary with respect to the model\'s joint distribution. To add an additional dynamic check for violation of stationarity with respect to the conditional distribution (conditioned on observations), we pass in an additional keyword argument containing a choice map with the observations:new_trace = k(traced, 2, check=true, observations=choicemap((:y, 1.2)))If check is set to false, then the observation check is not performed."
},

{
    "location": "ref/mcmc/#Composite-Kernel-DSL-1",
    "page": "Markov chain Monte Carlo",
    "title": "Composite Kernel DSL",
    "category": "section",
    "text": "You can freely compose the primitive kernels listed above into more complex kernels. Common types of composition including e.g. cycling through multiple kernels, randomly choosing a kernel to apply, and choosing which kernel to apply based on the current state. However, not all such compositions of stationary kernels will result in kernels that are themselves stationary.Gen\'s Composite Kernel DSL is an embedded inference DSL that allows for more safe composition of MCMC kernels, by formalizing properties of the compositions that are sufficient for stationarity, encouraging compositions with these properties, and dynamically checking for violation of these properties. Although the DSL does not guarantee stationarity of the composite kernels, its dynamic checks do catch common cases of non-stationary kernels. The dynamic checks can be enabled and disabled as needed (e.g. enabled during testing and prototyping and disabled during deployment for higher performance).The DSL consists of a macro – @kern for composing stationary kernels from primitive stationary kernels and composite stationary kernels, and two additional macros: –- @pkern for declaring Julia functions to be custom primitive stationary kernels, and @rkern for declaring the reversal of a custom primitive kernel (these two macros are advanced features not necessary for standard MCMC algorithms)."
},

{
    "location": "ref/mcmc/#Composing-Stationary-Kernels-1",
    "page": "Markov chain Monte Carlo",
    "title": "Composing Stationary Kernels",
    "category": "section",
    "text": "The @kern macro defines a composite MCMC kernel in a restricted DSL that is based on Julia\'s own function definition syntax.Suppose we are doing inference in the following model:@gen function model()\n    n = @trace(geometric(0.5), :n)\n    total = 0.\n    for i=1:n\n        total += @trace(normal(0, 1), (:x, i))\n    end\n    @trace(normal(total, 1.), :y)\n    total\nendHere is an example composite kernel for MCMC in this model:@kern function my_kernel(trace)\n    \n    # cycle through the x\'s and do a random walk update on each one\n    for i in 1:trace[:n]\n        trace ~ mh(trace, random_walk_proposal, (i,))\n    end\n\n    # repeatedly pick a random x and do a random walk update on it\n    if trace[:n] > 0\n        for rep in 1:10\n            let i ~ uniform_discrete(1, trace[:n])\n                trace ~ mh(trace, random_walk_proposal, (i,))\n            end\n        end\n    end\n\n    # remove the last x, or add a new one, a random number of times\n    let n_add_remove_reps ~ uniform_discrete(0, max_n_add_remove)\n        for rep in 1:n_add_remove_reps\n            trace ~ mh(trace, add_remove_proposal, (), add_remove_involution)\n        end\n    end\nendIn the DSL, the first argument (trace in this case) represents the trace on which the kernel is acting. the kernel may have additional arguments. The code inside the body can read from the trace (e.g. trace[:n] reads the value of the random choice :n). Finally, the return value of the composite kernel is automatically set to the trace. NOTE: It is not permitted to assign to the trace variable, except with ~ expressions. Also note that stationary kernels, when treated as Julia functions, return a tuple, where the first element is the trace and the remaining arguments are metadata. When applying these kernels with ~ syntax within the DSL, it is not necessary to unpack the tuple (the metadata is ignored automatically).The language constructs supported by this DSL are:Applying a stationary kernel. To apply a kernel, the syntax trace ~ k(trace, args..) is used. Note that the check and observations keyword arguments (see Enabling Dynamic Checks) should not be used here; they will be added automatically.For loops. The range of the for loop may be a deterministic function of the trace (as in trace[:n] above). The range must be invariant under all possible executions of the body of the for loop. For example, the random walk based kernel embedded in the for loop in our example above cannot modify the value of the random choice :n in the trace.If-end expressions The predicate condition may be a deterministic function of the trace, but it also must be invariant (i.e. remain true) under all possible executions of the body.Deterministic let expressions. We can use let x = value .. end to bind values to a variable, but the expression on the right-hand-side must be deterministic function of its free variables, its value must be invariant under all possible executions of the body.Stochastic let expressions. We can use let x ~ dist(args...) .. end to sample a stochastic value and bind to a variable, but the expression on the right-hand-side must be the application of a Gen Distribution to arguments, and the distribution and its arguments must be invariant under all possible executions of the body."
},

{
    "location": "ref/mcmc/#Declaring-primitive-kernels-for-use-in-composite-kernels-1",
    "page": "Markov chain Monte Carlo",
    "title": "Declaring primitive kernels for use in composite kernels",
    "category": "section",
    "text": "Note that all calls to built-in kernels like mh should be stationary, but that users are also free to declare their own arbitrary code as stationary. The @pkern macro declares a Julia function as a stationary MCMC kernel, for use with the MCMC Kernel DSL. The following custom primitive kernel permutes the random variables using random permutation generated from outside of Gen: @pkern function permute_move(trace; check=false, observations=EmptyChoiceMap())\n    perm = Random.randperm(trace[:n])\n    constraints = choicemap()\n    for (i, j) in enumerate(perm)\n        constraints[(:x, i)] = trace[(:x, j)]\n        constraints[(:x, j)] = trace[(:x, i)]\n    end\n    trace, = update(trace, (), (), constraints)\n    metadata = nothing\n    trace, metadata\nendThe first argument to the function should be the trace, and the function must have keyword arguments check and observations (see Enabling Dynamic Checks). The return value should be a tuple where the first element is the new trace (and any remaining elements are optional metadata).Primitive kernels are Julia functions. Note that although we will be invoking these kernels within @kern functions, these kernels can still be called like a regular Julia function.new_trace = permute_move(trace, 2)Indeed, they are just regular Julia functions, but with some extra information attached so that the composite kernel DSL knows they have been declared as stationary kernels."
},

{
    "location": "ref/mcmc/#Involutive-MCMC-1",
    "page": "Markov chain Monte Carlo",
    "title": "Involutive MCMC",
    "category": "section",
    "text": "Gen\'s most flexible variant of metropolis_hastings, called Involutive MCMC, allows users to specify any MCMC kernel in the reversible jump MCMC (RJMCMC) framework [2]. Involution MCMC allows you to express a broad class of custom MCMC kernels that are not expressible using the other, simpler variants of Metropolis-Hastings supported by Gen. These kernels are particularly useful for inferring the structure (e.g. control flow) of a model.[2] Green, Peter J. \"Reversible jump Markov chain Monte Carlo computation and Bayesian model determination.\" Biometrika 82.4 (1995): 711-732. LinkAn involutive MCMC kernel in Gen takes as input a previous trace of the model (whose choice map we will denote by t), and performs three phases to obtain a new trace of the model:First, it traces the execution of a proposal, which is an auxiliary generative function that takes the previous trace of the model as its first argument. Mathematically, we will denote the choice map associated with the trace of the proposal by u. The proposal can of course be defined using the Built-In Modeling Languages, just like the model itself. However, unlike many other uses of proposals in Gen, these proposals can make random choices at addresses that the model does not.\nNext, it takes the tuple (t u) and passes it into an involution (denoted mathematically by h), which is a function that returns a new tuple (t u), where t is the choice map for a new proposed trace of the model, and u are random choices for a new trace of the proposal. The defining property of the involution is that it is invertible, and it is its own inverse; i.e. (t u) = h(h(t u)). Intuitively, u is a description of a way that the proposal could be reversed, taking t to t.\nFinally, it computes an acceptance probability, which involves computing certain derivatives associated with the involution, and stochastically accepts or rejects the proposed model trace according to this probability. The involution is typically defined using the Trace Transform DSL, in which case the acceptance probability calculation is fully automated."
},

{
    "location": "ref/mcmc/#Example-1",
    "page": "Markov chain Monte Carlo",
    "title": "Example",
    "category": "section",
    "text": "Consider the following generative model of two pieces of observed data, at addresses :y1 and :y2.@gen function model()\n    if ({:z} ~ bernoulli(0.5))\n        m1 = ({:m1} ~ gamma(1, 1))\n        m2 = ({:m2} ~ gamma(1, 1))\n    else\n        m = ({:m} ~ gamma(1, 1))\n        (m1, m2) = (m, m)\n    end\n    {:y1} ~ normal(m1, 0.1)\n    {:y2} ~ normal(m2, 0.1)\nendBecause this model has stochastic control flow, it represents two distinct structural hypotheses about how the observed data could have been generated: If :z is true then we enter the first branch, and we hypothesize that the two data points were generated from separate means, sampled at addresses :m1 and :m2. If :z is false then we enter the second branch, and we hypohesize that there is a single mean that explains both data points, sampled at address :m.We want to construct an MCMC kernel that is able to transition between these two distinct structural hypotheses. We could construct such a kernel with the simpler \'selection\' variant of Metropolis-Hastings, by selecting the address :z, e.g.:select_mh_structure_kernel(trace) = mh(trace, select(:z))[1]Sometimes, this kernel would propose to change the value of :z. We could interleave this kernel with another kernel that does inference over the mean random choices, without changing the structure, e.g.:@gen function fixed_structure_proposal(trace)\n    if trace[:z]\n        {:m1} ~ normal(trace[:m1], 0.1)\n        {:m2} ~ normal(trace[:m2], 0.1)\n    else\n        {:m} ~ normal(trace[:m], 0.1)\n    end\nend\n\nfixed_structure_kernel(trace) = mh(trace, fixed_structure_proposal, ())[1]Combining these together, and applying to particular data and with a specific initial hypotheses:(y1, y2) = (1.0, 1.3)\ntrace, = generate(model, (), choicemap((:y1, y1), (:y2, y2), (:z, false), (:m, 1.2)))\nfor iter=1:100\n    trace = select_mh_structure_kernel(trace)\n    trace = fixed_structure_kernel(trace)\nendHowever, this algorithm will not be very efficient, because the internal proposal used by the selection variant of MH is not specialized to the model. In particular, when switching from the model with a single mean to the model with two means, the values of the new addresses :m1 and :m2 will be proposed from the prior distribution. This is wasteful, since if we have inferred an accurate value for :m, we expect the values for :m1 and :m2 to be near this value. The same is true when proposing a structure change in the opposite direction. That means it will take many more steps to get an accurate estimate of the posterior probability distribution on the two structures.We would like to use inferred values for :m1 and :m2 to inform our proposal for the value of :m. For example, we could take the geometric mean:m = sqrt(m1 * m2)However, there are many combinations of m1 and m2 that have the same geometric mean. In other words, the geometric mean is not invertible. However, if we return the additional degree of freedom alongside the geometric mean (dof), then we do have an invertible function:function merge_means(m1, m2)\n    m = sqrt(m1 * m2)\n    dof = m1 / (m1 + m2)\n    (m, dof)\nendThe inverse function is:function split_mean(m, dof)\n    m1 = m * sqrt((dof / (1 - dof)))\n    m2 = m * sqrt(((1 - dof) / dof))\n    (m1, m2)\nendWe use these two functions to construct an involution, and we use this involution with metropolis_hastings to construct an MCMC kernel that we call a \'split/merge\' kernel, because it either splits a parameter value, or merges two parameter values. The proposal is responsible for generating the extra degree of freedom when splitting:@gen function split_merge_proposal(trace)\n    if trace[:z]\n        # currently two segments, switch to one\n    else\n        # currently one segment, switch to two\n        {:dof} ~ uniform_continuous(0, 1)\n    end\nendFinally, we write the involution itself, using the Trace Transform DSL:@transform split_merge_involution (model_in, aux_in) to (model_out, aux_out) begin\n    if @read(model_in[:z], :discrete)\n\n        # currently two means, switch to one\n        @write(model_out[:z], false, :discrete)\n        m1 = @read(model_in[:m1], :continuous)\n        m2 = @read(model_in[:m2], :continuous)\n        (m, u) = merge_mean(m1, m2)\n        @write(model_out[:m], m, :continuous)\n        @write(aux_out[:u], u, :continuous)\n    else\n\n        # currently one mean, switch to two\n        @write(model_out[:z], true, :discrete)\n        m = @read(model_in[:m], :continuous)\n        u = @read(aux_in[:u], :continuous)\n        (m1, m2) = split_mean(m, u)\n        @write(model_out[:m1], m1, :continuous)\n        @write(model_out[:m2], m2, :continuous)\n    end\nendThe body of this function reads values from (t u) at specific addresses and writes values to (t u) at specific addresses, where t and t are called \'model\' choice maps, and u and u are called \'proposal\' choice maps. Note that the inputs and outputs of this function are not represented in the same way as arguments or return values of regular Julia functions –- they are implicit and can only be read from and written to, respectively, using a set of special macros (listed below). You should convince yourself that this function is invertible and its own inverse.Finally, we compose a structure-changing MCMC kernel using this involution:split_merge_kernel(trace) = mh(trace, split_merge_proposal, (), split_merge_involution)We then compose this move with the fixed structure move, and run it on the observed data:(y1, y2) = (1.0, 1.3)\ntrace, = generate(model, (), choicemap((:y1, y1), (:y2, y2), (:z, false), (:m, 1.)))\nfor iter=1:100\n    trace = split_merge_kernel(trace)\n    trace = fixed_structure_kernel(trace)\nendWe can then compare the results to the results from the Markov chain that used the selection-based structure-changing kernel:(Image: rjmcmc plot)We see that if we initialize the Markov chains from the same state with a single mean (:z is false) then the selection-based kernel fails to accept any moves to the two-mean structure within 100 iterations, whereas the split-merge kernel transitions back and forth many times, If we repeated the selection-based kernel for enough iterations, it would eventually transition back and forth at the same rate as the split-merge. The split-merge kernel gives a much more efficient inference algorithm for estimating the posterior probability on the two structures."
},

{
    "location": "ref/mcmc/#Reverse-Kernels-1",
    "page": "Markov chain Monte Carlo",
    "title": "Reverse Kernels",
    "category": "section",
    "text": "The reversal of a stationary MCMC kernel with distribution k_1(t t), for model with distribution p(t x), is another MCMC kernel with distribution:k_2(t t) = fracp(t x)p(t x) k_1(t t)For custom primitive kernels declared with @pkern, users can declare the reversal kernel with the @rkern macro:@rkern k1 : k2This also assigns k1 as the reversal of k2. The composite kernel DSL automatically generates the reversal kernel for composite kernels, and built-in stationary kernels like mh. The reversal of a kernel (primitive or composite) can be obtained with reversal."
},

{
    "location": "ref/mcmc/#Gen.metropolis_hastings",
    "page": "Markov chain Monte Carlo",
    "title": "Gen.metropolis_hastings",
    "category": "function",
    "text": "(new_trace, accepted) = metropolis_hastings(\n    trace, selection::Selection;\n    check=false, observations=EmptyChoiceMap())\n\nPerform a Metropolis-Hastings update that proposes new values for the selected addresses from the internal proposal (often using ancestral sampling), returning the new trace (which is equal to the previous trace if the move was not accepted) and a Bool indicating whether the move was accepted or not.\n\n\n\n\n\n(new_trace, accepted) = metropolis_hastings(\n    trace, proposal::GenerativeFunction, proposal_args::Tuple;\n    check=false, observations=EmptyChoiceMap())\n\nPerform a Metropolis-Hastings update that proposes new values for some subset of random choices in the given trace using the given proposal generative function, returning the new trace (which is equal to the previous trace if the move was not accepted) and a Bool indicating whether the move was accepted or not.\n\nThe proposal generative function should take as its first argument the current trace of the model, and remaining arguments proposal_args. If the proposal modifies addresses that determine the control flow in the model, values must be provided by the proposal for any addresses that are newly sampled by the model.\n\n\n\n\n\n(new_trace, accepted) = metropolis_hastings(\n    trace, proposal::GenerativeFunction, proposal_args::Tuple,\n    involution::Union{TraceTransformDSLProgram,Function};\n    check=false, observations=EmptyChoiceMap())\n\nPerform a generalized (reversible jump) Metropolis-Hastings update based on an involution (bijection that is its own inverse) on a space of choice maps, returning the new trace (which is equal to the previous trace if the move was not accepted) and a Bool indicating whether the move was accepted or not.\n\nMost users will want to construct involution using the Trace Transform DSL with the @transform macro, but for more user control it is also possible to provide a Julia function for involution, that has the following signature:\n\n(new_trace, bwd_choices::ChoiceMap, weight) = involution(trace::Trace, fwd_choices::ChoiceMap, fwd_retval, fwd_args::Tuple)\n\nThe generative function proposal is executed on arguments (trace, proposal_args...), producing a choice map fwd_choices and return value fwd_ret. For each value of model arguments (contained in trace) and proposal_args, the involution function applies an involution that maps the tuple (get_choices(trace), fwd_choices) to the tuple (get_choices(new_trace), bwd_choices). Note that fwd_ret is a deterministic function of fwd_choices and proposal_args. When only discrete random choices are used, the weight must be equal to get_score(new_trace) - get_score(trace).\n\nWhen continuous random choices are used, the weight returned by the involution must include an additive correction term that is the determinant of the the Jacobian of the bijection on the continuous random choices that is obtained by currying the involution on the discrete random choices (this correction term is automatically computed if the involution is constructed using the Trace Transform DSL). NOTE: The Jacobian matrix of the bijection on the continuous random choices must be full-rank (i.e. nonzero determinant). The check keyword argument to the involution can be used to enable or disable any dynamic correctness checks that the involution performs; for successful executions, check does not alter the return value.\n\n\n\n\n\n"
},

{
    "location": "ref/mcmc/#Gen.mh",
    "page": "Markov chain Monte Carlo",
    "title": "Gen.mh",
    "category": "function",
    "text": "(new_trace, accepted) = mh(trace, selection::Selection; ..)\n(new_trace, accepted) = mh(trace, proposal::GenerativeFunction, proposal_args::Tuple; ..)\n(new_trace, accepted) = mh(trace, proposal::GenerativeFunction, proposal_args::Tuple, involution; ..)\n\nAlias for metropolis_hastings. Perform a Metropolis-Hastings update on the given trace.\n\n\n\n\n\n"
},

{
    "location": "ref/mcmc/#Gen.mala",
    "page": "Markov chain Monte Carlo",
    "title": "Gen.mala",
    "category": "function",
    "text": "(new_trace, accepted) = mala(\n    trace, selection::Selection, tau::Real;\n    check=false, observations=EmptyChoiceMap())\n\nApply a Metropolis-Adjusted Langevin Algorithm (MALA) update.\n\nReference URL\n\n\n\n\n\n"
},

{
    "location": "ref/mcmc/#Gen.hmc",
    "page": "Markov chain Monte Carlo",
    "title": "Gen.hmc",
    "category": "function",
    "text": "(new_trace, accepted) = hmc(\n    trace, selection::Selection; L=10, eps=0.1,\n    check=false, observations=EmptyChoiceMap())\n\nApply a Hamiltonian Monte Carlo (HMC) update that proposes new values for the selected addresses, returning the new trace (which is equal to the previous trace if the move was not accepted) and a Bool indicating whether the move was accepted or not.\n\nHamilton\'s equations are numerically integrated using leapfrog integration with step size eps for L steps. See equations (5.18)-(5.20) of Neal (2011).\n\nReferences\n\nNeal, Radford M. (2011), \"MCMC Using Hamiltonian Dynamics\", Handbook of Markov Chain Monte Carlo, pp. 113-162. URL: http://www.mcmchandbook.net/HandbookChapter5.pdf\n\n\n\n\n\n"
},

{
    "location": "ref/mcmc/#Gen.elliptical_slice",
    "page": "Markov chain Monte Carlo",
    "title": "Gen.elliptical_slice",
    "category": "function",
    "text": "new_trace = elliptical_slice(\n    trace, addr, mu, cov;\n    check=false, observations=EmptyChoiceMap())\n\nApply an elliptical slice sampling update to a given random choice with a multivariate normal prior.\n\nAlso takes the mean vector and covariance matrix of the prior.\n\nReference URL\n\n\n\n\n\n"
},

{
    "location": "ref/mcmc/#Gen.@pkern",
    "page": "Markov chain Monte Carlo",
    "title": "Gen.@pkern",
    "category": "macro",
    "text": "@pkern function k(trace, ..; check=false, observations=EmptyChoiceMap())\n    ..\n    return trace\nend\n\nDeclare a Julia function as a primitive stationary kernel.\n\nThe first argument of the function should be a trace, and the return value of the function should be a trace. There should be keyword arguments check and observations.\n\n\n\n\n\n"
},

{
    "location": "ref/mcmc/#Gen.@kern",
    "page": "Markov chain Monte Carlo",
    "title": "Gen.@kern",
    "category": "macro",
    "text": "@kern function k(trace, ..)\n    ..\nend\n\nConstruct a composite MCMC kernel.\n\nThe resulting object is a Julia function that is annotated as a composite MCMC kernel, and can be called as a Julia function or applied within other composite kernels.\n\n\n\n\n\n"
},

{
    "location": "ref/mcmc/#Gen.@rkern",
    "page": "Markov chain Monte Carlo",
    "title": "Gen.@rkern",
    "category": "macro",
    "text": "@rkern k1 : k2\n\nDeclare that two primitive stationary kernels are reversals of one another.\n\nThe two kernels must have the same argument type signatures.\n\n\n\n\n\n"
},

{
    "location": "ref/mcmc/#Gen.reversal",
    "page": "Markov chain Monte Carlo",
    "title": "Gen.reversal",
    "category": "function",
    "text": "k2 = reversal(k1)\n\nReturn the reversal kernel for a given kernel.\n\n\n\n\n\n"
},

{
    "location": "ref/mcmc/#Gen.involutive_mcmc",
    "page": "Markov chain Monte Carlo",
    "title": "Gen.involutive_mcmc",
    "category": "function",
    "text": "(new_trace, accepted) = involutive_mcmc(trace, proposal::GenerativeFunction, proposal_args::Tuple, involution; ..)\n\nAlias for the involutive form of metropolis_hastings.\n\n\n\n\n\n"
},

{
    "location": "ref/mcmc/#API-1",
    "page": "Markov chain Monte Carlo",
    "title": "API",
    "category": "section",
    "text": "metropolis_hastings\nmh\nmala\nhmc\nelliptical_slice\n@pkern\n@kern\n@rkern\nreversal\ninvolutive_mcmc"
},

{
    "location": "ref/map/#",
    "page": "MAP Optimization",
    "title": "MAP Optimization",
    "category": "page",
    "text": ""
},

{
    "location": "ref/map/#Gen.map_optimize",
    "page": "MAP Optimization",
    "title": "Gen.map_optimize",
    "category": "function",
    "text": "new_trace = map_optimize(trace, selection::Selection,\n    max_step_size=0.1, tau=0.5, min_step_size=1e-16, verbose=false)\n\nPerform backtracking gradient ascent to optimize the log probability of the trace over selected continuous choices.\n\nSelected random choices must have support on the entire real line.\n\n\n\n\n\n"
},

{
    "location": "ref/map/#MAP-Optimization-1",
    "page": "MAP Optimization",
    "title": "MAP Optimization",
    "category": "section",
    "text": "map_optimize"
},

{
    "location": "ref/pf/#",
    "page": "Particle Filtering",
    "title": "Particle Filtering",
    "category": "page",
    "text": ""
},

{
    "location": "ref/pf/#Gen.initialize_particle_filter",
    "page": "Particle Filtering",
    "title": "Gen.initialize_particle_filter",
    "category": "function",
    "text": "state = initialize_particle_filter(model::GenerativeFunction, model_args::Tuple,\n    observations::ChoiceMap proposal::GenerativeFunction, proposal_args::Tuple,\n    num_particles::Int)\n\nInitialize the state of a particle filter using a custom proposal for the initial latent state.\n\n\n\n\n\nstate = initialize_particle_filter(model::GenerativeFunction, model_args::Tuple,\n    observations::ChoiceMap, num_particles::Int)\n\nInitialize the state of a particle filter, using the default proposal for the initial latent state.\n\n\n\n\n\n"
},

{
    "location": "ref/pf/#Gen.particle_filter_step!",
    "page": "Particle Filtering",
    "title": "Gen.particle_filter_step!",
    "category": "function",
    "text": "particle_filter_step!(state::ParticleFilterState, new_args::Tuple, argdiffs,\n    observations::ChoiceMap, proposal::GenerativeFunction, proposal_args::Tuple)\n\nPerform a particle filter update, where the model arguments are adjusted, new observations are added, and some combination of a custom proposal and the model\'s internal proposal is used for proposing new latent state.  That is, for each particle,\n\nThe proposal function proposal is evaluated with arguments Tuple(t_old, proposal_args...) (where t_old is the old model trace), and produces its own trace (call it proposal_trace); and\nThe old model trace is replaced by a new model trace (call it t_new).\n\nThe choicemap of t_new satisfies the following conditions:\n\nget_choices(t_old) is a subset of get_choices(t_new);\nobservations is a subset of get_choices(t_new);\nget_choices(proposal_trace) is a subset of get_choices(t_new).\n\nHere, when we say one choicemap a is a \"subset\" of another choicemap b, we mean that all keys that occur in a also occur in b, and the values at those addresses are equal.\n\nIt is an error if no trace t_new satisfying the above conditions exists in the support of the model (with the new arguments). If such a trace exists, then the random choices not determined by the above requirements are sampled using the internal proposal.\n\n\n\n\n\nparticle_filter_step!(state::ParticleFilterState, new_args::Tuple, argdiffs,\n    observations::ChoiceMap)\n\nPerform a particle filter update, where the model arguments are adjusted, new observations are added, and the default proposal is used for new latent state.\n\n\n\n\n\n"
},

{
    "location": "ref/pf/#Gen.maybe_resample!",
    "page": "Particle Filtering",
    "title": "Gen.maybe_resample!",
    "category": "function",
    "text": "did_resample::Bool = maybe_resample!(state::ParticleFilterState;\n    ess_threshold::Float64=length(state.traces)/2, verbose=false)\n\nDo a resampling step if the effective sample size is below the given threshold. Return true if a resample thus occurred, false otherwise.\n\n\n\n\n\n"
},

{
    "location": "ref/pf/#Gen.log_ml_estimate",
    "page": "Particle Filtering",
    "title": "Gen.log_ml_estimate",
    "category": "function",
    "text": "estimate = log_ml_estimate(state::ParticleFilterState)\n\nReturn the particle filter\'s current estimate of the log marginal likelihood.\n\n\n\n\n\n"
},

{
    "location": "ref/pf/#Gen.get_traces",
    "page": "Particle Filtering",
    "title": "Gen.get_traces",
    "category": "function",
    "text": "traces = get_traces(state::ParticleFilterState)\n\nReturn the vector of traces in the current state, one for each particle.\n\n\n\n\n\n"
},

{
    "location": "ref/pf/#Gen.get_log_weights",
    "page": "Particle Filtering",
    "title": "Gen.get_log_weights",
    "category": "function",
    "text": "log_weights = get_log_weights(state::ParticleFilterState)\n\nReturn the vector of log weights for the current state, one for each particle.\n\nThe weights are not normalized, and are in log-space.\n\n\n\n\n\n"
},

{
    "location": "ref/pf/#Gen.sample_unweighted_traces",
    "page": "Particle Filtering",
    "title": "Gen.sample_unweighted_traces",
    "category": "function",
    "text": "traces::Vector = sample_unweighted_traces(state::ParticleFilterState, num_samples::Int)\n\nSample a vector of num_samples traces from the weighted collection of traces in the given particle filter state.\n\n\n\n\n\n"
},

{
    "location": "ref/pf/#Particle-Filtering-1",
    "page": "Particle Filtering",
    "title": "Particle Filtering",
    "category": "section",
    "text": "initialize_particle_filter\nparticle_filter_step!\nmaybe_resample!\nlog_ml_estimate\nget_traces\nget_log_weights\nsample_unweighted_traces"
},

{
    "location": "ref/vi/#",
    "page": "Variational Inference",
    "title": "Variational Inference",
    "category": "page",
    "text": ""
},

{
    "location": "ref/vi/#Variational-Inference-1",
    "page": "Variational Inference",
    "title": "Variational Inference",
    "category": "section",
    "text": "Variational inference involves optimizing the parameters of a variational family to maximize a lower bound on the marginal likelihood called the ELBO. In Gen, variational families are represented as generative functions, and variational inference typically involves optimizing the trainable parameters of generative functions."
},

{
    "location": "ref/vi/#Gen.black_box_vi!",
    "page": "Variational Inference",
    "title": "Gen.black_box_vi!",
    "category": "function",
    "text": "(elbo_estimate, traces, elbo_history) = black_box_vi!(\n    model::GenerativeFunction, model_args::Tuple,\n    [model_update::ParamUpdate,]\n    observations::ChoiceMap,\n    var_model::GenerativeFunction, var_model_args::Tuple,\n    var_model_update::ParamUpdate;\n    options...)\n\nFit the parameters of a variational model (var_model) to the posterior distribution implied by the given model and observations using stochastic gradient methods. Users may optionally specify a model_update to jointly update the parameters of model.\n\nAdditional arguments:\n\niters=1000: Number of iterations of gradient descent.\nsamples_per_iter=100: Number of samples from the variational and generative       model to accumulate gradients over before a single gradient step.\nverbose=false: If true, print information about the progress of fitting.\ncallback: Callback function that takes (iter, traces, elbo_estimate)       as input, where iter is the iteration number and traces are samples       from var_model for that iteration.\n\n\n\n\n\n"
},

{
    "location": "ref/vi/#Gen.black_box_vimco!",
    "page": "Variational Inference",
    "title": "Gen.black_box_vimco!",
    "category": "function",
    "text": "(iwelbo_estimate, traces, iwelbo_history) = black_box_vimco!(\n    model::GenerativeFunction, model_args::Tuple,\n    [model_update::ParamUpdate,]\n    observations::ChoiceMap,\n    var_model::GenerativeFunction, var_model_args::Tuple,\n    var_model_update::ParamUpdate,\n    grad_est_samples::Int; options...)\n\nFit the parameters of a variational model (var_model) to the posterior distribution implied by the given model and observations using stochastic gradient methods applied to the Variational Inference with Monte Carlo Objectives (VIMCO) lower bound on the marginal likelihood. Users may optionally specify a model_update to jointly update the parameters of model.\n\nAdditional arguments:\n\ngrad_est_samples::Int: Number of samples for the VIMCO gradient estimate.\niters=1000: Number of iterations of gradient descent.\nsamples_per_iter=100: Number of samples from the variational and generative       model to accumulate gradients over before a single gradient step.\ngeometric=true: Whether to use the geometric or arithmetric baselines       described in Variational Inference with Monte Carlo       Objectives\nverbose=false: If true, print information about the progress of fitting.\ncallback: Callback function that takes (iter, traces, elbo_estimate)       as input, where iter is the iteration number and traces are samples       from var_model for that iteration.\n\n\n\n\n\n"
},

{
    "location": "ref/vi/#Black-box-variational-inference-1",
    "page": "Variational Inference",
    "title": "Black box variational inference",
    "category": "section",
    "text": "There are two procedures in the inference library for performing black box variational inference. Each of these procedures can also train the model using stochastic gradient descent, as in a variational autoencoder.black_box_vi!\nblack_box_vimco!"
},

{
    "location": "ref/vi/#Reparametrization-trick-1",
    "page": "Variational Inference",
    "title": "Reparametrization trick",
    "category": "section",
    "text": "To use the reparametrization trick to reduce the variance of gradient estimators, users currently need to write two versions of their variational family, one that is reparametrized and one that is not. Gen does not currently include inference library support for this. We plan add add automated support for reparametrization and other variance reduction techniques in the future."
},

{
    "location": "ref/learning/#",
    "page": "Learning Generative Functions",
    "title": "Learning Generative Functions",
    "category": "page",
    "text": ""
},

{
    "location": "ref/learning/#Learning-Generative-Functions-1",
    "page": "Learning Generative Functions",
    "title": "Learning Generative Functions",
    "category": "section",
    "text": "Learning and inference are closely related concepts, and the distinction between the two is not always clear. Often, learning refers to inferring long-lived unobserved quantities that will be reused across many problem instances (like a dynamics model for an entity that we are trying to track), whereas inference refers to inferring shorter-lived quantities (like a specific trajectory of a specific entity). Learning is the way to use data to automatically generate models of the world, or to automatically fill in unknown parameters in hand-coded models. These resulting models are then used in various inference tasks.There are many variants of the learning task–we could be training the weights of a neural network, estimating a handful of parameters in a structured and hand-coded model, or we could be learning the structure or architecture of a model. Also, we could do Bayesian learning in which we seek a probability distribution on possible models, or we could seek just the best model, as measured by e.g. maximum likelihood. This section focuses on maximum likelihood learning of the Trainable parameters of a generative function. These are numerical quantities that are part of the generative function\'s state, with respect to which generative functions are able to report gradients of their (log) probability density function of their density function. Trainable parameters are different from random choices–random choices are per-trace and trainable parameters are a property of the generative function (which is associated with many traces). Also, unlike random choices, trainable parameters do not have a prior distribution.There are two settings in which we might learn these parameters using maximum likelihood. If our observed data contains values for all of the random choices made by the generative function, this is called learning from complete data, and is a relatively straightforward task. If our observed data is missing values for some random choices (either because the value happened to be missing, or because it was too expensive to acquire it, or because it is an inherently unmeasurable quantity), this is called learning from incomplete data, and is a substantially harder task. Gen provides programming primitives and design patterns for both tasks. In both cases, the models we are learning can be either generative or discriminative."
},

{
    "location": "ref/learning/#Learning-from-Complete-Data-1",
    "page": "Learning Generative Functions",
    "title": "Learning from Complete Data",
    "category": "section",
    "text": "This section discusses maximizing the log likelihood of observed data over the space of trainable parameters, when all of the random variables are observed. In Gen, the likelihood of complete data is simply the joint probability (density) of a trace, and maximum likelihood with complete data amounts to maximizing the sum of log joint probabilities of a collection of traces t_i for i = 1ldots N with respect to the trainable parameters of the generative function, which are denoted theta.max_theta sum_i=1^N log p(t_i x theta)For example, here is a simple generative model that we might want to learn:@gen function model()\n    @param x_mu::Float64\n    @param a::Float64\n    @param b::Float64\n    x = @trace(normal(x_mu, 1.), :x)\n    @trace(normal(a * x + b, 1.), :y)\nendThere are three components to theta for this generative function: (x_mu, a, b).Note that maximum likelihood can be used to learn generative and discriminative models, but for discriminative models, the arguments to the generative function will be different for each training example:max_theta sum_i=1^N log p(t_i x_i theta)Here is a minimal discriminative model:@gen function disc_model(x::Float64)\n    @param a::Float64\n    @param b::Float64\n    @trace(normal(a * x + b, 1.), :y)\nendLet\'s suppose we are training the generative model. The first step is to initialize the values of the trainable parameters, which for generative functions constructed using the built-in modeling languages, we do with init_param!:init_param!(model, :a, 0.)\ninit_param!(model, :b, 0.)Each trace in the collection contains the observed data from an independent draw from our model. We can populate each trace with its observed data using generate:traces = []\nfor observations in data\n    trace, = generate(model, model_args, observations)\n    push!(traces, trace)\nendFor the complete data case, we assume that all random choices in the model are constrained by the observations choice map (we will analyze the case when not all random choices are constrained in the next section). We can evaluate the objective function by summing the result of get_score over our collection of traces:objective = sum([get_score(trace) for trace in traces])We can compute the gradient of this objective function with respect to the trainable parameters using accumulate_param_gradients!:for trace in traces\n    accumulate_param_gradients!(trace)\nendFinally, we can construct and gradient-based update with ParamUpdate and apply it with apply!. We can put this all together into a function:function train_model(data::Vector{ChoiceMap})\n    init_param!(model, :theta, 0.1)\n    traces = []\n    for observations in data\n        trace, = generate(model, model_args, observations)\n        push!(traces, trace)\n    end\n    update = ParamUpdate(FixedStepSizeGradientDescent(0.001), model)\n    for iter=1:max_iter\n        objective = sum([get_score(trace) for trace in traces])\n        println(\"objective: $objective\")\n        for trace in traces\n            accumulate_param_gradients!(trace)\n        end\n        apply!(update)\n    end\nendNote that using the same primitives (generate and accumulate_param_gradients!), you can compose various more sophisticated learning algorithms involving e.g. stochastic gradient descent and minibatches, and more sophisticated stochastic gradient optimizers like ADAM. For example, train! trains a generative function from complete data with minibatches."
},

{
    "location": "ref/learning/#Learning-from-Incomplete-Data-1",
    "page": "Learning Generative Functions",
    "title": "Learning from Incomplete Data",
    "category": "section",
    "text": "When there are random variables in our model whose value is not observed in our data set, then doing maximum learning is significantly more difficult. Specifically, maximum likelihood is aiming to maximize the marginal likelihood of the observed data, which is an integral or sum over the values of the unobserved random variables. Let\'s denote the observed variables as y and the hidden variables as z:sum_i=1^N log p(y_i x theta) = sum_i=1^N log left( sum_z_i p(z_i y_i x theta)right)It is often intractable to evaluate this quantity for specific values of the parameters, let alone maximize it. Most techniques for learning models from incomplete data, from the EM algorithm to variational autoencoders address this problem by starting with some initial theta = theta_0 and iterating between two steps:Doing inference about the hidden variables z_i given the observed variables y_i, for the model with the current values of theta, which produces some completions of the hidden variables z_i or some representation of the posterior distribution on these hidden variables. This step does not update the parameters theta.\nOptimize the parameters theta to maximize the data of the complete log likelihood, as in the setting of complete data. This step does not involve inference about the hidden variables z_i.Various algorithms can be understood as examples of this general pattern, although they differ in several details including (i) how they represent the results of inferences, (ii) how they perform the inference step, (iii) whether they try to solve each of the inference and parameter-optimization problems incrementally or not, and (iv) their formal theoretical justification and analysis:[Expectation maximization (EM) [1], including incremental variants [2]\nMonte Carlo EM [3] and online variants [4]\nVariational EM\nThe wake-sleep algorithm [5] and reweighted wake-sleep algorithms [6]\nVariational autoencoders [7]In Gen, the results of inference are typically represented as a collection of traces of the model, which include values for the latent variables. The section Learning from Complete Data describes how to perform the parameter update step given a collection of such traces. In the remainder of this section, we describe various learning algorithms, organized by the inference approach they take to obtain traces."
},

{
    "location": "ref/learning/#Monte-Carlo-EM-1",
    "page": "Learning Generative Functions",
    "title": "Monte Carlo EM",
    "category": "section",
    "text": "Monte Carlo EM is a broad class of algorithms that use Monte Carlo sampling within the inference step to generate the set of traces that is used for the learning step. There are many variants possible, based on which Monte Carlo inference algorithm is used. For example:function train_model(data::Vector{ChoiceMap})\n    init_param!(model, :theta, 0.1)\n    update = ParamUpdate(FixedStepSizeGradientDescent(0.001), model)\n    for iter=1:max_iter\n        traces = do_monte_carlo_inference(data)\n        for trace in traces\n            accumulate_param_gradients!(trace)\n        end\n        apply!(update)\n    end\nend\n\nfunction do_monte_carlo_inference(data)\n    num_traces = 1000\n    (traces, log_weights, _) = importance_sampling(model, (), data, num_samples)\n    weights = exp.(log_weights)\n    [traces[categorical(weights)] for _=1:num_samples]\nendNote that it is also possible to use a weighted collection of traces directly without resampling:function train_model(data::Vector{ChoiceMap})\n    init_param!(model, :theta, 0.1)\n    update = ParamUpdate(FixedStepSizeGradientDescent(0.001), model)\n    for iter=1:max_iter\n        traces, weights = do_monte_carlo_inference_with_weights(data)\n        for (trace, weight) in zip(traces, weights)\n            accumulate_param_gradients!(trace, nothing, weight)\n        end\n        apply!(update)\n    end\nendMCMC and other algorithms can be used for inference as well."
},

{
    "location": "ref/learning/#Online-Monte-Carlo-EM-1",
    "page": "Learning Generative Functions",
    "title": "Online Monte Carlo EM",
    "category": "section",
    "text": "The Monte Carlo EM example performed inference from scratch within each iteration. However, if the change tothe parameters during each iteration is small, it is likely that the traces from the previous iteration can be reused. There are various ways of reusing traces: We can use the traces obtained for the previous traces to initialize MCMC for the new parameters. We can reweight the traces based on the change to their importance weights [4]."
},

{
    "location": "ref/learning/#Wake-sleep-algorithm-1",
    "page": "Learning Generative Functions",
    "title": "Wake-sleep algorithm",
    "category": "section",
    "text": "The wake-sleep algorithm [5] is an approach to training generative models that uses an inference network, a neural network that takes in the values of observed random variables and returns parameters of a probability distribution on latent variables. We call the conditional probability distribution on the latent variables, given the observed variables, the inference model. In Gen, both the generative model and the inference model are represented as generative functions. The wake-sleep algorithm trains the inference model as it trains the generative model. At each iteration, during the wake phase, the generative model is trained on complete traces generated by running the current version of the inference on the observed data. At each iteration, during the sleep phase, the inference model is trained on data generated by simulating from the current generative model. The lecture! or lecture_batched! methods can be used for the sleep phase training."
},

{
    "location": "ref/learning/#Reweighted-wake-sleep-algorithm-1",
    "page": "Learning Generative Functions",
    "title": "Reweighted wake-sleep algorithm",
    "category": "section",
    "text": "The reweighted wake-sleep algorithm [6] is an extension of the wake-sleep algorithm, where during the wake phase, for each observation, a collection of latent completions are taken by simulating from the inference model multiple times. Then, each of these is weighted by an importance weight. This extension can be implemented with importance_sampling."
},

{
    "location": "ref/learning/#Variational-inference-1",
    "page": "Learning Generative Functions",
    "title": "Variational inference",
    "category": "section",
    "text": "Variational inference can be used to for the inference step. Here, the parameters of the variational approximation, represented as a generative function, are fit to the posterior during the inference step. black_box_vi! or black_box_vimco! can be used to fit the variational approximation. Then, the traces of the model can be obtained by simulating from the variational approximation and merging the resulting choice maps with the observed data."
},

{
    "location": "ref/learning/#Amortized-variational-inference-(VAEs)-1",
    "page": "Learning Generative Functions",
    "title": "Amortized variational inference (VAEs)",
    "category": "section",
    "text": "Instead of fitting the variational approximation from scratch for each observation, it is possible to fit an inference model instead, that takes as input the observation, and generates a distribution on latent variables as output (as in the wake sleep algorithm). When we train the variational approximation by minimizing the evidence lower bound (ELBO) this is called amortized variational inference. Variational autencoders are an example. It is possible to perform amortized variational inference using black_box_vi or black_box_vimco!."
},

{
    "location": "ref/learning/#References-1",
    "page": "Learning Generative Functions",
    "title": "References",
    "category": "section",
    "text": "[1] Dempster, Arthur P., Nan M. Laird, and Donald B. Rubin. \"Maximum likelihood from incomplete data via the EM algorithm.\" Journal of the Royal Statistical Society: Series B (Methodological) 39.1 (1977): 1-22. Link[2] Neal, Radford M., and Geoffrey E. Hinton. \"A view of the EM algorithm that justifies incremental, sparse, and other variants.\" Learning in graphical models. Springer, Dordrecht, 1998. 355-368. Link[3] Wei, Greg CG, and Martin A. Tanner. \"A Monte Carlo implementation of the EM algorithm and the poor man\'s data augmentation algorithms.\" Journal of the American statistical Association 85.411 (1990): 699-704. Link[4] Levine, Richard A., and George Casella. \"Implementations of the Monte Carlo EM algorithm.\" Journal of Computational and Graphical Statistics 10.3 (2001): 422-439. Link[5] Hinton, Geoffrey E., et al. \"The\" wake-sleep\" algorithm for unsupervised neural networks.\" Science 268.5214 (1995): 1158-1161. Link[6] Jorg Bornschein and Yoshua Bengio. Reweighted wake sleep. ICLR 2015. Link[7] Diederik P. Kingma, Max Welling: Auto-Encoding Variational Bayes. ICLR 2014 Link"
},

{
    "location": "ref/learning/#Gen.lecture!",
    "page": "Learning Generative Functions",
    "title": "Gen.lecture!",
    "category": "function",
    "text": "score = lecture!(\n    p::GenerativeFunction, p_args::Tuple,\n    q::GenerativeFunction, get_q_args::Function)\n\nSimulate a trace of p representing a training example, and use to update the gradients of the trainable parameters of q.\n\nUsed for training q via maximum expected conditional likelihood. Random choices will be mapped from p to q based on their address. getqargs maps a trace of p to an argument tuple of q. score is the conditional log likelihood (or an unbiased estimate of a lower bound on it, if not all of q\'s random choices are constrained, or if q uses non-addressable randomness).\n\n\n\n\n\n"
},

{
    "location": "ref/learning/#Gen.lecture_batched!",
    "page": "Learning Generative Functions",
    "title": "Gen.lecture_batched!",
    "category": "function",
    "text": "score = lecture_batched!(\n    p::GenerativeFunction, p_args::Tuple,\n    q::GenerativeFunction, get_q_args::Function)\n\nSimulate a batch of traces of p representing training samples, and use them to update the gradients of the trainable parameters of q.\n\nLike lecture! but q is batched, and must make random choices for training sample i under hierarchical address namespace i::Int (e.g. i => :z). getqargs maps a vector of traces of p to an argument tuple of q.\n\n\n\n\n\n"
},

{
    "location": "ref/learning/#Gen.train!",
    "page": "Learning Generative Functions",
    "title": "Gen.train!",
    "category": "function",
    "text": "train!(gen_fn::GenerativeFunction, data_generator::Function,\n       update::ParamUpdate,\n       num_epoch, epoch_size, num_minibatch, minibatch_size; verbose::Bool=false)\n\nTrain the given generative function to maximize the expected conditional log probability (density) that gen_fn generates the assignment constraints given inputs, where the expectation is taken under the output distribution of data_generator.\n\nThe function data_generator is a function of no arguments that returns a tuple (inputs, constraints) where inputs is a Tuple of inputs (arguments) to gen_fn, and constraints is an ChoiceMap.\n\nconf configures the optimization algorithm used.\n\nparam_lists is a map from generative function to lists of its parameters. This is equivalent to minimizing the expected KL divergence from the conditional distribution constraints | inputs of the data generator to the distribution represented by the generative function, where the expectation is taken under the marginal distribution on inputs determined by the data generator.\n\n\n\n\n\n"
},

{
    "location": "ref/learning/#API-1",
    "page": "Learning Generative Functions",
    "title": "API",
    "category": "section",
    "text": "lecture!\nlecture_batched!\ntrain!"
},

{
    "location": "ref/internals/parameter_optimization/#",
    "page": "Optimizing Trainable Parameters",
    "title": "Optimizing Trainable Parameters",
    "category": "page",
    "text": ""
},

{
    "location": "ref/internals/parameter_optimization/#Gen.init_update_state",
    "page": "Optimizing Trainable Parameters",
    "title": "Gen.init_update_state",
    "category": "function",
    "text": "state = init_update_state(conf, gen_fn::GenerativeFunction, param_list::Vector)\n\nGet the initial state for a parameter update to the given parameters of the given generative function.\n\nparam_list is a vector of references to parameters of gen_fn. conf configures the update.\n\n\n\n\n\n"
},

{
    "location": "ref/internals/parameter_optimization/#Gen.apply_update!",
    "page": "Optimizing Trainable Parameters",
    "title": "Gen.apply_update!",
    "category": "function",
    "text": "apply_update!(state)\n\nApply one parameter update, mutating the values of the trainable parameters, and possibly also the given state.\n\n\n\n\n\n"
},

{
    "location": "ref/internals/parameter_optimization/#optimizing-internal-1",
    "page": "Optimizing Trainable Parameters",
    "title": "Optimizing Trainable Parameters",
    "category": "section",
    "text": "To add support for a new type of gradient-based parameter update, create a new type with the following methods defined for the types of generative functions that are to be supported.Gen.init_update_state\nGen.apply_update!"
},

{
    "location": "ref/internals/language_implementation/#",
    "page": "Modeling Language Implementation",
    "title": "Modeling Language Implementation",
    "category": "page",
    "text": ""
},

{
    "location": "ref/internals/language_implementation/#language-implementation-1",
    "page": "Modeling Language Implementation",
    "title": "Modeling Language Implementation",
    "category": "section",
    "text": ""
},

{
    "location": "ref/internals/language_implementation/#Parsing-@gen-functions-1",
    "page": "Modeling Language Implementation",
    "title": "Parsing @gen functions",
    "category": "section",
    "text": "Gen\'s built-in modeling languages are designed to preserve Julia\'s syntax as far as possible, apart from the Tilde syntax for calling generative functions, and the restrictions imposed on the Static Modeling Language. In order to preserve that syntax, including the use of non-Gen macros within @gen functions, we relegate as much of the parsing of @gen functions as possible to Julia\'s macro-expander and parser.In particular, we adopt an implementation strategy that enforces a separation between the surface syntax associated with Gen-specific macros (i.e., @trace and @param) and their corresponding implementations, which differ across the Dynamic Modeling Language (DML) and the Static Modeling Language (SML). We do this by introducing the custom expressions Expr(:gentrace, call, addr) and Expr(:genparam, name, type), which serve as intermediate representations in the macro-expanded abstract syntax tree.Each modeling language can then handle these custom expressions in their own manner, either by parsing them to nodes in the Static Computation Graph (for the SML), or by substituting them with their implementations (for the DML). This effectively allows the SML and DML to have separate implementations of @trace and @param.For clarity, below is a procedural description of how the @gen macro processes Julia function syntax:macroexpand the entire function body with respect to the calling module. This expands any (properly-scoped) @trace calls to Expr(:gentrace, ...) expressions, and any (properly-scoped) @param calls to Expr(:genparam, ...) expressions, while also expanding non-Gen macros.\nDesugar any tilde expressions x ~ gen_fn(), including those that may have been generated by macros, to Expr(:gentrace, ...) expressions.\nPass the macro-expanded and de-sugared function body on to make_static_gen_function or make_dynamic_gen_function accordingly.\nFor static @gen functions, match :gentrace expressions when adding address nodes to the static computation graph, and match :genparam expressions when adding parameter nodes to the static computation graph. A StaticIRGenerativeFunction is then compiled from the static computation graph.\nFor dynamic @gen functions, rewrite any :gentrace expression with its implementation dynamic_trace_impl, and rewrite any :genparam expression with its implementation dynamic_param_impl. The rewritten syntax tree is then evaluated as a standard Julia function, which serves as the implementation of the constructed DynamicDSLFunction."
},

]}
