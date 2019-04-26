var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#Gen-1",
    "page": "Home",
    "title": "Gen",
    "category": "section",
    "text": "A General-Purpose Probabilistic Programming System with Programmable InferencePages = [\n    \"getting_started.md\",\n    \"tutorials.md\",\n    \"guide.md\",\n]\nDepth = 2ReferencePages = [\n    \"ref/modeling.md\",\n    \"ref/combinators.md\",\n    \"ref/assignments.md\",\n    \"ref/selections.md\",\n    \"ref/parameter_optimization.md\",\n    \"ref/inference.md\",\n    \"ref/gfi.md\",\n    \"ref/distributions.md\"\n]\nDepth = 2"
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
    "text": "First, obtain Julia 1.0 or later, available here.The Gen package can be installed with the Julia package manager. From the Julia REPL, type ] to enter the Pkg REPL mode and then run:pkg> add https://github.com/probcomp/GenTo test the installation, run the example in the next section, or run the tests with:using Pkg; Pkg.test(\"Gen\")"
},

{
    "location": "getting_started/#Example-1",
    "page": "Getting Started",
    "title": "Example",
    "category": "section",
    "text": "Let\'s write a short Gen program that does Bayesian linear regression: given a set of points in the (x, y) plane, we want to find a line that fits them well.There are three main components to a typical Gen program.First, we define a generative model: a Julia function, extended with some extra syntax, that, conceptually, simulates a fake dataset. The model below samples slope and intercept parameters, and then for each of the x-coordinates that it accepts as input, samples a corresponding y-coordinate. We name the random choices we make with @trace, so we can refer to them in our inference program.using Gen\n\n@gen function my_model(xs::Vector{Float64})\n    slope = @trace(normal(0, 2), :slope)\n    intercept = @trace(normal(0, 10), :intercept)\n    for (i, x) in enumerate(xs)\n        @trace(normal(slope * x + intercept, 1), \"y-$i\")\n    end\nendSecond, we write an inference program that implements an algorithm for manipulating the execution traces of the model. Inference programs are regular Julia code, and make use of Gen\'s standard inference library.The inference program below takes in a data set, and runs an iterative MCMC algorithm to fit slope and intercept parameters:function my_inference_program(xs::Vector{Float64}, ys::Vector{Float64}, num_iters::Int)\n    # Create a set of constraints fixing the \n    # y coordinates to the observed y values\n    constraints = choicemap()\n    for (i, y) in enumerate(ys)\n        constraints[\"y-$i\"] = y\n    end\n    \n    # Run the model, constrained by `constraints`,\n    # to get an initial execution trace\n    (trace, _) = generate(my_model, (xs,), constraints)\n    \n    # Iteratively update the slope then the intercept,\n    # using Gen\'s metropolis_hastings operator.\n    for iter=1:num_iters\n        (trace, _) = metropolis_hastings(trace, select(:slope))\n        (trace, _) = metropolis_hastings(trace, select(:intercept))\n    end\n    \n    # From the final trace, read out the slope and\n    # the intercept.\n    choices = get_choices(trace)\n    return (choices[:slope], choices[:intercept])\nendFinally, we run the inference program on some data, and get the results:xs = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]\nys = [8.23, 5.87, 3.99, 2.59, 0.23, -0.66, -3.53, -6.91, -7.24, -9.90]\n(slope, intercept) = my_inference_program(xs, ys, 1000)\nprintln(\"slope: $slope, intercept: $slope\")"
},

{
    "location": "getting_started/#Visualization-Framework-1",
    "page": "Getting Started",
    "title": "Visualization Framework",
    "category": "section",
    "text": "Because inference programs are regular Julia code, users can use whatever visualization or plotting libraries from the Julia ecosystem that they want. However, we have paired Gen with the GenViz package, which is specialized for visualizing the output and operation of inference algorithms written in Gen.An example demonstrating the use of GenViz for this example linear regression problem is available in the gen-quickstart repository. The code there is mostly the same as above, with a few small changes to incorporate an animated visualization of the inference process:It starts a visualization server and initializes a visualization before performing inference:# Start a visualization server on port 8000\nserver = VizServer(8000)\n\n# Initialize a visualization with some parameters\nviz = Viz(server, joinpath(@__DIR__, \"vue/dist\"), Dict(\"xs\" => xs, \"ys\" => ys, \"num\" => length(xs), \"xlim\" => [minimum(xs), maximum(xs)], \"ylim\" => [minimum(ys), maximum(ys)]))\n\n# Open the visualization in a browser\nopenInBrowser(viz)The \"vue/dist\" is a path to a custom trace renderer that draws the (x, y) points and the line represented by a trace; see the GenViz documentation for more details. The code for the renderer is here.It passes the visualization object into the inference program.(slope, intercept) = my_inference_program(xs, ys, 1000000, viz)In the inference program, it puts the current trace into the visualization at each iteration:for iter=1:num_iters\n    putTrace!(viz, 1, trace_to_dict(trace))\n    (trace, _) = metropolis_hastings(trace, select(:slope))\n    (trace, _) = metropolis_hastings(trace, select(:intercept))\nend"
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
    "text": "See Gen Quickstart repository for tutorials and case studiesAdditional examples are available in the examples/ directory of the Gen repository."
},

{
    "location": "ref/gfi/#",
    "page": "Generative Functions",
    "title": "Generative Functions",
    "category": "page",
    "text": ""
},

{
    "location": "ref/gfi/#Gen.GenerativeFunction",
    "page": "Generative Functions",
    "title": "Gen.GenerativeFunction",
    "category": "type",
    "text": "GenerativeFunction{T,U <: Trace}\n\nAbstract type for a generative function with return value type T and trace type U.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Generative-Functions-1",
    "page": "Generative Functions",
    "title": "Generative Functions",
    "category": "section",
    "text": "One of the core abstractions in Gen is the generative function. Generative functions are used to represent a variety of different types of probabilistic computations including generative models, inference models, custom proposal distributions, and variational approximations.Generative functions are represented by the following abstact type:GenerativeFunctionThere are various kinds of generative functions, which are represented by concrete subtypes of GenerativeFunction. For example, the Built-in Modeling Language allows generative functions to be constructed using Julia function definition syntax:@gen function foo(a, b)\n    if @trace(bernoulli(0.5), :z)\n        return a + b + 1\n    else\n        return a + b\n    end\nendGenerative functions behave like Julia functions in some respects. For example, we can call a generative function foo on arguments and get an output value using regular Julia call syntax:>julia foo(2, 4)\n7However, generative functions are distinct from Julia functions because they support additional behaviors, described in the remainder of this section."
},

{
    "location": "ref/gfi/#Mathematical-definition-1",
    "page": "Generative Functions",
    "title": "Mathematical definition",
    "category": "section",
    "text": "Generative functions represent computations that accept some arguments, may use randomness internally, return an output, and cannot mutate externally observable state. We represent the randomness used during an execution of a generative function as a map from unique addresses to values, denoted t  A to V where A is an address set and V is a set of possible values that random choices can take. In this section, we assume that random choices are discrete to simplify notation. We say that two random choice maps t and s agree if they assign the same value for any address that is in both of their domains.Generative functions may also use untraced randomness, which is not included in the map t. However, the state of untraced random choices is maintained by the trace internally. We denote untraced randomness by r. Untraced randomness is useful for example, when calling black box Julia code that implements a randomized algorithm.The observable behavior of every generative function is defined by the following mathematical objects:"
},

{
    "location": "ref/gfi/#.-Input-type-1",
    "page": "Generative Functions",
    "title": "1. Input type",
    "category": "section",
    "text": "The set of valid argument tuples to the function, denoted X."
},

{
    "location": "ref/gfi/#.-Probability-distribution-family-1",
    "page": "Generative Functions",
    "title": "2. Probability distribution family",
    "category": "section",
    "text": "A family of probability distributions p(t r x) on maps t from random choice addresses to their values, and untraced randomness r, indexed by arguments x, for all x in X. Note that the distribution must be normalized:sum_t r p(t r x) = 1  mboxfor all  x in XThis corresponds to a requirement that the function terminate with probabability 1 for all valid arguments. We use p(t x) to denote the marginal distribution on the map t:p(t x) = sum_r p(t r x)And we denote the conditional distribution on untraced randomness r, given the map t, as:p(r x t) = p(t r x)  p(t x)"
},

{
    "location": "ref/gfi/#.-Return-value-function-1",
    "page": "Generative Functions",
    "title": "3. Return value function",
    "category": "section",
    "text": "A (deterministic) function f that maps the tuple (x t) of the arguments and the random choice map to the return value of the function (which we denote by y). Note that the return value cannot depend on the untraced randomness."
},

{
    "location": "ref/gfi/#.-Internal-proposal-distribution-family-1",
    "page": "Generative Functions",
    "title": "4. Internal proposal distribution family",
    "category": "section",
    "text": "A family of probability distributions q(t x u) on maps t from random choice addresses to their values, indexed by tuples (x u) where u is a map from random choice addresses to values, and where x are the arguments to the function. It must satisfy the following conditions:sum_t q(t x u) = 1  mboxfor all  x in X up(t x)  0 mbox if and only if  q(t x u)  0 mbox for all  u mbox where  u mbox and  t mbox agree q(t x u)  0 mbox implies that  u mbox and  t mbox agree There is also a family of probability distributions q(r x t) on untraced randomness, that satisfies:q(r x t)  0 mbox if and only if  p(r x t)  0"
},

{
    "location": "ref/gfi/#Gen.simulate",
    "page": "Generative Functions",
    "title": "Gen.simulate",
    "category": "function",
    "text": "trace = simulate(gen_fn, args)\n\nExecute the generative function and return the trace.\n\nGiven arguments (args), sample t sim p(cdot x) and r sim p(cdot x t), and return a trace with choice map t.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.generate",
    "page": "Generative Functions",
    "title": "Gen.generate",
    "category": "function",
    "text": "(trace::U, weight) = generate(gen_fn::GenerativeFunction{T,U}, args::Tuple)\n\nReturn a trace of a generative function.\n\n(trace::U, weight) = generate(gen_fn::GenerativeFunction{T,U}, args::Tuple,\n                                constraints::ChoiceMap)\n\nReturn a trace of a generative function that is consistent with the given constraints on the random choices.\n\nGiven arguments x (args) and assignment u (constraints) (which is empty for the first form), sample t sim q(cdot u x) and r sim q(cdot x t), and return the trace (x t r) (trace).  Also return the weight (weight):\n\nlog fracp(t r x)q(t u x) q(r x t)\n\nExample without constraints:\n\n(trace, weight) = generate(foo, (2, 4))\n\nExample with constraint that address :z takes value true.\n\n(trace, weight) = generate(foo, (2, 4), choicemap((:z, true))\n\n\n\n\n\n"
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
    "text": "get_score(trace)\n\nReturn P(r t x)  Q(r tx t). When there is no non-addressed randomness, this simplifies to the log probability `P(t x).\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.get_gen_fn",
    "page": "Generative Functions",
    "title": "Gen.get_gen_fn",
    "category": "function",
    "text": "gen_fn::GenerativeFunction = get_gen_fn(trace)\n\nReturn the generative function that produced the given trace.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Traces-1",
    "page": "Generative Functions",
    "title": "Traces",
    "category": "section",
    "text": "An execution trace (or just trace) is a record of an execution of a generative function. There is no abstract type representing all traces. Different concrete types of generative functions use different data structures and different Jula types for their traces. The trace type that a generative function uses is the second type parameter of the GenerativeFunction abstract type.A generative function can be executed and traced using:simulateAn traced execution that satisfies constraints on the choice map can be generated using:generateThe trace contains various information about the execution, including:The arguments to the generative function:get_argsThe return value of the generative function:get_retvalThe map t from addresses of random choices to their values:get_choicesThe log probability that the random choices took the values they did:get_scoreA reference to the generative function that was executed:get_gen_fn"
},

{
    "location": "ref/gfi/#Trace-update-methods-1",
    "page": "Generative Functions",
    "title": "Trace update methods",
    "category": "section",
    "text": "It is often important to update or adjust the trace of a generative function. In Gen, traces are persistent data structures, meaning they can be treated as immutable values. There are several methods that take a trace of a generative function as input and return a new trace of the generative function based on adjustments to the execution history of the function. We will illustrate these methods using the following generative function:@gen function foo()\n    val = @trace(bernoulli(0.3), :a)\n    if @trace(bernoulli(0.4), :b)\n        val = @trace(bernoulli(0.6), :c) && val\n    else\n        val = @trace(bernoulli(0.1), :d) && val\n    end\n    val = @trace(bernoulli(0.7), :e) && val\n    return val\nendSuppose we have a trace (trace) with initial choices:│\n├── :a : false\n│\n├── :b : true\n│\n├── :c : false\n│\n└── :e : trueNote that address :d is not present because the branch in which :d is sampled was not taken because random choice :b had value true."
},

{
    "location": "ref/gfi/#Gen.update",
    "page": "Generative Functions",
    "title": "Gen.update",
    "category": "function",
    "text": "(new_trace, weight, retdiff, discard) = update(trace, args::Tuple, argdiffs::Tuple,\n                                               constraints::ChoiceMap)\n\nUpdate a trace by changing the arguments and/or providing new values for some existing random choice(s) and values for any newly introduced random choice(s).\n\nGiven a previous trace (x t r) (trace), new arguments x (args), and a map u (constraints), return a new trace (x t r) (new_trace) that is consistent with u.  The values of choices in t are deterministically copied either from t or from u (with u taking precedence).  All choices in u must appear in t.  Also return an assignment v (discard) containing the choices in t that were overwritten by values from u, and any choices in t whose address does not appear in t. The new non-addressed randomness is sampled from r sim q(cdot x t). Also return a weight (weight):\n\nlog fracp(r t x) q(r x t)p(r t x) q(r x t)\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Update-1",
    "page": "Generative Functions",
    "title": "Update",
    "category": "section",
    "text": "updateSuppose we run update on the example trace, with the following constraints:│\n├── :b : false\n│\n└── :d : trueconstraints = choicemap((:b, false), (:d, true))\n(new_trace, w, _, discard) = update(trace, (), noargdiff, constraints)Then get_choices(new_trace) will be:│\n├── :a : false\n│\n├── :b : false\n│\n├── :d : true\n│\n└── :e : trueand discard will be:│\n├── :b : true\n│\n└── :c : falseNote that the discard contains both the previous values of addresses that were overwritten, and the values for addresses that were in the previous trace but are no longer in the new trace. The weight (w) is computed as:p(t x) = 07  04  04  07 = 00784\np(t x) = 07  06  01  07 = 00294\nw = log p(t x)p(t x) = log 0029400784 = log 0375"
},

{
    "location": "ref/gfi/#Gen.regenerate",
    "page": "Generative Functions",
    "title": "Gen.regenerate",
    "category": "function",
    "text": "(new_trace, weight, retdiff) = regenerate(trace, args::Tuple, argdiffs::Tuple,\n                                          selection::AddressSet)\n\nUpdate a trace by changing the arguments and/or randomly sampling new values for selected random choices using the internal proposal distribution family.\n\nGiven a previous trace (x t r) (trace), new arguments x (args), and a set of addresses A (selection), return a new trace (x t) (new_trace) such that t agrees with t on all addresses not in A (t and t may have different sets of addresses).  Let u denote the restriction of t to the complement of A.  Sample t sim Q(cdot u x) and sample r sim Q(cdot x t). Return the new trace (x t r) (new_trace) and the weight (weight):\n\nlog fracp(r t x) q(t u x) q(r x t)p(r t x) q(t u x) q(r x t)\n\nwhere u is the restriction of t to the complement of A.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Regenerate-1",
    "page": "Generative Functions",
    "title": "Regenerate",
    "category": "section",
    "text": "regenerateSuppose we run regenerate on the example trace, with selection :a and :b:(new_trace, w, _) = regenerate(trace, (), noargdiff, select(:a, :b))Then, a new value for :a will be sampled from bernoulli(0.3), and a new value for :b will be sampled from bernoulli(0.4). If the new value for :b is true, then the previous value for :c (false) will be retained. If the new value for :b is false, then a new value for :d will be sampled from bernoulli(0.7). The previous value for :c will always be retained. Suppose the new value for :a is true, and the new value for :b is true. Then get_choices(new_trace) will be:│\n├── :a : true\n│\n├── :b : true \n│\n├── :c : false\n│\n└── :e : trueThe weight (w) is log 1 = 0."
},

{
    "location": "ref/gfi/#Gen.extend",
    "page": "Generative Functions",
    "title": "Gen.extend",
    "category": "function",
    "text": "(new_trace, weight, retdiff) = extend(trace, args::Tuple, argdiffs::Tuple,\n                                      constraints::ChoiceMap)\n\nExtend a trace with new random choices by changing the arguments.\n\nGiven a previous trace (x t r) (trace), new arguments x (args), and an assignment u (choices) that shares no addresses with t, return a new trace (x t r) (new_trace) such that t agrees with t on all addresses in t and t agrees with u on all addresses in u. Sample t sim Q(cdot t + u x) and r sim Q(cdot t x). Also return the weight (weight):\n\nlog fracp(r t x) q(r x t)p(r t x) q(t t + u x) q(r x t)\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Extend-1",
    "page": "Generative Functions",
    "title": "Extend",
    "category": "section",
    "text": "extend"
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
    "text": "In addition to the input trace, and other arguments that indicate how to adjust the trace, each of these methods also accepts an args argument and an argdiffs argument, both of which are tuples. The args argument contains the new arguments to the generative function, which may differ from the previous arguments to the generative function (which can be retrieved by applying get_args to the previous trace). In many cases, the adjustment to the execution specified by the other arguments to these methods is \'small\' and only effects certain parts of the computation. Therefore, it is often possible to generate the new trace and the appropriate log probability ratios required for these methods without revisiting every state of the computation of the generative function. To enable this, the argdiffs argument provides additional information about the difference between each of the previous arguments to the generative function, and its new argument value. This argdiff information permits the implementation of the update method to avoid inspecting the entire argument data structure to identify which parts were updated. Note that the correctness of the argdiff is in general not verified by Gen–-passing incorrect argdiff information may result in incorrect behavior.The trace update methods for all generative functions above should accept at least the following types of argdiffs:NoChange\nUnknownChangeGenerative functions may also be able to process more specialized diff data types for each of their arguments, that allow more precise information about the different to be supplied."
},

{
    "location": "ref/gfi/#Retdiffs-1",
    "page": "Generative Functions",
    "title": "Retdiffs",
    "category": "section",
    "text": "To enable generative functions that invoke other functions to efficiently make use of incremental computation, the trace update methods of generative functions also return a retdiff value, which provides information about the difference in the return value of the previous trace an the return value of the new trace."
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
    "text": "arg_grads = accumulate_param_gradients!(trace, retgrad, scaler=1.)\n\nIncrement gradient accumulators for parameters by the gradient of the log-probability of the trace, optionally scaled, and return the gradient with respect to the arguments (not scaled).\n\nGiven a previous trace (x t) (trace) and a gradient with respect to the return value _y J (retgrad), return the following gradient (arg_grads) with respect to the arguments x:\n\n_x left( log P(t x) + J right)\n\nAlso increment the gradient accumulators for the trainable parameters Θ of the function by:\n\n_Θ left( log P(t x) + J right)\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.choice_gradients",
    "page": "Generative Functions",
    "title": "Gen.choice_gradients",
    "category": "function",
    "text": "(arg_grads, choice_values, choice_grads) = choice_gradients(trace, selection::AddressSet,\n                                                            retgrad)\n\nGiven a previous trace (x t) (trace) and a gradient with respect to the return value _y J (retgrad), return the following gradient (arg_grads) with respect to the arguments x:\n\n_x left( log P(t x) + J right)\n\nAlso given a set of addresses A (selection) that are continuous-valued random choices, return the folowing gradient (choice_grads) with respect to the values of these choices:\n\n_A left( log P(t x) + J right)\n\nAlso return the assignment (choice_values) that is the restriction of t to A.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.get_params",
    "page": "Generative Functions",
    "title": "Gen.get_params",
    "category": "function",
    "text": "get_params(gen_fn::GenerativeFunction)\n\nReturn an iterable over the trainable parameters of the generative function.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Differentiable-programming-1",
    "page": "Generative Functions",
    "title": "Differentiable programming",
    "category": "section",
    "text": "Generative functions may support computation of gradients with respect to (i) all or a subset of its arguments, (ii) its trainable parameters, and (iii) the value of certain random choices. The set of elements (either arguments, trainable parameters, or random choices) for which gradients are available is called the gradient source set. A generative function statically reports whether or not it is able to compute gradients with respect to each of its arguments, through the function has_argument_grads. Let x_G denote the set of arguments for which the generative function does support gradient computation. Similarly, a generative function supports gradients with respect the value of random choices made at all or a subset of addresses. If the return value of the function is conditionally independent of each element in the gradient source set given the other elements in the gradient source set and values of all other random choices, for all possible traces of the function, then the generative function requires a return value gradient to compute gradients with respect to elements of the gradient source set. This static property of the generative function is reported by accepts_output_grad.has_argument_grads\naccepts_output_grad\naccumulate_param_gradients!\nchoice_gradients\nget_params"
},

{
    "location": "ref/gfi/#Gen.project",
    "page": "Generative Functions",
    "title": "Gen.project",
    "category": "function",
    "text": "weight = project(trace::U, selection::AddressSet)\n\nEstimate the probability that the selected choices take the values they do in a trace. \n\nGiven a trace (x t r) (trace) and a set of addresses A (selection), let u denote the restriction of t to A. Return the weight (weight):\n\nlog fracp(r t x)q(t u x) q(r x t)\n\n\n\n\n\n"
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
    "text": "(weight, retval) = assess(gen_fn::GenerativeFunction, args::Tuple, choices::ChoiceMap)\n\nReturn the probability of proposing an assignment\n\nGiven arguments x (args) and an assignment t (choices) such that p(t x)  0, sample r sim q(cdot x t) and  return the weight (weight):\n\nlog fracp(r t x)q(r x t)\n\nIt is an error if p(t x) = 0.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Additional-methods-1",
    "page": "Generative Functions",
    "title": "Additional methods",
    "category": "section",
    "text": "project\npropose\nassess"
},

{
    "location": "ref/gfi/#Custom-generative-function-types-1",
    "page": "Generative Functions",
    "title": "Custom generative function types",
    "category": "section",
    "text": "Most users can just use generative functions written in the Built-in Modeling Language, and can skip this section. However, to develop new modeling DSLs, or optimized implementations of certain probabilistic modeling components, users can also implement custom types of generative functions. We recommend the following steps for implementing a new type of generative function, and also looking at the implementation for the DynamicDSLFunction type as an example."
},

{
    "location": "ref/gfi/#Define-a-trace-data-type-1",
    "page": "Generative Functions",
    "title": "Define a trace data type",
    "category": "section",
    "text": "struct MyTraceType\n    ..\nend"
},

{
    "location": "ref/gfi/#Decide-the-return-type-for-the-generative-function-1",
    "page": "Generative Functions",
    "title": "Decide the return type for the generative function",
    "category": "section",
    "text": "Suppose our return type is Vector{Float64}."
},

{
    "location": "ref/gfi/#Define-a-data-type-for-your-generative-function-1",
    "page": "Generative Functions",
    "title": "Define a data type for your generative function",
    "category": "section",
    "text": "This should be a subtype of GenerativeFunction, with the appropriate type parameters.struct MyGenerativeFunction <: GenerativeFunction{Vector{Float64},MyTraceType}\n..\nendNote that your generative function may not need to have any fields. You can create a constructor for it, e.g.:function MyGenerativeFunction(...)\n..\nend"
},

{
    "location": "ref/gfi/#Decide-what-the-arguments-to-a-generative-function-should-be-1",
    "page": "Generative Functions",
    "title": "Decide what the arguments to a generative function should be",
    "category": "section",
    "text": "For example, our generative functions might take two arguments, a (of type Int) and b (of type Float64). Then, the argument tuple passed to e.g. generate will have two elements.NOTE: Be careful to distinguish between arguments to the generative function itself, and arguments to the constructor of the generative function. For example, if you have a generative function type that is parametrized by, for example, modeling DSL code, this DSL code would be a parameter of the generative function constructor."
},

{
    "location": "ref/gfi/#Decide-what-the-traced-random-choices-(if-any)-will-be-1",
    "page": "Generative Functions",
    "title": "Decide what the traced random choices (if any) will be",
    "category": "section",
    "text": "Remember that each random choice is assigned a unique address in (possibly) hierarchical address space. You are free to design this address space as you wish, although you should document it for users of your generative function type."
},

{
    "location": "ref/gfi/#Implement-the-methods-of-the-interface-1",
    "page": "Generative Functions",
    "title": "Implement the methods of the interface",
    "category": "section",
    "text": "At minimum, you need to implement all methods under the Traces heading (e.g. generate, ..)\nTo support metropolis_hastings or local optimization, or local iterative adjustments to traces, be sure to implement the update and regenerate methods.\nTo support gradients of the log probability density with respect to the arguments and/or random choices made by the function, implement the choice_gradients method.\nGenerative functions can also have trainable parameters (e.g. neural network weights). To support these, implement the accumulate_param_gradients! method.\nTo support use of your generative function in custom proposals (instead of just generative models), implement assess and propose methods."
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
    "text": ""
},

{
    "location": "ref/distributions/#Gen.bernoulli",
    "page": "Probability Distributions",
    "title": "Gen.bernoulli",
    "category": "constant",
    "text": "bernoulli(prob_true::Real)\n\nSamples a Bool value which is true with given probability\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.normal",
    "page": "Probability Distributions",
    "title": "Gen.normal",
    "category": "constant",
    "text": "normal(mu::Real, std::Real)\n\nSamples a Float64 value from a normal distribution.\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.mvnormal",
    "page": "Probability Distributions",
    "title": "Gen.mvnormal",
    "category": "constant",
    "text": "mvnormal(mu::AbstractVector{T}, cov::AbstractMatrix{U}} where {T<:Real,U<:Real}\n\nSamples a Vector{Float64} value from a multivariate normal distribution.\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.gamma",
    "page": "Probability Distributions",
    "title": "Gen.gamma",
    "category": "constant",
    "text": "gamma(shape::Real, scale::Real)\n\nSample a Float64 from a gamma distribution.\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.inv_gamma",
    "page": "Probability Distributions",
    "title": "Gen.inv_gamma",
    "category": "constant",
    "text": "inv_gamma(shape::Real, scale::Real)\n\nSample a Float64 from a inverse gamma distribution.\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.beta",
    "page": "Probability Distributions",
    "title": "Gen.beta",
    "category": "constant",
    "text": "beta(alpha::Real, beta::Real)\n\nSample a Float64 from a beta distribution.\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.categorical",
    "page": "Probability Distributions",
    "title": "Gen.categorical",
    "category": "constant",
    "text": "categorical(probs::AbstractArray{U, 1}) where {U <: Real}\n\nGiven a vector of probabilities probs where sum(probs) = 1, sample an Int i from the set {1, 2, .., length(probs)} with probability probs[i].\n\n\n\n\n\n"
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
    "location": "ref/distributions/#Gen.poisson",
    "page": "Probability Distributions",
    "title": "Gen.poisson",
    "category": "constant",
    "text": "poisson(lambda::Real)\n\nSample an Int from the Poisson distribution with rate lambda.\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.piecewise_uniform",
    "page": "Probability Distributions",
    "title": "Gen.piecewise_uniform",
    "category": "constant",
    "text": "piecewise_uniform(bounds, probs)\n\nSamples a Float64 value from a piecewise uniform continuous distribution.\n\nThere are n bins where n = length(probs) and n + 1 = length(bounds). Bounds must satisfy bounds[i] < bounds[i+1] for all i. The probability density at x is zero if x <= bounds[1] or x >= bounds[end] and is otherwise probs[bin] / (bounds[bin] - bounds[bin+1]) where bounds[bin] < x <= bounds[bin+1].\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.beta_uniform",
    "page": "Probability Distributions",
    "title": "Gen.beta_uniform",
    "category": "constant",
    "text": "beta_uniform(theta::Real, alpha::Real, beta::Real)\n\nSamples a Float64 value from a mixture of a uniform distribution on [0, 1] with probability 1-theta and a beta distribution with parameters alpha and beta with probability theta.\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Built-In-Distributions-1",
    "page": "Probability Distributions",
    "title": "Built-In Distributions",
    "category": "section",
    "text": "bernoulli\nnormal\nmvnormal\ngamma\ninv_gamma\nbeta\ncategorical\nuniform\nuniform_discrete\npoisson\npiecewise_uniform\nbeta_uniform"
},

{
    "location": "ref/distributions/#Gen.random",
    "page": "Probability Distributions",
    "title": "Gen.random",
    "category": "function",
    "text": "val::T = random(dist::Distribution{T}, args...)\n\nSample a random choice from the given distribution with the given arguments.\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.logpdf",
    "page": "Probability Distributions",
    "title": "Gen.logpdf",
    "category": "function",
    "text": "lpdf = logpdf(dist::Distribution{T}, value::T, args...)\n\nEvaluate the log probability (density) of the value.\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.has_output_grad",
    "page": "Probability Distributions",
    "title": "Gen.has_output_grad",
    "category": "function",
    "text": "has::Bool = has_output_grad(dist::Distribution)\n\nReturn true of the gradient if the distribution computes the gradient of the logpdf with respect to the value of the random choice.\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Gen.logpdf_grad",
    "page": "Probability Distributions",
    "title": "Gen.logpdf_grad",
    "category": "function",
    "text": "grads::Tuple = logpdf_grad(dist::Distribution{T}, value::T, args...)\n\nCompute the gradient of the logpdf with respect to the value, and each of the arguments.\n\nIf has_output_grad returns false, then the first element of the returned tuple is nothing. Otherwise, the first element of the tuple is the gradient with respect to the value. If the return value of has_argument_grads has a false value for at position i, then the i+1th element of the returned tuple has value nothing. Otherwise, this element contains the gradient with respect to the ith argument.\n\n\n\n\n\n"
},

{
    "location": "ref/distributions/#Defining-New-Distributions-1",
    "page": "Probability Distributions",
    "title": "Defining New Distributions",
    "category": "section",
    "text": "Probability distributions are singleton types whose supertype is Distribution{T}, where T indicates the data type of the random sample.abstract type Distribution{T} endBy convention, distributions have a global constant lower-case name for the singleton value. For example:struct Bernoulli <: Distribution{Bool} end\nconst bernoulli = Bernoulli()Distributions must implement two methods, random and logpdf.random returns a random sample from the distribution:x::Bool = random(bernoulli, 0.5)\nx::Bool = random(Bernoulli(), 0.5)logpdf returns the log probability (density) of the distribution at a given value:logpdf(bernoulli, false, 0.5)\nlogpdf(Bernoulli(), false, 0.5)Distribution values should also be callable, which is a syntactic sugar with the same behavior of calling random:bernoulli(0.5) # identical to random(bernoulli, 0.5) and random(Bernoulli(), 0.5)A new Distribution type must implement the following methods:random\nlogpdf\nhas_output_grad\nlogpdf_gradA new Distribution type must also implement has_argument_grads."
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
    "text": "Gen provides a built-in embedded modeling language for defining generative functions. The language uses a syntax that extends Julia\'s syntax for defining regular Julia functions.Generative functions in the modeling language are identified using the @gen keyword in front of a Julia function definition. Here is an example @gen function that samples two random choices:@gen function foo(prob::Float64)\n    z1 = @trace(bernoulli(prob), :a)\n    z2 = @trace(bernoulli(prob), :b)\n    return z1 || z2\nendAfter running this code, foo is a Julia value of type DynamicDSLFunction:DynamicDSLFunctionWe can call the resulting generative function like we would a regular Julia function:retval::Bool = foo(0.5)We can also trace its execution:(trace, _) = generate(foo, (0.5,))See Generative Functions for the full set of operations supported by a generative function. Note that the built-in modeling language described in this section is only one of many ways of defining a generative function – generative functions can also be constructed using other embedded languages, or by directly implementing the methods of the generative function interface. However, the built-in modeling language is intended to being flexible enough cover a wide range of use cases. In the remainder of this section, we refer to generative functions defined using the built-in modeling language as @gen functions."
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
    "location": "ref/modeling/#Choices-should-have-constant-support-1",
    "page": "Built-in Modeling Language",
    "title": "Choices should have constant support",
    "category": "section",
    "text": "The support of a random choice at a given address (the set of values with nonzero probability or probability density) must be constant across all possible executions of the @gen function. Violating this discipline will cause errors in certain cases. If the support of a random choice needs to change, use a different address for each distinct value of the support. For example, consider the following generative function:@gen function foo()\n    n = @trace(categorical([0.5, 0.5]), :n) + 1\n    @trace(categorical(ones(n) / n), :x)\nendThe support of the random choice with address :x is either the set 1 2 or 1 2 3. Therefore, this random choice does satisfy our condition above. This would cause an error with the following, in which the :n address is modified, which could result in a change to the domain of the :x variable:tr, _ = generate(foo, (), choicemap((:n, 2), (:x, 3)))\ntr, _ = mh(tr, select(:n))We can modify the address to satisfy the condition by including the domain in the address:@gen function foo()\n    n = @trace(categorical([0.5, 0.5]), :n) + 1\n    @trace(categorical(ones(n) / n), (:x, n))\nend"
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
    "text": "A @gen function may begin with an optional block of trainable parameter declarations. The block consists of a sequence of statements, beginning with @param, that declare the name and Julia type for each trainable parameter. The function below has a single trainable parameter theta with type Float64:@gen function foo(prob::Float64)\n    @param theta::Float64\n    z1 = @trace(bernoulli(prob), :a)\n    z2 = @trace(bernoulli(theta), :b)\n    return z1 || z2\nendTrainable parameters obey the same scoping rules as Julia local variables defined at the beginning of the function body. The value of a trainable parameter is undefined until it is initialized using init_param!. In addition to the current value, each trainable parameter has a current gradient accumulator value. The gradent accumulator value has the same shape (e.g. array dimension) as the parameter value. It is initialized to all zeros, and is incremented by accumulate_param_gradients!.The following methods are exported for the trainable parameters of @gen functions:init_param!\nget_param\nget_param_grad\nset_param!\nzero_param_grad!Trainable parameters are designed to be trained using gradient-based methods. This is discussed in the next section."
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
    "text": "Using the static annotation instructs Gen to statically construct a directed acyclic graph for the computation represented by the body of the function. For the function foo above, the static graph looks like:<div style=\"text-align:center\">\n    <img src=\"../../images/static_graph.png\" alt=\"example static computation graph\" width=\"50%\"/>\n</div>In this graph, oval nodes represent random choices, square nodes represent Julia computations, and diamond nodes represent arguments. The light blue shaded node is the return value of the function. Having access to the static graph allows Gen to generate specialized code for Trace update operations that skips unecessary parts of the computation. Specifically, when applying an update operation, a the graph is analyzed, and each value in the graph identified as having possibly changed, or not. Nodes in the graph do not need to be re-executed if none of their input values could have possibly changed. Also, even if some inputs to a generative function node may have changed, knowledge that some of the inputs have not changed often allows the generative function being called to more efficiently perform its update operation. This is the case for functions produced by Generative Function Combinators.You can plot the graph for a function with the static annotation if you have PyCall installed, and a Python environment that contains the graphviz Python package, using, e.g.:using PyCall\n@pyimport graphviz\nusing Gen: draw_graph\ndraw_graph(foo, graphviz, \"test\")This will produce a file test.pdf in the current working directory containing the rendered graph."
},

{
    "location": "ref/modeling/#Restrictions-1",
    "page": "Built-in Modeling Language",
    "title": "Restrictions",
    "category": "section",
    "text": "In order to be able to construct the static graph, Gen restricts the permitted syntax that can be used in functions annotated with static. In particular, each statement in the body must be one of the following:A pure functional Julia expression on the right-hand side, and a symbol on the left-hand side, e.g.:z4 = !z3A @trace expression on the right-hand side, and a symbol on the left-hand side, e.g.:z2 = @trace(bernoulli(prob), :b)The trace statement must use a literal Julia symbol for the first component in the address. Unlike the full built-in modeling-language, the address is not optional.A return statement, with a literal Julia symbol on the right-hand side, e.g.:return z4The functions must also satisfy the following rules: @trace expressions cannot appear anywhere in the function body except for as the outer-most expression on the right-hand side of a statement.\nEach literal symbol used in the left-hand side of a statement must be unique (e.g. you cannot re-assign to a variable).\nJulia closures and list comprehensions are not allowed.\nFor composite addresses (e.g. :a => 2 => :c) the first component of the address must be a literal symbol, and there may only be one statement in the function body that uses this symbol for the first component of its address.\nJulia control flow constructs (e.g. if, for, while) cannot be used as top-level statements in the function body. Control flow should be implemented inside Julia functions that are called, generative functions that are called such as generative functions produced using Generative Function Combinators."
},

{
    "location": "ref/modeling/#Gen.load_generated_functions",
    "page": "Built-in Modeling Language",
    "title": "Gen.load_generated_functions",
    "category": "function",
    "text": "load_generated_functions()\n\nPermit use of generative functions written in the static modeling language up to this point.\n\n\n\n\n\n"
},

{
    "location": "ref/modeling/#Loading-generated-functions-1",
    "page": "Built-in Modeling Language",
    "title": "Loading generated functions",
    "category": "section",
    "text": "Before a function with a static annotation can be used, the load_generated_functions method must be called:load_generated_functionsTypically, one call to this function, at the top level of a script, separates the definition of generative functions from the execution of inference code, e.g.:using Gen: load_generated_functions\n\n# define generative functions and inference code\n..\n\n# allow static generative functions defined above to be used\nload_generated_functions()\n\n# run inference code\n.."
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
    "text": "By default, the values of Julia computations (all calls that are not random choices or calls to generative functions) are cached as part of the trace, so that Trace update operations can avoid unecessary re-execution of Julia code. However, this cache may grow the memory footprint of a trace. To disable caching of Julia values, use the function annotation nojuliacache (this annotation is ignored unless the static function annotation is also used)."
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
    "text": "gen_fn = Map(kernel::GenerativeFunction)\n\nReturn a new generative function that applies the kernel independently for a vector of inputs.\n\nThe returned generative function has one argument with type Vector{X} for each argument of the input generative function with type X. The length of each argument, which must be the same for each argument, determines the number of times the input generative function is called (N). Each call to the input function is made under address namespace i for i=1..N. The return value of the returned function has type FunctionalCollections.PersistentVector{Y} where Y is the type of the return value of the input function. The map combinator is similar to the \'map\' higher order function in functional programming, except that the map combinator returns a new generative function that must then be separately applied.\n\n\n\n\n\n"
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
    "text": "gen_fn = Unfold(kernel::GenerativeFunction)\n\nReturn a new generative function that applies the kernel in sequence, passing the return value of one application as an input to the next.\n\nThe kernel accepts the following arguments:\n\nThe first argument is the Int index indicating the position in the sequence (starting from 1).\nThe second argument is the state.\nThe kernel may have additional arguments after the state.\n\nThe return type of the kernel must be the same type as the state.\n\nThe returned generative function accepts the following arguments:\n\nThe number of times (N) to apply the kernel.\nThe initial state.\nThe rest of the arguments (not including the state) that will be passed to each kernel application.\n\nThe return type of the returned generative function is FunctionalCollections.PersistentVector{T} where T is the return type of the kernel.\n\n\n\n\n\n"
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
    "text": "key_submap_iterable = get_values_shallow(choices::ChoiceMap)\n\nReturn an iterable collection of tuples (key, value) for each top-level key associated with a value.\n\n\n\n\n\n"
},

{
    "location": "ref/choice_maps/#Gen.get_submaps_shallow",
    "page": "Choice Maps",
    "title": "Gen.get_submaps_shallow",
    "category": "function",
    "text": "key_submap_iterable = get_submaps_shallow(choices::ChoiceMap)\n\nReturn an iterable collection of tuples (key, submap::ChoiceMap) for each top-level key that has a non-empty sub-assignment.\n\n\n\n\n\n"
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
    "location": "ref/choice_maps/#Gen.address_set",
    "page": "Choice Maps",
    "title": "Gen.address_set",
    "category": "function",
    "text": "addrs::AddressSet = address_set(choices::ChoiceMap)\n\nReturn an AddressSet containing the addresses of values in the given assignment.\n\n\n\n\n\n"
},

{
    "location": "ref/choice_maps/#Choice-Maps-1",
    "page": "Choice Maps",
    "title": "Choice Maps",
    "category": "section",
    "text": "Maps from the addresses of random choices to their values are stored in associative tree-structured data structures that have the following abstract type:ChoiceMapChoice maps are constructed by users to express observations and/or constraints on the traces of generative functions. Choice maps are also returned by certain Gen inference methods, and are used internally by various Gen inference methods.Choice maps provide the following methods:has_value\nget_value\nget_submap\nget_values_shallow\nget_submaps_shallow\nto_array\nfrom_array\naddress_setNote that none of these methods mutate the choice map.Choice maps also implement:Base.isempty, which tests of there are no random choices in the choice map\nBase.merge, which takes two choice maps, and returns a new choice map containing all random choices in either choice map. It is an error if the choice maps both have values at the same address, or if one choice map has a value at an address that is the prefix of the address of a value in the other choice map.\n==, which tests if two choice maps have the same addresses and values at those addresses."
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
    "location": "ref/selections/#Selections-1",
    "page": "Selections",
    "title": "Selections",
    "category": "section",
    "text": "A selection is a set of addresses. Users typically construct selections and pass them to Gen inference library methods.There are various concrete types for selections, each of which is a subtype of AddressSet. One such concrete type is DynamicAddressSet, which users can populate using Base.push!, e.g.:sel = DynamicAddressSet()\npush!(sel, :x)\npush!(sel, \"foo\")\npush!(sel, :y => 1 => :z)There is also the following syntactic sugar:sel = select(:x, \"foo\", :y => 1 => :z)"
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
    "location": "ref/inference/#",
    "page": "Inference Library",
    "title": "Inference Library",
    "category": "page",
    "text": ""
},

{
    "location": "ref/inference/#Inference-Library-1",
    "page": "Inference Library",
    "title": "Inference Library",
    "category": "section",
    "text": ""
},

{
    "location": "ref/inference/#Gen.importance_sampling",
    "page": "Inference Library",
    "title": "Gen.importance_sampling",
    "category": "function",
    "text": "(traces, log_norm_weights, lml_est) = importance_sampling(model::GenerativeFunction,\n    model_args::Tuple, observations::ChoiceMap, num_samples::Int)\n\n(traces, log_norm_weights, lml_est) = importance_sampling(model::GenerativeFunction,\n    model_args::Tuple, observations::ChoiceMap,\n    proposal::GenerativeFunction, proposal_args::Tuple,\n    num_samples::Int)\n\nRun importance sampling, returning a vector of traces with associated log weights.\n\nThe log-weights are normalized. Also return the estimate of the marginal likelihood of the observations (lml_est). The observations are addresses that must be sampled by the model in the given model arguments. The first variant uses the internal proposal distribution of the model. The second variant uses a custom proposal distribution defined by the given generative function. All addresses of random choices sampled by the proposal should also be sampled by the model function.\n\n\n\n\n\n"
},

{
    "location": "ref/inference/#Gen.importance_resampling",
    "page": "Inference Library",
    "title": "Gen.importance_resampling",
    "category": "function",
    "text": "(trace, lml_est) = importance_resampling(model::GenerativeFunction,\n    model_args::Tuple, observations::ChoiceMap, num_samples::Int)\n\n(traces, lml_est) = importance_resampling(model::GenerativeFunction,\n    model_args::Tuple, observations::ChoiceMap,\n    proposal::GenerativeFunction, proposal_args::Tuple,\n    num_samples::Int)\n\nRun sampling importance resampling, returning a single trace.\n\nUnlike importance_sampling, the memory used constant in the number of samples.\n\n\n\n\n\n"
},

{
    "location": "ref/inference/#Importance-Sampling-1",
    "page": "Inference Library",
    "title": "Importance Sampling",
    "category": "section",
    "text": "importance_sampling\nimportance_resampling"
},

{
    "location": "ref/inference/#Gen.metropolis_hastings",
    "page": "Inference Library",
    "title": "Gen.metropolis_hastings",
    "category": "function",
    "text": "(new_trace, accepted) = metropolis_hastings(trace, selection::AddressSet)\n\nPerform a Metropolis-Hastings update that proposes new values for the selected addresses from the internal proposal (often using ancestral sampling).\n\n\n\n\n\n(new_trace, accepted) = metropolis_hastings(trace, proposal::GenerativeFunction, proposal_args::Tuple)\n\nPerform a Metropolis-Hastings update that proposes new values for some subset of random choices in the given trace using the given proposal generative function.\n\nThe proposal generative function should take as its first argument the current trace of the model, and remaining arguments proposal_args. If the proposal modifies addresses that determine the control flow in the model, values must be provided by the proposal for any addresses that are newly sampled by the model.\n\n\n\n\n\n(new_trace, accepted) = metropolis_hastings(trace, proposal::GenerativeFunction, proposal_args::Tuple, involution::Function)\n\nPerform a generalized Metropolis-Hastings update based on an involution (bijection that is its own inverse) on a space of assignments.\n\nThe `involution\' Julia function has the following signature:\n\n(new_trace, bwd_choices::ChoiceMap, weight) = involution(trace, fwd_choices::ChoiceMap, fwd_ret, proposal_args::Tuple)\n\nThe generative function proposal is executed on arguments (trace, proposal_args...), producing an assignment fwd_choices and return value fwd_ret. For each value of model arguments (contained in trace) and proposal_args, the involution function applies an involution that maps the tuple (get_choices(trace), fwd_choices) to the tuple (get_choices(new_trace), bwd_choices). Note that fwd_ret is a deterministic function of fwd_choices and proposal_args. When only discrete random choices are used, the weight must be equal to get_score(new_trace) - get_score(trace).\n\nIncluding Continuous Random Choices When continuous random choices are used, the weight must include an additive term that is the determinant of the the Jacobian of the bijection on the continuous random choices that is obtained by currying the involution on the discrete random choices.\n\n\n\n\n\n"
},

{
    "location": "ref/inference/#Gen.mh",
    "page": "Inference Library",
    "title": "Gen.mh",
    "category": "function",
    "text": "(new_trace, accepted) = mh(trace, selection::AddressSet)\n(new_trace, accepted) = mh(trace, proposal::GenerativeFunction, proposal_args::Tuple)\n(new_trace, accepted) = mh(trace, proposal::GenerativeFunction, proposal_args::Tuple, involution::Function)\n\nAlias for metropolis_hastings. Perform a Metropolis-Hastings update on the given trace.\n\n\n\n\n\n"
},

{
    "location": "ref/inference/#Gen.mala",
    "page": "Inference Library",
    "title": "Gen.mala",
    "category": "function",
    "text": "(new_trace, accepted) = mala(trace, selection::AddressSet, tau::Real)\n\nApply a Metropolis-Adjusted Langevin Algorithm (MALA) update.\n\nReference URL\n\n\n\n\n\n"
},

{
    "location": "ref/inference/#Gen.hmc",
    "page": "Inference Library",
    "title": "Gen.hmc",
    "category": "function",
    "text": "(new_trace, accepted) = hmc(trace, selection::AddressSet, mass=0.1, L=10, eps=0.1)\n\nApply a Hamiltonian Monte Carlo (HMC) update.\n\nNeal, Radford M. \"MCMC using Hamiltonian dynamics.\" Handbook of Markov Chain Monte Carlo 2.11 (2011): 2.\n\nReference URL\n\n\n\n\n\n"
},

{
    "location": "ref/inference/#Markov-Chain-Monte-Carlo-1",
    "page": "Inference Library",
    "title": "Markov Chain Monte Carlo",
    "category": "section",
    "text": "The following inference library methods take a trace and return a new trace.metropolis_hastings\nmh\nmala\nhmc"
},

{
    "location": "ref/inference/#Gen.map_optimize",
    "page": "Inference Library",
    "title": "Gen.map_optimize",
    "category": "function",
    "text": "new_trace = map_optimize(trace, selection::AddressSet, \n    max_step_size=0.1, tau=0.5, min_step_size=1e-16, verbose=false)\n\nPerform backtracking gradient ascent to optimize the log probability of the trace over selected continuous choices.\n\nSelected random choices must have support on the entire real line.\n\n\n\n\n\n"
},

{
    "location": "ref/inference/#Optimization-over-Random-Choices-1",
    "page": "Inference Library",
    "title": "Optimization over Random Choices",
    "category": "section",
    "text": "map_optimize"
},

{
    "location": "ref/inference/#Gen.initialize_particle_filter",
    "page": "Inference Library",
    "title": "Gen.initialize_particle_filter",
    "category": "function",
    "text": "state = initialize_particle_filter(model::GenerativeFunction, model_args::Tuple,\n    observations::ChoiceMap proposal::GenerativeFunction, proposal_args::Tuple,\n    num_particles::Int)\n\nInitialize the state of a particle filter using a custom proposal for the initial latent state.\n\n\n\n\n\nstate = initialize_particle_filter(model::GenerativeFunction, model_args::Tuple,\n    observations::ChoiceMap, num_particles::Int)\n\nInitialize the state of a particle filter, using the default proposal for the initial latent state.\n\n\n\n\n\n"
},

{
    "location": "ref/inference/#Gen.particle_filter_step!",
    "page": "Inference Library",
    "title": "Gen.particle_filter_step!",
    "category": "function",
    "text": "particle_filter_step!(state::ParticleFilterState, new_args::Tuple, argdiffs,\n    observations::ChoiceMap, proposal::GenerativeFunction, proposal_args::Tuple)\n\nPerform a particle filter update, where the model arguments are adjusted, new observations are added, and a custom proposal is used for new latent state.\n\n\n\n\n\nparticle_filter_step!(state::ParticleFilterState, new_args::Tuple, argdiffs,\n    observations::ChoiceMap)\n\nPerform a particle filter update, where the model arguments are adjusted, new observations are added, and the default proposal is used for new latent state.\n\n\n\n\n\n"
},

{
    "location": "ref/inference/#Gen.maybe_resample!",
    "page": "Inference Library",
    "title": "Gen.maybe_resample!",
    "category": "function",
    "text": "did_resample::Bool = maybe_resample!(state::ParticleFilterState;\n    ess_threshold::Float64=length(state.traces)/2, verbose=false)\n\nDo a resampling step if the effective sample size is below the given threshold.\n\n\n\n\n\n"
},

{
    "location": "ref/inference/#Gen.log_ml_estimate",
    "page": "Inference Library",
    "title": "Gen.log_ml_estimate",
    "category": "function",
    "text": "estimate = log_ml_estimate(state::ParticleFilterState)\n\nReturn the particle filter\'s current estimate of the log marginal likelihood.\n\n\n\n\n\n"
},

{
    "location": "ref/inference/#Gen.get_traces",
    "page": "Inference Library",
    "title": "Gen.get_traces",
    "category": "function",
    "text": "traces = get_traces(state::ParticleFilterState)\n\nReturn the vector of traces in the current state, one for each particle.\n\n\n\n\n\n"
},

{
    "location": "ref/inference/#Gen.get_log_weights",
    "page": "Inference Library",
    "title": "Gen.get_log_weights",
    "category": "function",
    "text": "log_weights = get_log_weights(state::ParticleFilterState)\n\nReturn the vector of log weights for the current state, one for each particle.\n\nThe weights are not normalized, and are in log-space.\n\n\n\n\n\n"
},

{
    "location": "ref/inference/#Gen.sample_unweighted_traces",
    "page": "Inference Library",
    "title": "Gen.sample_unweighted_traces",
    "category": "function",
    "text": "traces::Vector = sample_unweighted_traces(state::ParticleFilterState, num_samples::Int)\n\nSample a vector of num_samples traces from the weighted collection of traces in the given particle filter state.\n\n\n\n\n\n"
},

{
    "location": "ref/inference/#Particle-Filtering-1",
    "page": "Inference Library",
    "title": "Particle Filtering",
    "category": "section",
    "text": "initialize_particle_filter\nparticle_filter_step!\nmaybe_resample!\nlog_ml_estimate\nget_traces\nget_log_weights\nsample_unweighted_traces"
},

{
    "location": "ref/inference/#Gen.train!",
    "page": "Inference Library",
    "title": "Gen.train!",
    "category": "function",
    "text": "train!(gen_fn::GenerativeFunction, data_generator::Function,\n       update::ParamUpdate,\n       num_epoch, epoch_size, num_minibatch, minibatch_size; verbose::Bool=false)\n\nTrain the given generative function to maximize the expected conditional log probability (density) that gen_fn generates the assignment constraints given inputs, where the expectation is taken under the output distribution of data_generator.\n\nThe function data_generator is a function of no arguments that returns a tuple (inputs, constraints) where inputs is a Tuple of inputs (arguments) to gen_fn, and constraints is an ChoiceMap. conf configures the optimization algorithm used. param_lists is a map from generative function to lists of its parameters. This is equivalent to minimizing the expected KL divergence from the conditional distribution constraints | inputs of the data generator to the distribution represented by the generative function, where the expectation is taken under the marginal distribution on inputs determined by the data generator.\n\n\n\n\n\n"
},

{
    "location": "ref/inference/#Supervised-Training-1",
    "page": "Inference Library",
    "title": "Supervised Training",
    "category": "section",
    "text": "train!"
},

{
    "location": "ref/inference/#Gen.black_box_vi!",
    "page": "Inference Library",
    "title": "Gen.black_box_vi!",
    "category": "function",
    "text": "black_box_vi!(model::GenerativeFunction, args::Tuple,\n              observations::ChoiceMap,\n              proposal::GenerativeFunction, proposal_args::Tuple,\n              update::ParamUpdate;\n              iters=1000, samples_per_iter=100, verbose=false)\n\nFit the parameters of a generative function (proposal) to the posterior distribution implied by the given model and observations using stochastic gradient methods.\n\n\n\n\n\n"
},

{
    "location": "ref/inference/#Variational-Inference-1",
    "page": "Inference Library",
    "title": "Variational Inference",
    "category": "section",
    "text": "black_box_vi!"
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
    "text": "To add support for a new type of gradient-based parameter update, create a new type with the following methods deifned for the types of generative functions that are to be supported.Gen.init_update_state\nGen.apply_update!"
},

]}
