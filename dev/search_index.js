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
    "text": "A General-Purpose Probabilistic Programming System with Programmable InferencePages = [\n    \"getting_started.md\",\n    \"tutorials.md\",\n    \"guide.md\",\n]\nDepth = 2ReferencePages = [\n    \"ref/modeling.md\",\n    \"ref/combinators.md\",\n    \"ref/assignments.md\",\n    \"ref/selections.md\",\n    \"ref/inference.md\",\n    \"ref/gfi.md\",\n    \"ref/distributions.md\"\n]\nDepth = 2"
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
    "text": "First, obtain Julia 1.0 or later, available here.The Gen package can be installed with the Julia package manager. From the Julia REPL, type ] to enter the Pkg REPL mode and then run:pkg> add https://github.com/probcomp/Gen"
},

{
    "location": "getting_started/#Quick-Start-1",
    "page": "Getting Started",
    "title": "Quick Start",
    "category": "section",
    "text": "Let\'s write a short Gen program that does Bayesian linear regression: given a set of points in the (x, y) plane, we want to find a line that fits them well.There are three main components to a typical Gen program.First, we define a generative model: a Julia function, extended with some extra syntax, that, conceptually, simulates a fake dataset. The model below samples slope and intercept parameters, and then for each of the x-coordinates that it accepts as input, samples a corresponding y-coordinate. We name the random choices we make with @addr, so we can refer to them in our inference program.using Gen\n\n@gen function my_model(xs::Vector{Float64})\n    slope = @addr(normal(0, 2), :slope)\n    intercept = @addr(normal(0, 10), :intercept)\n    for (i, x) in enumerate(xs)\n        @addr(normal(slope * x + intercept, 1), \"y-$i\")\n    end\nendSecond, we write an inference program that implements an algorithm for manipulating the execution traces of the model. Inference programs are regular Julia code, and make use of Gen\'s standard inference library.The inference program below takes in a data set, and runs an iterative MCMC algorithm to fit slope and intercept parameters:function my_inference_program(xs::Vector{Float64}, ys::Vector{Float64}, num_iters::Int)\n    # Create a set of constraints fixing the \n    # y coordinates to the observed y values\n    constraints = DynamicAssignment()\n    for (i, y) in enumerate(ys)\n        constraints[\"y-$i\"] = y\n    end\n    \n    # Run the model, constrained by `constraints`,\n    # to get an initial execution trace\n    (trace, _) = initialize(my_model, (xs,), constraints)\n    \n    # Iteratively update the slope then the intercept,\n    # using Gen\'s default_mh operator.\n    slope_selection = select(:slope)\n    intercept_selection = select(:intercept)\n    for iter=1:num_iters\n        (trace, _) = default_mh(trace, slope_selection)\n        (trace, _) = default_mh(trace, intercept_selection)\n    end\n    \n    # From the final trace, read out the slope and\n    # the intercept.\n    assmt = get_assmt(trace)\n    return (assmt[:slope], assmt[:intercept])\nendFinally, we run the inference program on some data, and get the results:xs = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]\nys = [8.23, 5.87, 3.99, 2.59, 0.23, -0.66, -3.53, -6.91, -7.24, -9.90]\n(slope, intercept) = my_inference_program(xs, ys, 1000)\nprintln(\"slope: $slope, intercept: $slope\")"
},

{
    "location": "getting_started/#Visualization-Framework-1",
    "page": "Getting Started",
    "title": "Visualization Framework",
    "category": "section",
    "text": "Because inference programs are regular Julia code, users can use whatever visualization or plotting libraries from the Julia ecosystem that they want. However, we have paired Gen with the GenViz package, which is specialized for visualizing the output and operation of inference algorithms written in Gen.An example demonstrating the use of GenViz for this Quick Start linear regression problem is available in the gen-examples repository. The code there is mostly the same as above, with a few small changes to incorporate an animated visualization of the inference process:It starts a visualization server and initializes a visualization before performing inference:# Start a visualization server on port 8000\nserver = VizServer(8000)\n\n# Initialize a visualization with some parameters\nviz = Viz(server, joinpath(@__DIR__, \"vue/dist\"), Dict(\"xs\" => xs, \"ys\" => ys, \"num\" => length(xs), \"xlim\" => [minimum(xs), maximum(xs)], \"ylim\" => [minimum(ys), maximum(ys)]))\n\n# Open the visualization in a browser\nopenInBrowser(viz)The \"vue/dist\" is a path to a custom trace renderer that draws the (x, y) points and the line represented by a trace; see the GenViz documentation for more details. The code for the renderer is here.It passes the visualization object into the inference program.(slope, intercept) = my_inference_program(xs, ys, 1000000, viz)In the inference program, it puts the current trace into the visualization at each iteration:for iter=1:num_iters\n    putTrace!(viz, 1, trace_to_dict(trace))\n    (trace, _) = default_mh(trace, slope_selection)\n    (trace, _) = default_mh(trace, intercept_selection)\nend"
},

{
    "location": "tutorials/#",
    "page": "Tutorials",
    "title": "Tutorials",
    "category": "page",
    "text": ""
},

{
    "location": "tutorials/#Tutorials-1",
    "page": "Tutorials",
    "title": "Tutorials",
    "category": "section",
    "text": ""
},

{
    "location": "tutorials/#Modeling-and-Basic-Inference-1",
    "page": "Tutorials",
    "title": "Modeling and Basic Inference",
    "category": "section",
    "text": ""
},

{
    "location": "tutorials/#MCMC-and-MAP-Inference-1",
    "page": "Tutorials",
    "title": "MCMC and MAP Inference",
    "category": "section",
    "text": ""
},

{
    "location": "tutorials/#Using-Deep-Learning-for-Inference-1",
    "page": "Tutorials",
    "title": "Using Deep Learning for Inference",
    "category": "section",
    "text": ""
},

{
    "location": "guide/#",
    "page": "Guide",
    "title": "Guide",
    "category": "page",
    "text": ""
},

{
    "location": "guide/#Guide-1",
    "page": "Guide",
    "title": "Guide",
    "category": "section",
    "text": ""
},

{
    "location": "guide/#Modeling-1",
    "page": "Guide",
    "title": "Modeling",
    "category": "section",
    "text": ""
},

{
    "location": "guide/#Defining-Generative-Functions-with-the-Dynamic-DSL-1",
    "page": "Guide",
    "title": "Defining Generative Functions with the Dynamic DSL",
    "category": "section",
    "text": ""
},

{
    "location": "guide/#Defining-Generative-Functions-with-the-Static-DSL-1",
    "page": "Guide",
    "title": "Defining Generative Functions with the Static DSL",
    "category": "section",
    "text": ""
},

{
    "location": "guide/#Generative-Function-Combinators-1",
    "page": "Guide",
    "title": "Generative Function Combinators",
    "category": "section",
    "text": ""
},

{
    "location": "guide/#Traces-1",
    "page": "Guide",
    "title": "Traces",
    "category": "section",
    "text": ""
},

{
    "location": "guide/#Obtaining-an-Initial-Trace-1",
    "page": "Guide",
    "title": "Obtaining an Initial Trace",
    "category": "section",
    "text": ""
},

{
    "location": "guide/#Trainable-Parameters-1",
    "page": "Guide",
    "title": "Trainable Parameters",
    "category": "section",
    "text": ""
},

{
    "location": "guide/#Writing-Tractable-Models-1",
    "page": "Guide",
    "title": "Writing Tractable Models",
    "category": "section",
    "text": ""
},

{
    "location": "guide/#Getting-Good-Performance-Using-Incremental-Computation-1",
    "page": "Guide",
    "title": "Getting Good Performance Using Incremental Computation",
    "category": "section",
    "text": ""
},

{
    "location": "guide/#Getting-Good-Performance-From-the-Static-DSL-1",
    "page": "Guide",
    "title": "Getting Good Performance From the Static DSL",
    "category": "section",
    "text": ""
},

{
    "location": "guide/#Inference-and-Learning-1",
    "page": "Guide",
    "title": "Inference and Learning",
    "category": "section",
    "text": ""
},

{
    "location": "guide/#Importance-Sampling-1",
    "page": "Guide",
    "title": "Importance Sampling",
    "category": "section",
    "text": ""
},

{
    "location": "guide/#Iterative-Inference:-MCMC-and-MAP-Optimization-1",
    "page": "Guide",
    "title": "Iterative Inference: MCMC and MAP Optimization",
    "category": "section",
    "text": ""
},

{
    "location": "guide/#Default-MH-1",
    "page": "Guide",
    "title": "Default MH",
    "category": "section",
    "text": ""
},

{
    "location": "guide/#Custom-MH-1",
    "page": "Guide",
    "title": "Custom MH",
    "category": "section",
    "text": ""
},

{
    "location": "guide/#General-MH-1",
    "page": "Guide",
    "title": "General MH",
    "category": "section",
    "text": ""
},

{
    "location": "guide/#Amortized-Inference-1",
    "page": "Guide",
    "title": "Amortized Inference",
    "category": "section",
    "text": ""
},

{
    "location": "guide/#Expectation-Maximiziation-1",
    "page": "Guide",
    "title": "Expectation Maximiziation",
    "category": "section",
    "text": ""
},

{
    "location": "guide/#Variational-Auto-Encoders-1",
    "page": "Guide",
    "title": "Variational Auto-Encoders",
    "category": "section",
    "text": ""
},

{
    "location": "guide/#Particle-Filtering-1",
    "page": "Guide",
    "title": "Particle Filtering",
    "category": "section",
    "text": ""
},

{
    "location": "guide/#Generative-Function-Interface-1",
    "page": "Guide",
    "title": "Generative Function Interface",
    "category": "section",
    "text": ""
},

{
    "location": "guide/#Internal-Proposals-1",
    "page": "Guide",
    "title": "Internal Proposals",
    "category": "section",
    "text": ""
},

{
    "location": "guide/#Updating-a-Trace-1",
    "page": "Guide",
    "title": "Updating a Trace",
    "category": "section",
    "text": ""
},

{
    "location": "guide/#Writing-New-Inference-Functions-1",
    "page": "Guide",
    "title": "Writing New Inference Functions",
    "category": "section",
    "text": ""
},

{
    "location": "guide/#Implementing-Custom-Generative-Functions-1",
    "page": "Guide",
    "title": "Implementing Custom Generative Functions",
    "category": "section",
    "text": ""
},

{
    "location": "ref/modeling/#",
    "page": "Built-in Modeling Language",
    "title": "Built-in Modeling Language",
    "category": "page",
    "text": ""
},

{
    "location": "ref/modeling/#Built-in-Modeling-Language-1",
    "page": "Built-in Modeling Language",
    "title": "Built-in Modeling Language",
    "category": "section",
    "text": "Gen provides a built-in embedded modeling language for defining generative functions. The language uses a syntax that extends Julia\'s syntax for defining regular Julia functions.Generative functions in the modeling language are identified using the @gen keyword in front of a Julia function definition. Here is an example @gen function that samples two random choices:@gen function foo(prob::Float64)\n    z1 = @addr(bernoulli(prob), :a)\n    z2 = @addr(bernoulli(prob), :b)\n    return z1 || z2\nendAfter running this code, foo is a Julia value of type DynamicDSLFunction, which is a subtype of GenerativeFunction. We can call the resulting generative function:retval::Bool = foo(0.5)We can also trace its execution:(trace, _) = initialize(foo, (0.5,))See Generative Function Interface for the full set of operations supported by a generative function. Note that the built-in modeling language described in this section is only one of many ways of defining a generative function – generative functions can also be constructed using other embedded languages, or by directly implementing the methods of the generative function interface. However, the built-in modeling language is intended to being flexible enough cover a wide range of use cases and to use when learning the Gen programming model.In the remainder of this section, we refer to generative functions defined using the embedded modeling language as @gen functions."
},

{
    "location": "ref/modeling/#Annotations-1",
    "page": "Built-in Modeling Language",
    "title": "Annotations",
    "category": "section",
    "text": "Annotations are a syntactic construct in the modeling language that allows users to provide additional information about how @gen functions should be interpreted. There are two types of annotations – argument annotations and function annotations.Argument annotations. In addition to type declarations on arguments like regular Julia functions, @gen functions also support additional annotations on arguments. Each argument can have the following different syntactic forms:y: No type declaration; no annotations.\ny::Float64: Type declaration; but no annotations.\n(grad)(y): No type declaration provided;, annotated with grad.\n(grad)(y::Float64): Type declaration provided; and annotated with grad.Currently, the possible argument annotations are:grad (see Differentiable programming).Function annotations. The @gen function itself can also be optionally associated with zero or more annotations, which are separate from the per-argument annotations. Function-level annotations use the following different syntactic forms:@gen function foo(<args>) <body> end: No function annotations.\n@gen function (grad) foo(<args>) <body> end: The function has the grad annotation. \n@gen function (grad,static) foo(<args>) <body> end: The function has both the grad and static annotations.Currently the possible function annotations are:grad (see Differentiable programming).\nstatic (see Static DSL)."
},

{
    "location": "ref/modeling/#Making-random-choices-1",
    "page": "Built-in Modeling Language",
    "title": "Making random choices",
    "category": "section",
    "text": "Random choices are made by applying a probability distribution to some arguments:val::Bool = bernoulli(0.5)See Probability Distributions for the set of built-in probability distributions.In the body of a @gen function, wrapping the right-hand side of the expression with an @addr expression associates the random choice with an address, and evaluates to the value of the random choice. The syntax is:@addr(<distribution>(<args>), <addr>)Addresses can be any Julia value. Here, we give the Julia symbol address :z to a Bernoulli random choice.val::Bool = @addr(bernoulli(0.5), :z)Not all random choices need to be given addresses. An address is required if the random choice will be observed, or will be referenced by a custom inference algorithm (e.g. if it will be proposed to by a custom proposal distribution).It is recommended to ensure that the support of a random choice at a given address (the set of values with nonzero probability or probability density) is constant across all possible executions of the @gen function. This discipline will simplify reasoning about the probabilistic behavior of the function, and will help avoid difficult-to-debug NaNs or Infs from appearing. If the support of a random choice needs to change, consider using a different address for each distinct support."
},

{
    "location": "ref/modeling/#Calling-generative-functions-1",
    "page": "Built-in Modeling Language",
    "title": "Calling generative functions",
    "category": "section",
    "text": "@gen functions can invoke other generative functions in three ways:Untraced call: If foo is a generative function, we can invoke foo from within the body of a @gen function using regular call syntax. The random choices made within the call are not given addresses in our trace, and are therefore non-addressable random choices (see Generative Function Interface for details on non-addressable random choices).val = foo(0.5)Traced call with shared address namespace: We can include the addressable random choices made by foo in the caller\'s trace using @splice:val = @splice(foo(0.5))Now, all random choices made by foo are included in our trace. The caller must guarantee that there are no address collisions.Traced call with a nested address namespace: We can include the addressable random choices made by foo in the caller\'s trace, under a namespace, using @addr:val = @addr(foo(0.5), :x)Now, all random choices made by foo are included in our trace, under the namespace :x. For example, if foo makes random choices at addresses :a and :b, these choices will have addresses :x => :a and :x => :b in the caller\'s trace."
},

{
    "location": "ref/modeling/#Composite-addresses-1",
    "page": "Built-in Modeling Language",
    "title": "Composite addresses",
    "category": "section",
    "text": "In Julia, Pair values can be constructed using the => operator. For example, :a => :b is equivalent to Pair(:a, :b) and :a => :b => :c is equivalent to Pair(:a, Pair(:b, :c)). A Pair value (e.g. :a => :b => :c) can be passed as the address field in an @addr expression, provided that there is not also a random choice or generative function called with @addr at any prefix of the address.Consider the following examples.This example is invalid because :a => :b is a prefix of :a => :b => :c:@addr(normal(0, 1), :a => :b => :c)\n@addr(normal(0, 1), :a => :b)This example is invalid because :a is a prefix of :a => :b => :c:@addr(normal(0, 1), :a => :b => :c)\n@addr(normal(0, 1), :a)This example is invalid because :a => :b is a prefix of :a => :b => :c:@addr(normal(0, 1), :a => :b => :c)\n@addr(foo(0.5), :a => :b)This example is invalid because :a is a prefix of :a => :b:@addr(normal(0, 1), :a)\n@addr(foo(0.5), :a => :b)This example is valid because :a => :b and :a => :c are not prefixes of one another:@addr(normal(0, 1), :a => :b)\n@addr(normal(0, 1), :a => :c)This example is valid because :a => :b and :a => :c are not prefixes of one another:@addr(normal(0, 1), :a => :b)\n@addr(foo(0.5), :a => :c)"
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
    "text": "init_param!(gen_fn, name::Symbol, value)\n\nInitialize the the value of a named static parameter of a generative function.\n\nAlso initializes the gradient accumulator for that parameter to zero(value).\n\n\n\n\n\n"
},

{
    "location": "ref/modeling/#Static-parameters-1",
    "page": "Built-in Modeling Language",
    "title": "Static parameters",
    "category": "section",
    "text": "A @gen function may begin with an optional block of static paramter declarations. The block consists of a sequence of statements, beginning with @param, that declare the name and Julia type for each static parameter. The function below has a single static parameter theta with type Float64:@gen function foo(prob::Float64)\n    @param theta::Float64\n    z1 = @addr(bernoulli(prob), :a)\n    z2 = @addr(bernoulli(theta), :b)\n    return z1 || z2\nendStatic parameters obey the same scoping rules as Julia local variables defined at the beginning of the function body. The value of a static parameter is undefined until it is initialized using the following method:init_param!For example:init_param!(foo, :theta, 0.6)Static parameters are designed to be trained using gradient-based methods. This is discussed in Differentiable programming."
},

{
    "location": "ref/modeling/#Differentiable-programming-1",
    "page": "Built-in Modeling Language",
    "title": "Differentiable programming",
    "category": "section",
    "text": "@gen functions can use automatic differentiation to compute gradients with respect to (i) all or a subset of the arguments to the function, and (ii) the values of certain random choices, (iii) the static parameters of the @gen function. We first discuss the semantics of these gradient computations, and then discuss what how to write and use Julia code in the body of a @gen function so that it can be automatically differentiated by the gradient computation."
},

{
    "location": "ref/modeling/#Supported-gradient-computations-1",
    "page": "Built-in Modeling Language",
    "title": "Supported gradient computations",
    "category": "section",
    "text": "Gradients with respect to arguments. A @gen function may have a fixed set of its arguments annotated with grad, which indicates that gradients with respect to that argument should be supported. For example, in the function below, we indicate that we want to support differentiation with respect to the y argument, but that we do not want to support differentiation with respect to the x argument.@gen function foo(x, (grad)(y))\n    if x > 5\n        @addr(normal(y, 1), :z)\n    else\n        @addr(normal(y, 10), :z)\n    end\nendBy default, the function being differentiated is the log probability (density) of the random choices. For the function foo above, when x > 5, the gradient with respect to y is the gradient of the log probability density of a normal distribution with standard deviation 1, with respect to its mean, evaluated at mean y. When x <= 5, we instead differentiate the log density of a normal distribution with standard deviation 10, relative to its mean.Gradients with respect to values of random choices. The author of a @gen function also identifies a set of addresses of random choices with respect to which they wish to support gradients of the log probability (density). Gradients of the log probability (density) with respect to the values of random choices are used in gradient-based numerical optimization of random choices, as well as certain MCMC updates that require gradient information.Gradients with respect to static parameters. The gradient of the log probability (density) with respect to the static parameters can also be computed using automatic differentiation. Currently, the log probability (density) must be a differentiable function of all static parameters.Gradients of a function of the return value. Differentiable programming in Gen composes across function calls. If the return value of the @gen function is conditionally dependent on source elements including (i) any arguments annotated with grad or (ii) any random choices for which gradients are supported, or (ii) any static parameters, then the gradient computation requires a gradient of the an external function with respect to the return value in order to the compute the correct gradients. Thus, the function being differentiated always includes a term representing the log probability (density) of all random choices made by the function, but can be extended with a term that depends on the return value of the function. The author of a @gen function can indicate that the return value depends on the source elements (causing the gradient with respect to the return value is required for all gradient computations) by adding the grad annotation to the @gen function itself. For example, in the function below, the return value is conditionally dependent (and actually identical to) on the random value at address :z:@gen function foo(x, (grad)(y))\n    if x > 5\n        @addr(normal(y, 1), :z)\n    else\n        @addr(normal(y, 10), :z)\n    end\nendIf the author of foo wished to support the computation of gradients with respect to the value of :z, they would need to add the grad annotation to foo using the following syntax:@gen (grad) function foo(x, (grad)(y))\n    if x > 5\n        @addr(normal(y, 1), :z)\n    else\n        @addr(normal(y, 10), :z)\n    end\nend"
},

{
    "location": "ref/modeling/#Writing-differentiable-code-1",
    "page": "Built-in Modeling Language",
    "title": "Writing differentiable code",
    "category": "section",
    "text": "In order to compute the gradients described above, the code in the body of the @gen function needs to be differentiable. Code in the body of a @gen function consists of:Julia code\nMaking random choices\nCalling generative functionsWe now discuss how to ensure that code of each of these forms is differentiable. Note that the procedures for differentiation of code described below are only performed during certain operations on @gen functions (backprop_trace and backprop_params).Julia code. Julia code used within a body of a @gen function is made differentiable using the ReverseDiff package, which implements  reverse-mode automatic differentiation. Specifically, values whose gradient is required (either values of arguments, random choices, or static parameters) are \'tracked\' by boxing them into special values and storing the tracked value on a \'tape\'. For example a Float64 value is boxed into a ReverseDiff.TrackedReal value. Methods (including e.g. arithmetic operators) are defined that operate on these tracked values and produce other tracked values as a result. As the computation proceeds all the values are placed onto the tape, with back-references to the parent operation and operands. Arithmetic operators, array and linear algebra functions, and common special numerical functions, as well as broadcasting, are automatically supported. See ReverseDiff for more details.Making random choices. When making a random choice, each argument is either a tracked value or not. If the argument is a tracked value, then the probability distribution must support differentiation of the log probability (density) with respect to that argument. Otherwise, an error is thrown. The has_argument_grads function indicates which arguments support differentiation for a given distribution (see Probability Distributions). If the gradient is required for the value of a random choice, the distribution must support differentiation of the log probability (density) with respect to the value. This is indicated by the has_output_grad function.Calling generative functions. Like distributions, generative functions indicate which of their arguments support differentiation, using the has_argument_grads function. It is an error if a tracked value is passed as an argument of a generative function, when differentiation is not supported by the generative function for that argument. If a generative function gen_fn has accepts_output_grad(gen_fn) = true, then the return value of the generative function call will be tracked and will propagate further through the caller @gen function\'s computation."
},

{
    "location": "ref/modeling/#Update-code-blocks-1",
    "page": "Built-in Modeling Language",
    "title": "Update code blocks",
    "category": "section",
    "text": ""
},

{
    "location": "ref/modeling/#Static-DSL-1",
    "page": "Built-in Modeling Language",
    "title": "Static DSL",
    "category": "section",
    "text": "The Static DSL supports a subset of the built-in modeling language. A static DSL function is identified by adding the static annotation to the function. For example:@gen (static) function foo(prob::Float64)\n    z1 = @addr(bernoulli(prob), :a)\n    z2 = @addr(bernoulli(prob), :b)\n    z3 = z1 || z2\n    return z3\nendAfter running this code, foo is a Julia value whose type is a subtype of StaticIRGenerativeFunction, which is a subtype of GenerativeFunction.The static DSL permits a subset of the syntax of the built-in modeling language. In particular, each statement must be one of the following forms:<symbol> = <julia-expr>\n<symbol> = @addr(<dist|gen-fn>(..),<symbol> [ => ..])\n@addr(<dist|gen-fn>(..),<symbol> [ => ..])\nreturn <symbol>Currently, static parameters are not supported in static DSL functions.Note that the @addr keyword may only appear in at the top-level of the right-hand-side expresssion. Also, addresses used with the @addr keyword must be a literal Julia symbol (e.g. :a). If multi-part addresses are used, the first component in the multi-part address must be a literal Julia symbol (e.g. :a => i is valid).Also, symbols used on the left-hand-side of assignment statements must be unique (this is called \'static single assignment\' (SSA) form) (this is called \'static single-assignment\' (SSA) form).Loading generated functions. Before a static DSL function can be invoked at runtime, Gen.load_generated_functions() method must be called. Typically, this call immediately preceeds the execution of the inference algorithm.Performance tips. For better performance, annotate the left-hand side of random choices with the type. This permits a more optimized trace data structure to be generated for the generative function. For example:@gen (static) function foo(prob::Float64)\n    z1::Bool = @addr(bernoulli(prob), :a)\n    z2::Bool = @addr(bernoulli(prob), :b)\n    z3 = z1 || z2\n    return z3\nend"
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
    "location": "ref/combinators/#Map-combinator-1",
    "page": "Generative Function Combinators",
    "title": "Map combinator",
    "category": "section",
    "text": "The map combinator takes a generative function as input, and returns a generative function that applies the given generative function independently to a vector of arguments. The returned generative function has one argument with type Vector{T} for each argument of type T of the input generative function. The length of each argument, which must be the same for each argument, determines the number of times the input generative function is called (N). Each call to the input function is made under address namespace i for i=1..N. The return value of the returned function has type Vector{T} where T is the type of the return value of the input function. The map combinator is similar to the \'map\' higher order function in functional programming, except that the map combinator returns a new generative function that must then be separately applied.For example, consider the following generative function, which makes one random choice at address :z:@gen function foo(x::Float64, y::Float64)\n    @addr(normal(x + y, 1.0), :z)\nendWe apply the map combinator to produce a new generative function bar:bar = Map(foo)We can then obtain a trace of bar:xs = [0.0, 0.5]\nys = [0.5, 1.0]\n(trace, _) = initialize(bar, (xs, ys))This causes foo to be invoked twice, once with arguments (xs[1], ys[1]) in address namespace 1 and once with arguments (xs[2], ys[2]) in address namespace 2. The resulting trace has random choices at addresses 1 => :z and 2 => :z."
},

{
    "location": "ref/combinators/#Unfold-combinator-1",
    "page": "Generative Function Combinators",
    "title": "Unfold combinator",
    "category": "section",
    "text": ""
},

{
    "location": "ref/combinators/#Recurse-combinator-1",
    "page": "Generative Function Combinators",
    "title": "Recurse combinator",
    "category": "section",
    "text": ""
},

{
    "location": "ref/assignments/#",
    "page": "Assignments",
    "title": "Assignments",
    "category": "page",
    "text": ""
},

{
    "location": "ref/assignments/#Gen.has_value",
    "page": "Assignments",
    "title": "Gen.has_value",
    "category": "function",
    "text": "has_value(assmt::Assignment, addr)\n\nReturn true if there is a value at the given address.\n\n\n\n\n\n"
},

{
    "location": "ref/assignments/#Gen.get_value",
    "page": "Assignments",
    "title": "Gen.get_value",
    "category": "function",
    "text": "value = get_value(assmt::Assignment, addr)\n\nReturn the value at the given address in the assignment, or throw a KeyError if no value exists. A syntactic sugar is Base.getindex:\n\nvalue = assmt[addr]\n\n\n\n\n\n"
},

{
    "location": "ref/assignments/#Gen.get_subassmt",
    "page": "Assignments",
    "title": "Gen.get_subassmt",
    "category": "function",
    "text": "subassmt = get_subassmt(assmt::Assignment, addr)\n\nReturn the sub-assignment containing all choices whose address is prefixed by addr.\n\nIt is an error if the assignment contains a value at the given address. If there are no choices whose address is prefixed by addr then return an EmptyAssignment.\n\n\n\n\n\n"
},

{
    "location": "ref/assignments/#Gen.get_values_shallow",
    "page": "Assignments",
    "title": "Gen.get_values_shallow",
    "category": "function",
    "text": "key_subassmt_iterable = get_values_shallow(assmt::Assignment)\n\nReturn an iterable collection of tuples (key, subassmt::Assignment) for each top-level key associated with a value.\n\n\n\n\n\n"
},

{
    "location": "ref/assignments/#Gen.get_subassmts_shallow",
    "page": "Assignments",
    "title": "Gen.get_subassmts_shallow",
    "category": "function",
    "text": "key_subassmt_iterable = get_subassmts_shallow(assmt::Assignment)\n\nReturn an iterable collection of tuples (key, subassmt::Assignment) for each top-level key that has a non-empty sub-assignment.\n\n\n\n\n\n"
},

{
    "location": "ref/assignments/#Gen.to_array",
    "page": "Assignments",
    "title": "Gen.to_array",
    "category": "function",
    "text": "arr::Vector{T} = to_array(assmt::Assignment, ::Type{T}) where {T}\n\nPopulate an array with values of choices in the given assignment.\n\nIt is an error if each of the values cannot be coerced into a value of the given type.\n\nImplementation\n\nTo support to_array, a concrete subtype T <: Assignment should implement the following method:\n\nn::Int = _fill_array!(assmt::T, arr::Vector{V}, start_idx::Int) where {V}\n\nPopulate arr with values from the given assignment, starting at start_idx, and return the number of elements in arr that were populated.\n\n\n\n\n\n"
},

{
    "location": "ref/assignments/#Gen.from_array",
    "page": "Assignments",
    "title": "Gen.from_array",
    "category": "function",
    "text": "assmt::Assignment = from_array(proto_assmt::Assignment, arr::Vector)\n\nReturn an assignment with the same address structure as a prototype assignment, but with values read off from the given array.\n\nThe order in which addresses are populated is determined by the prototype assignment. It is an error if the number of choices in the prototype assignment is not equal to the length the array.\n\nImplementation\n\nTo support from_array, a concrete subtype T <: Assignment should implement the following method:\n\n(n::Int, assmt::T) = _from_array(proto_assmt::T, arr::Vector{V}, start_idx::Int) where {V}\n\nReturn an assignment with the same address structure as a prototype assignment, but with values read off from arr, starting at position start_idx, and the number of elements read from arr.\n\n\n\n\n\n"
},

{
    "location": "ref/assignments/#Assignments-1",
    "page": "Assignments",
    "title": "Assignments",
    "category": "section",
    "text": "An assignment is a map from addresses of random choices to their values. Assignments are constructed by users to express observations and/or constraints on the traces of generative functions. Assignments are also returned by certain Gen inference methods, and are used internally by various Gen inference methods.There are various concrete types for assignments, each of which is a subtype of Assignment. Assignments provide the following methods:has_value\nget_value\nget_subassmt\nget_values_shallow\nget_subassmts_shallow\nto_array\nfrom_arrayNote that none of these methods mutate the assignment.Assignments also provide Base.isempty, which tests of there are no random choices in the assignment, and Base.merge, which takes two assignments, and returns a new assignment containing all random choices in either assignment. It is an error if the assignments both have values at the same address, or if one assignment has a value at an address that is the prefix of the address of a value in the other assignment."
},

{
    "location": "ref/assignments/#Gen.DynamicAssignment",
    "page": "Assignments",
    "title": "Gen.DynamicAssignment",
    "category": "type",
    "text": "assmt = DynamicAssignment()\n\nConstruct an empty dynamic assignment.\n\n\n\n\n\n"
},

{
    "location": "ref/assignments/#Gen.set_value!",
    "page": "Assignments",
    "title": "Gen.set_value!",
    "category": "function",
    "text": "set_value!(assmt::DynamicAssignment, addr, value)\n\nSet the given value for the given address.\n\nWill cause any previous value or sub-assignment at this address to be deleted. It is an error if there is already a value present at some prefix of the given address.\n\nThe following syntactic sugar is provided:\n\nassmt[addr] = value\n\n\n\n\n\n"
},

{
    "location": "ref/assignments/#Gen.set_subassmt!",
    "page": "Assignments",
    "title": "Gen.set_subassmt!",
    "category": "function",
    "text": "set_subassmt!(assmt::DynamicAssignment, addr, subassmt::Assignment)\n\nReplace the sub-assignment rooted at the given address with the given sub-assignment. Set the given value for the given address.\n\nWill cause any previous value or sub-assignment at the given address to be deleted. It is an error if there is already a value present at some prefix of address.\n\n\n\n\n\n"
},

{
    "location": "ref/assignments/#Dynamic-Assignment-1",
    "page": "Assignments",
    "title": "Dynamic Assignment",
    "category": "section",
    "text": "One concrete assignment type is DynamicAssignment, which is mutable. Users construct DynamicAssignments and populate them for use as observations or constraints, e.g.:assmt = DynamicAssignment()\nassmt[:x] = true\nassmt[\"foo\"] = 1.25\nassmt[:y => 1 => :z] = -6.3DynamicAssignment\nset_value!\nset_subassmt!"
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
    "text": "(traces, log_norm_weights, lml_est) = importance_sampling(model::GenerativeFunction,\n    model_args::Tuple, observations::Assignment, num_samples::Int)\n\n(traces, log_norm_weights, lml_est) = importance_sampling(model::GenerativeFunction,\n    model_args::Tuple, observations::Assignment,\n    proposal::GenerativeFunction, proposal_args::Tuple,\n    num_samples::Int)\n\nRun importance sampling, returning a vector of traces with associated log weights.\n\nThe log-weights are normalized. Also return the estimate of the marginal likelihood of the observations (lml_est). The observations are addresses that must be sampled by the model in the given model arguments. The first variant uses the internal proposal distribution of the model. The second variant uses a custom proposal distribution defined by the given generative function. All addresses of random choices sampled by the proposal should also be sampled by the model function.\n\n\n\n\n\n"
},

{
    "location": "ref/inference/#Gen.importance_resampling",
    "page": "Inference Library",
    "title": "Gen.importance_resampling",
    "category": "function",
    "text": "(trace, lml_est) = importance_resampling(model::GenerativeFunction,\n    model_args::Tuple, observations::Assignment, num_samples::Int)\n\n(traces, lml_est) = importance_resampling(model::GenerativeFunction,\n    model_args::Tuple, observations::Assignment,\n    proposal::GenerativeFunction, proposal_args::Tuple,\n    num_samples::Int)\n\nRun sampling importance resampling, returning a single trace.\n\nUnlike importance_sampling, the memory used constant in the number of samples.\n\n\n\n\n\n"
},

{
    "location": "ref/inference/#Importance-Sampling-1",
    "page": "Inference Library",
    "title": "Importance Sampling",
    "category": "section",
    "text": "importance_sampling\nimportance_resampling"
},

{
    "location": "ref/inference/#Gen.default_mh",
    "page": "Inference Library",
    "title": "Gen.default_mh",
    "category": "function",
    "text": "(new_trace, accepted) = default_mh(trace, selection::AddressSet)\n\nPerform a Metropolis-Hastings update that proposes new values for the selected addresses from the internal proposal (often using ancestral sampling).\n\n\n\n\n\n"
},

{
    "location": "ref/inference/#Gen.simple_mh",
    "page": "Inference Library",
    "title": "Gen.simple_mh",
    "category": "function",
    "text": "(new_trace, accepted) = simple_mh(trace, proposal::GenerativeFunction, proposal_args::Tuple)\n\nPerform a Metropolis-Hastings update that proposes new values for some subset of random choices in the given trace using the given proposal generative function.\n\nThe proposal generative function should take as its first argument the current trace of the model, and remaining arguments proposal_args. All addresses sampled by the proposal must be in the existing model trace. The proposal may modify the control flow of the model, but values of new addresses in the model are sampled from the model\'s internal proposal distribution.\n\n\n\n\n\n"
},

{
    "location": "ref/inference/#Gen.custom_mh",
    "page": "Inference Library",
    "title": "Gen.custom_mh",
    "category": "function",
    "text": "(new_trace, accepted) = custom_mh(trace, proposal::GenerativeFunction, proposal_args::Tuple)\n\nPerform a Metropolis-Hastings update that proposes new values for some subset of random choices in the given trace using the given proposal generative function.\n\nThe proposal generative function should take as its first argument the current trace of the model, and remaining arguments proposal_args. If the proposal modifies addresses that determine the control flow in the model, values must be provided by the proposal for any addresses that are newly sampled by the model.\n\n\n\n\n\n"
},

{
    "location": "ref/inference/#Gen.general_mh",
    "page": "Inference Library",
    "title": "Gen.general_mh",
    "category": "function",
    "text": "(new_trace, accepted) = general_mh(trace, proposal::GenerativeFunction, proposal_args::Tuple, involution::Function)\n\nPerform a generalized Metropolis-Hastings update based on an involution (bijection that is its own inverse) on a space of assignments.\n\nThe `involution\' Julia function has the following signature:\n\n(new_trace, bwd_assmt::Assignment, weight) = involution(trace, fwd_assmt::Assignment, fwd_ret, proposal_args::Tuple)\n\nThe generative function proposal is executed on arguments (trace, proposal_args...), producing an assignment fwd_assmt and return value fwd_ret. For each value of model arguments (contained in trace) and proposal_args, the involution function applies an involution that maps the tuple (get_assmt(trace), fwd_assmt) to the tuple (get_assmt(new_trace), bwd_assmt). Note that fwd_ret is a deterministic function of fwd_assmt and proposal_args. When only discrete random choices are used, the weight must be equal to get_score(new_trace) - get_score(trace).\n\nIncluding Continuous Random Choices When continuous random choices are used, the weight must include an additive term that is the determinant of the the Jacobian of the bijection on the continuous random choices that is obtained by currying the involution on the discrete random choices.\n\n\n\n\n\n"
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
    "text": "The following inference library methods take a trace and return a new trace.default_mh\nsimple_mh\ncustom_mh\ngeneral_mh\nmala\nhmc"
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
    "location": "ref/inference/#Gen.particle_filter_default",
    "page": "Inference Library",
    "title": "Gen.particle_filter_default",
    "category": "function",
    "text": "(traces, log_norm_weights, lml_est) = particle_filter_default(\n    model::GenerativeFunction, model_args::Tuple, num_steps::Int,\n    num_particles::Int, ess_threshold::Real,\n    init_observations::Assignment, step_observations::Function;\n    verbose=false)\n\nRun particle filtering using the internal proposal of the model.\n\nThe first argument to the model must be an integer, starting with 1, that defines the step. The remaining arguments are given by model_args. The model traces will be initialized with step=1 using the constraints given by init_observations. Then, the step will be consecutively incremented by 1. The function step_observations takes the step and returns a tuple (observations, argdiff) where observations is an assignment containing the values for newly observed random choices for the step, and argdiff describes the argument change from the previous step to the current step.\n\n\n\n\n\n"
},

{
    "location": "ref/inference/#Gen.particle_filter_custom",
    "page": "Inference Library",
    "title": "Gen.particle_filter_custom",
    "category": "function",
    "text": "(traces, log_norm_weights, lml_est) = particle_filter_custom(\n    model::GenerativeFunction, model_args::Tuple, num_steps::Int,\n    num_steps::Int, num_particles::Int, ess_threshold::Real,\n    init_observations::Assignment, init_proposal_args::Tuple,\n    step_observations::Function, step_proposal_args::Function,\n    init_proposal::GenerativeFunction, step_proposal::GenerativeFunction;\n    verbose::Bool=false)\n\nRun particle filtering using custom proposal(s) at each step.\n\n\n\n\n\n"
},

{
    "location": "ref/inference/#Particle-Filtering-1",
    "page": "Inference Library",
    "title": "Particle Filtering",
    "category": "section",
    "text": "particle_filter_default\nparticle_filter_custom"
},

{
    "location": "ref/inference/#Training-Generative-Functions-1",
    "page": "Inference Library",
    "title": "Training Generative Functions",
    "category": "section",
    "text": "sgd_train_batch"
},

{
    "location": "ref/gfi/#",
    "page": "Generative Function Interface",
    "title": "Generative Function Interface",
    "category": "page",
    "text": ""
},

{
    "location": "ref/gfi/#Generative-Function-Interface-1",
    "page": "Generative Function Interface",
    "title": "Generative Function Interface",
    "category": "section",
    "text": "A trace is a record of an execution of a generative function. There is no abstract type representing all traces. Generative functions implement the generative function interface, which is a set of methods that involve the execution traces and probabilistic behavior of generative functions. In the mathematical description of the interface methods, we denote arguments to a function by x, complete assignments of values to addresses of random choices (containing all the random choices made during some execution) by t and partial assignments by either u or v. We denote a trace of a generative function by the tuple (x t). We say that two assignments u and t agree when they assign addresses that appear in both assignments to the same values (they can different or even disjoint sets of addresses and still agree). A generative function is associated with a family of probability distributions P(t x) on assignments t, parameterized by arguments x, and a second family of distributions Q(t u x) on assignments t parameterized by partial assignment u and arguments x. Q is called the internal proposal family of the generative function, and satisfies that if u and t agree then P(t x)  0 if and only if Q(t x u)  0, and that Q(t x u)  0 implies that u and t agree. See the Gen technical report for additional details."
},

{
    "location": "ref/gfi/#Non-addressabe-random-choices-1",
    "page": "Generative Function Interface",
    "title": "Non-addressabe random choices",
    "category": "section",
    "text": "Generative functions may also use non-addressable random choices, denoted r. Unlike regular (addressable) random choices, non-addressable random choices do not have addresses, and the value of non-addressable random choices is not exposed through the generative function interface. However, the state of non-addressable random choices is maintained in the trace. A trace that contains non-addressable random choices is denoted (x t r). Non-addressable random choices manifest to the user of the interface as stochasticity in weights returned by generative function interface methods. The behavior of non-addressable random choices is defined by an additional pair of families of distributions associated with the generative function, denoted Q(r x t) and P(r x t), which are defined for P(t x)  0, and which satisfy Q(r x t)  0 if and only if P(r x t)  0. For each generative function below, we describe its semantics first in the basic setting where there is no non-addressable random choices, and then in the more general setting that may include non-addressable random choices."
},

{
    "location": "ref/gfi/#Differentiable-programming-1",
    "page": "Generative Function Interface",
    "title": "Differentiable programming",
    "category": "section",
    "text": "Generative functions may support computation of gradients with respect to (i) all or a subset of its arguments, (ii) its trainable parameters, and (iii) the value of certain random choices. The set of elements (either arguments, trainable parameters, or random choices) for which gradients are available is called the gradient source set. A generative function statically reports whether or not it is able to compute gradients with respect to each of its arguments, through the function has_argument_grads. Let x_G denote the set of arguments for which the generative function does support gradient computation. Similarly, a generative function supports gradients with respect the value of random choices made at all or a subset of addresses. If the return value of the function is conditionally independent of each element in the gradient source set given the other elements in the gradient source set and values of all other random choices, for all possible traces of the function, then the generative function requires a return value gradient to compute gradients with respect to elements of the gradient source set. This static property of the generative function is reported by accepts_output_grad."
},

{
    "location": "ref/gfi/#Static-parameters-1",
    "page": "Generative Function Interface",
    "title": "Static parameters",
    "category": "section",
    "text": ""
},

{
    "location": "ref/gfi/#Gen.has_argument_grads",
    "page": "Generative Function Interface",
    "title": "Gen.has_argument_grads",
    "category": "function",
    "text": "bools::Tuple = has_argument_grads(gen_fn::GenerativeFunction)\n\nReturn a tuple of booleans indicating whether a gradient is available for each of its arguments.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.accepts_output_grad",
    "page": "Generative Function Interface",
    "title": "Gen.accepts_output_grad",
    "category": "function",
    "text": "req::Bool = accepts_output_grad(gen_fn::GenerativeFunction)\n\nReturn a boolean indicating whether the return value is dependent on any of the gradient source elements for any trace.\n\nThe gradient source elements are:\n\nAny argument whose position is true in has_argument_grads\nAny static parameter\nRandom choices made at a set of addresses that are selectable by backprop_trace.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.initialize",
    "page": "Generative Function Interface",
    "title": "Gen.initialize",
    "category": "function",
    "text": "(trace::U, weight) = initialize(gen_fn::GenerativeFunction{T,U}, args::Tuple,\n                                assmt::Assignment)\n\nReturn a trace of a generative function that is consistent with the given assignment.\n\nBasic case\n\nGiven arguments x (args) and assignment u (assmt), sample t sim Q(cdot u x) and return the trace (x t) (trace).  Also return the weight (weight):\n\nfracP(t x)Q(t u x)\n\nGeneral case\n\nIdentical to the basic case, except that we also sample r sim Q(cdot x t), the trace is (x t r) and the weight is:\n\nfracP(t x)Q(t u x)\ncdot fracP(r x t)Q(r x t)\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.project",
    "page": "Generative Function Interface",
    "title": "Gen.project",
    "category": "function",
    "text": "weight = project(trace::U, selection::AddressSet)\n\nEstimate the probability that the selected choices take the values they do in a trace. \n\nBasic case\n\nGiven a trace (x t) (trace) and a set of addresses A (selection), let u denote the restriction of t to A. Return the weight (weight):\n\nfracP(t x)Q(t u x)\n\nGeneral case\n\nIdentical to the basic case except that the previous trace is (x t r) and the weight is:\n\nfracP(t x)Q(t u x)\ncdot fracP(r x t)Q(r x t)\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.propose",
    "page": "Generative Function Interface",
    "title": "Gen.propose",
    "category": "function",
    "text": "(assmt, weight, retval) = propose(gen_fn::GenerativeFunction, args::Tuple)\n\nSample an assignment and compute the probability of proposing that assignment.\n\nBasic case\n\nGiven arguments (args), sample t sim P(cdot x), and return t (assmt) and the weight (weight) P(t x).\n\nGeneral case\n\nIdentical to the basic case, except that we also sample r sim P(cdot x t), and the weight is:\n\nP(t x)\ncdot fracP(r x t)Q(r x t)\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.assess",
    "page": "Generative Function Interface",
    "title": "Gen.assess",
    "category": "function",
    "text": "(weight, retval) = assess(gen_fn::GenerativeFunction, args::Tuple, assmt::Assignment)\n\nReturn the probability of proposing an assignment\n\nBasic case\n\nGiven arguments x (args) and an assignment t (assmt) such that P(t x)  0, return the weight (weight) P(t x).  It is an error if P(t x) = 0.\n\nGeneral case\n\nIdentical to the basic case except that we also sample r sim Q(cdot x t), and the weight is:\n\nP(t x)\ncdot fracP(r x t)Q(r x t)\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.force_update",
    "page": "Generative Function Interface",
    "title": "Gen.force_update",
    "category": "function",
    "text": "(new_trace, weight, discard, retdiff) = force_update(args::Tuple, argdiff, trace,\n                                                     assmt::Assignment)\n\nUpdate a trace by changing the arguments and/or providing new values for some existing random choice(s) and values for any newly introduced random choice(s).\n\nBasic case\n\nGiven a previous trace (x t) (trace), new arguments x (args), and an assignment u (assmt), return a new trace (x t) (new_trace) that is consistent with u.  The values of choices in t are deterministically copied either from t or from u (with u taking precedence).  All choices in u must appear in t.  Also return an assignment v (discard) containing the choices in t that were overwritten by values from u, and any choices in t whose address does not appear in t.  Also return the weight (weight):\n\nfracP(t x)P(t x)\n\nGeneral case\n\nIdentical to the basic case except that the previous trace is (x t r), the new trace is (x t r) where r sim Q(cdot x t), and the weight is:\n\nfracP(t x)P(t x)\ncdot fracP(r x t) Q(r x t)P(r x t) Q(r x t)\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.fix_update",
    "page": "Generative Function Interface",
    "title": "Gen.fix_update",
    "category": "function",
    "text": "(new_trace, weight, discard, retdiff) = fix_update(args::Tuple, argdiff, trace,\n                                                   assmt::Assignment)\n\nUpdate a trace, by changing the arguments and/or providing new values for some existing random choice(s).\n\nBasic case\n\nGiven a previous trace (x t) (trace), new arguments x (args), and an assignment u (assmt), return a new trace (x t) (new_trace) that is consistent with u.  Let u + t denote the merge of u and t (with u taking precedence).  Sample t sim Q(cdot u + t x). All addresses in u must appear in t and in t.  Also return an assignment v (discard) containing the values from t for addresses in u.  Also return the weight (weight):\n\nfracP(t x)P(t x) cdot fracQ(t v + t x)Q(t u + t x)\n\nGeneral case\n\nIdentical to the basic case except that the previous trace is (x t r), the new trace is (x t r) where r sim Q(cdot x t), and the weight is:\n\nfracP(t x)P(t x)\ncdot fracQ(t v + t x)Q(t u + t x)\ncdot fracP(r x t) Q(r x t)P(r x t) Q(r x t)\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.free_update",
    "page": "Generative Function Interface",
    "title": "Gen.free_update",
    "category": "function",
    "text": "(new_trace, weight, retdiff) = free_update(args::Tuple, argdiff, trace,\n                                           selection::AddressSet)\n\nUpdate a trace by changing the arguments and/or randomly sampling new values for selected random choices.\n\nBasic case\n\nGiven a previous trace (x t) (trace), new arguments x (args), and a set of addresses A (selection), return a new trace (x t) (new_trace) such that t agrees with t on all addresses not in A (t and t may have different sets of addresses).  Let u denote the restriction of t to the complement of A.  Sample t sim Q(cdot u x).  Return the new trace (x t) (new_trace) and the weight (weight):\n\nfracP(t x)P(t x)\ncdot fracQ(t u x)Q(t u x)\n\nwhere u is the restriction of t to the complement of A.\n\nGeneral case\n\nIdentical to the basic case except that the previous trace is (x t r), the new trace is (x t r) where r sim Q(cdot x t), and the weight is:\n\nfracP(t x)P(t x)\ncdot fracQ(t u x)Q(t u x)\ncdot fracP(r x t) Q(r x t)P(r x t) Q(r x t)\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.extend",
    "page": "Generative Function Interface",
    "title": "Gen.extend",
    "category": "function",
    "text": "(new_trace, weight, retdiff) = extend(args::Tuple, argdiff, trace, assmt::Assignment)\n\nExtend a trace with new random choices by changing the arguments.\n\nBasic case\n\nGiven a previous trace (x t) (trace), new arguments x (args), and an assignment u (assmt) that shares no addresses with t, return a new trace (x t) (new_trace) such that t agrees with t on all addresses in t and t agrees with u on all addresses in u. Sample t sim Q(cdot t + u x). Also return the weight (weight):\n\nfracP(t x)P(t x) Q(t t + u x)\n\nGeneral case\n\nIdentical to the basic case except that the previous trace is (x t r), and we also sample r sim Q(cdot t x), the new trace is (x t r), and the weight is:\n\nfracP(t x)P(t x) Q(t t + u x)\ncdot fracP(r x t) Q(r x t)P(r x t) Q(r x t)\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.backprop_params",
    "page": "Generative Function Interface",
    "title": "Gen.backprop_params",
    "category": "function",
    "text": "arg_grads = backprop_params(trace, retgrad)\n\nIncrement gradient accumulators for parameters by the gradient of the log-probability of the trace.\n\nBasic case\n\nGiven a previous trace (x t) (trace) and a gradient with respect to the return value _y J (retgrad), return the following gradient (arg_grads) with respect to the arguments x:\n\n_x left( log P(t x) + J right)\n\nAlso increment the gradient accumulators for the static parameters Θ of the function by:\n\n_Θ left( log P(t x) + J right)\n\nGeneral case\n\nNot yet formalized.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.backprop_trace",
    "page": "Generative Function Interface",
    "title": "Gen.backprop_trace",
    "category": "function",
    "text": "(arg_grads, choice_values, choice_grads) = backprop_trace(trace, selection::AddressSet,\n                                                          retgrad)\n\nBasic case\n\nGiven a previous trace (x t) (trace) and a gradient with respect to the return value _y J (retgrad), return the following gradient (arg_grads) with respect to the arguments x:\n\n_x left( log P(t x) + J right)\n\nAlso given a set of addresses A (selection) that are continuous-valued random choices, return the folowing gradient (choice_grads) with respect to the values of these choices:\n\n_A left( log P(t x) + J right)\n\nAlso return the assignment (choice_values) that is the restriction of t to A.\n\nGeneral case\n\nNot yet formalized.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.get_assmt",
    "page": "Generative Function Interface",
    "title": "Gen.get_assmt",
    "category": "function",
    "text": "get_assmt(trace)\n\nReturn a value implementing the assignment interface\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.get_args",
    "page": "Generative Function Interface",
    "title": "Gen.get_args",
    "category": "function",
    "text": "get_args(trace)\n\nReturn the argument tuple for a given execution.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.get_retval",
    "page": "Generative Function Interface",
    "title": "Gen.get_retval",
    "category": "function",
    "text": "get_retval(trace)\n\nReturn the return value of the given execution.\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Gen.get_score",
    "page": "Generative Function Interface",
    "title": "Gen.get_score",
    "category": "function",
    "text": "get_score(trace)\n\nBasic case\n\nReturn P(t x)\n\nGeneral case\n\nReturn P(r t x)  Q(r tx t)\n\n\n\n\n\n"
},

{
    "location": "ref/gfi/#Interface-methods-1",
    "page": "Generative Function Interface",
    "title": "Interface methods",
    "category": "section",
    "text": "has_argument_grads\naccepts_output_grad\ninitialize\nproject\npropose\nassess\nforce_update\nfix_update\nfree_update\nextend\nbackprop_params\nbackprop_trace\nget_assmt\nget_args\nget_retval\nget_score"
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
    "location": "ref/distributions/#Defining-New-Distributions-1",
    "page": "Probability Distributions",
    "title": "Defining New Distributions",
    "category": "section",
    "text": "Probability distributions are singleton types whose supertype is Distribution{T}, where T indicates the data type of the random sample.abstract type Distribution{T} endBy convention, distributions have a global constant lower-case name for the singleton value. For example:struct Bernoulli <: Distribution{Bool} end\nconst bernoulli = Bernoulli()Distributions must implement two methods, random and logpdf.random returns a random sample from the distribution:x::Bool = random(bernoulli, 0.5)\nx::Bool = random(Bernoulli(), 0.5)logpdf returns the log probability (density) of the distribution at a given value:logpdf(bernoulli, false, 0.5)\nlogpdf(Bernoulli(), false, 0.5)Distribution values are also callable, which is a syntactic sugar with the same behavior of calling random:bernoulli(0.5) # identical to random(bernoulli, 0.5) and random(Bernoulli(), 0.5)"
},

]}
