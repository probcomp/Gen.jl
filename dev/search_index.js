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
    "text": "A General-Purpose Probabilistic Programming System with Programmable InferencePages = [\n    \"getting_started.md\",\n    \"tutorials.md\",\n    \"guide.md\",\n    \"ref.md\"\n]\nDepth = 2"
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
    "text": "There are three main components to a typical Gen program. First, we define a generative model, which are like Julia functions, but extended with some extra syntax. The model below samples slope and intercept parameters, and then samples a y-coordinate for each of the x-coordinates that it takes as input.using Gen\n\n@gen function my_model(xs::Vector{Float64})\n    slope = @addr(normal(0, 2), :slope)\n    intercept = @addr(normal(0, 10), :intercept)\n    for (i, x) in enumerate(xs)\n        @addr(normal(slope * x + intercept, 1), \"y-$i\")\n    end\nendThen, we write an inference program that implements an algorithm for manipulating the execution traces of the model. Inference programs are regular Julia code. The inference program below takes a data set, and runs a simple MCMC algorithm to fit slope and intercept parameters:function my_inference_program(xs::Vector{Float64}, ys::Vector{Float64}, num_iters::Int)\n    constraints = DynamicAssignment()\n    for (i, y) in enumerate(ys)\n        constraints[\"y-$i\"] = y\n    end\n    (trace, _) = initialize(my_model, (xs,), constraints)\n    slope_selection = select(:slope)\n    intercept_selection = select(:intercept)\n    for iter=1:num_iters\n        (trace, _) = default_mh(trace, slope_selection)\n        (trace, _) = default_mh(trace, intercept_selection)\n    end\n    assmt = get_assmt(trace)\n    return (assmt[:slope], assmt[:intercept])\nendFinally, we run the inference program on some data, and get the results:xs = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]\nys = [8.23, 5.87, 3.99, 2.59, 0.23, -0.66, -3.53, -6.91, -7.24, -9.90]\n(slope, intercept) = my_inference_program(xs, ys, 1000)\nprintln(\"slope: $slope, intercept: $slope\")"
},

{
    "location": "getting_started/#Visualization-Framework-1",
    "page": "Getting Started",
    "title": "Visualization Framework",
    "category": "section",
    "text": "Because inference programs are regular Julia code, users can use whatever visualization or plotting libraries from the Julia ecosystem that they want. However, we have paired Gen with the GenViz package, which is specialized for visualizing the output and operation of inference algorithms written in Gen."
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
    "location": "ref/#",
    "page": "Reference",
    "title": "Reference",
    "category": "page",
    "text": ""
},

{
    "location": "ref/#Reference-1",
    "page": "Reference",
    "title": "Reference",
    "category": "section",
    "text": ""
},

{
    "location": "ref/#Modeling-Languages-1",
    "page": "Reference",
    "title": "Modeling Languages",
    "category": "section",
    "text": ""
},

{
    "location": "ref/#Inference-Library-1",
    "page": "Reference",
    "title": "Inference Library",
    "category": "section",
    "text": ""
},

{
    "location": "ref/#Generative-Function-Interface-1",
    "page": "Reference",
    "title": "Generative Function Interface",
    "category": "section",
    "text": ""
},

]}
