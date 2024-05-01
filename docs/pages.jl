pages = [
    "Home" => "index.md",
    "Getting Started" => [
        "Example 1: Linear Regression" => "getting_started/linear_regression.md",
    ],
    "Tutorials" => [
        "Basics" => [
            "tutorials/basics/modeling_in_gen.md",
            "tutorials/basics/gfi.md",
            "tutorials/basics/combinators.md",
            "tutorials/basics/vi.md"
        ],
        "Advanced" => [
            "tutorials/trace_translators.md",
        ],
        "Model Optmizations" => [
            "Speeding Inference with the Static Modeling Language" => "tutorials/model_optimizations/scaling_with_sml.md",
        ],
    ],
    "How-to Guides" => [
        "MCMC Kernels" => "how_to/mcmc_kernels.md",
        "Custom Distributions" => "how_to/custom_distributions.md",
        "Custom Modeling Languages" => "how_to/custom_dsl.md",
        "Custom Gradients" => "how_to/custom_derivatives.md",
        "Incremental Computation" => "how_to/custom_incremental_computation.md",
    ],
    "API Reference" => [
        "Modeling Library" => [
            "Generative Functions" => "api/model/gfi.md",
            "Probability Distributions" => "api/model/distributions.md",
            "Built-in Modeling Languages" => "api/model/modeling.md",
            "Combinators" => "api/model/combinators.md",
            "Choice Maps" => "api/model/choice_maps.md",
            "Selections" => "api/model/selections.md",
            "Optimizing Trainable Parameters" => "api/model/parameter_optimization.md",
            "Trace Translators" => "api/model/trace_translators.md",
            "Differential Programming" => "api/model/differential_programming.md"
        ],
        "Inference Library" => [
            "Importance Sampling" => "api/inference/importance.md",
            "MAP Optimization" => "api/inference/map.md",
            "Markov chain Monte Carlo" => "api/inference/mcmc.md",
            "MAP Optimization" => "api/inference/map.md",
            "Particle Filtering" => "api/inference/pf.md",
            "Variational Inference" => "api/inference/vi.md",
            "Learning Generative Functions" => "api/inference/learning.md"
        ],
    ],
    "Explanation and Internals" => [
        "Modeling Language Implementation" => "explanations/language_implementation.md"
    ]
]
