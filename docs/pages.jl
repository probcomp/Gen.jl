pages = [
    "Gen.jl" => "index.md",
    "Tutorials" => [
        "Getting Started" => "tutorials/getting_started.md",
        "Introduction to Modeling in Gen" => "tutorials/modeling_in_gen.md",
        "Basics of MCMC and MAP Inference" => "tutorials/mcmc_map.md",
        "Object Tracking with SMC" => "tutorials/smc.md",
        "Variational Inference in Gen" => "tutorials/vi.md",
        "Learning Generative Functions" => "tutorials/learning_gen_fns.md",
        "Speeding Up Inference with the SML" => "tutorials/scaling_with_sml.md",
    ],
    "How-to Guides" => [
        "Extending Gen" => "how_to/extending_gen.md",
        "Adding New Distributions" => "how_to/custom_distributions.md",
        "Adding New Generative Functions" => "how_to/custom_gen_fns.md",
        "Custom Gradients" => "how_to/custom_gradients.md",
        "Custom Incremental Computation" => "how_to/custom_incremental_computation.md",
    ],
    "Reference" => [
        "Core Interfaces" => [
            "Generative Function Interface" => "ref/core/gfi.md",
            "Choice Maps" => "ref/core/choice_maps.md",
            "Selections" => "ref/core/selections.md",
            "Change Hints" => "ref/core/change_hints.md",
        ],
        "Modeling Library" => [
            "Built-In Modeling Language" => "ref/modeling/dml.md",
            "Static Modeling Language" => "ref/modeling/sml.md",
            "Probability Distributions" => "ref/modeling/distributions.md",
            "Combinators" => "ref/modeling/combinators.md",
            "Custom Generative Functions" => "ref/modeling/custom_gen_fns.md",
        ],
        "Inference Library" => [
            "Enumerative Inference" => "ref/inference/enumerative.md",
            "Importance Sampling" => "ref/inference/importance.md",
            "Markov Chain Monte Carlo" => "ref/inference/mcmc.md",
            "Particle Filtering & SMC" => "ref/inference/pf.md",
            "Trace Translators" => "ref/inference/trace_translators.md",
            "Parameter Optimization" => "ref/inference/parameter_optimization.md",
            "MAP Optimization" => "ref/inference/map.md",
            "Variational Inference" => "ref/inference/vi.md",
            "Wake-Sleep Learning" => "ref/inference/wake_sleep.md",
        ],
        "Internals" => [
            "Modeling Language Implementation" => "ref/internals/language_implementation.md",
        ]
    ],
]
