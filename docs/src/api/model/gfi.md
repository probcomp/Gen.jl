## [Generative Functions](@id gfi_api)

```@docs
GenerativeFunction
Trace
```

The complete set of methods in the generative function interface (GFI) is:

```@docs
simulate
generate
update
regenerate
get_args
get_retval
get_choices
get_score
get_gen_fn
Base.getindex
project
propose
assess
has_argument_grads
has_submap
accepts_output_grad
accumulate_param_gradients!
choice_gradients
get_params
```

```@docs
Diff
NoChange
UnknownChange
SetDiff
Diffed
```

```@docs
CustomUpdateGF
apply_with_state
update_with_state
```

```@docs
CustomGradientGF
apply
gradient
```

```@docs
Gen.init_update_state
Gen.apply_update!
```