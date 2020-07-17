# [Optimizing Trainable Parameters](@id optimizing-internal)

To add support for a new type of gradient-based parameter update, create a new type with the following methods defined for the types of generative functions that are to be supported.
```@docs
Gen.init_update_state
Gen.apply_update!
```
