# Optimizing Static Parameters

To add support for a new type of gradient-based parameter update, create a new type with the following methods deifned for the types of generative functions that are to be supported.
```@docs
Gen.init
Gen.apply_update!
```
