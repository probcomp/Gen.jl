# Variational Inference
```@docs
black_box_vi!
black_box_vimco!
```

## Reparametrization trick

To use the reparametrization trick to reduce the variance of gradient estimators, users currently need to write two versions of their variational family, one that is reparametrized and one that is not.
Gen does not currently include inference library support for this.
We plan add add automated support for reparametrization and other variance reduction techniques in the future.
