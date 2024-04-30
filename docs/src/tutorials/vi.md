# Variational Inference (@id vi)

Variational inference involves optimizing the parameters of a variational family to maximize a lower bound on the marginal likelihood called the ELBO.
In Gen, variational families are represented as generative functions, and variational inference typically involves optimizing the trainable parameters of generative functions.

## Reparametrization trick

To use the reparametrization trick to reduce the variance of gradient estimators, users currently need to write two versions of their variational family, one that is reparametrized and one that is not.
Gen does not currently include inference library support for this.
We plan add add automated support for reparametrization and other variance reduction techniques in the future.