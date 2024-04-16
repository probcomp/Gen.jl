# [Hidden Markov Models] (@id hmm)

```@example hmm_tutorial
using Gen

@gen function kernel(n, latent)
    x ~ normal(latent, 1.0) 
    y ~ normal(x, 1.0)
    return y
end

hmm_model = Unfold(kernel)
```

```@example hmm_tutorial
tr = simulate(hmm_model, (2,0))
get_choices(tr)
```

# Plot this trajectory multiple times...
```@example hmm_tutorial
traces = [simulate(hmm_mode, (0,0)) for i in range(1000)]
```
