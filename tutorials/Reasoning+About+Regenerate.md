---
layout: splash
---
<br>

# Reasoning About Regenerate

Gen provides a primitive called [`regenerate`](https://probcomp.github.io/Gen/dev/ref/gfi/#Regenerate-1) that allows users to ask for certain random choices in a trace to be re-generated from scratch. `regenerate` is the basis of one variant of the [`metropolis_hastings`](https://probcomp.github.io/Gen/dev/ref/inference/#Gen.metropolis_hastings) operator in Gen's inference library.

This notebook aims to help you understand the computation that `regenerate` is performing.


```julia
using Gen: bernoulli, @gen, @trace
```

Let's start by defining a simple generative function:


```julia
@gen function foo(prob_a)
    val = true
    if @trace(bernoulli(prob_a), :a)
        val = @trace(bernoulli(0.6), :b) && val
    end
    prob_c = val ? 0.9 : 0.2
    val = @trace(bernoulli(prob_c), :c) && val
    return val
end;
```

Recall the distribution on choice maps for this generative function:

$$
\begin{array}{l|l|l}
\mbox{Random choice map } t & \mbox{Probability } p(t; x) & \mbox{Return value } f(x, t) \\
\hline
\{a \mapsto \mbox{true}, b \mapsto \mbox{true}, c \mapsto \mbox{true}\} & \mbox{prob_a} \cdot 0.6\cdot 0.9 & \mbox{true}\\
\{a \mapsto \mbox{true}, b \mapsto \mbox{true}, c \mapsto \mbox{false}\} & \mbox{prob_a} \cdot 0.6 \cdot 0.1 & \mbox{false}\\
\{a \mapsto \mbox{true}, b \mapsto \mbox{false}, c \mapsto \mbox{true}\} & \mbox{prob_a} \cdot 0.4 \cdot 0.2 & \mbox{false}\\
\{a \mapsto \mbox{true}, b \mapsto \mbox{false}, c \mapsto \mbox{false}\} & \mbox{prob_a} \cdot 0.4 \cdot 0.8 & \mbox{false}\\
\{a \mapsto \mbox{false}, c \mapsto \mbox{true}\} & (1-\mbox{prob_a}) \cdot 0.9 & \mbox{true}\\
\{a \mapsto \mbox{false}, c \mapsto \mbox{false}\} & (1-\mbox{prob_a}) \cdot 0.1 & \mbox{false}
\end{array}
$$

Let's first obtain an initial trace with $\{a \mapsto \mbox{true}, b \mapsto \mbox{false}, c \mapsto \mbox{true}\}$, using `generate`:


```julia
using Gen: generate, choicemap

trace, weight = generate(foo, (0.3,), choicemap((:a, true), (:b, false), (:c, true)));
```

Now, we ask for the value at address `:a` to be re-generated:


```julia
using Gen: regenerate, select, NoChange
(trace, weight, retdiff) = regenerate(trace, (0.3,), (NoChange(),), select(:a));
```

Note that unlike [`update`](https://probcomp.github.io/Gen/dev/ref/gfi/#Gen.update), we do not provide the new values for the random choices that we want to change. Instead, we simply pass in a [selection](https://probcomp.github.io/Gen/dev/ref/selections/#Selections-1) indicating the addresses that we want to propose new values for.

Note that `select(:a)` is equivalent to:
```julia
selection = DynamicAddressSet()
push!(selection, :a)
```

We print the choices in the new trace:


```julia
using Gen: get_choices

println(get_choices(trace))
```

    │
    ├── :a : true
    │
    ├── :b : false
    │
    └── :c : true
    


Re-run the regenerate command until you get a trace where `a` is `false`. Note that the address `b` doesn't appear in the resulting trace. Then, run the command again until you get a trace where `a` is `true`. Note that now there is a value for `b`. This value of `b` was sampled along with the new value for `a`---`regenerate` will regenerate new values for the selected adddresses, but also any new addresses that may be introduced as a consequence of stochastic control flow.

What distribution is `regenerate` sampling the selected values from? It turns out that `regenerate` is using the [*internal proposal distribution family*](https://probcomp.github.io/Gen/dev/ref/gfi/#.-Internal-proposal-distribution-family-1) $q(t; x, u)$, just like like `generate`. Recall that for `@gen` functions, the internal proposal distribution is based on *ancestral sampling*.  But whereas `generate` was given the expicit choice map of constraints ($u$) as an argument, `regenerate` constructs $u$ by starting with the previous trace $t$ and then removing any selected addresses. In other words, `regenerate` is like `generate`, but where the constraints are the choices made in the previous trace less the selected choices.

We can make this concrete. Let us start with a deterministic trace again:


```julia
trace, weight = generate(foo, (0.3,), choicemap((:a, true), (:b, false), (:c, true)));
```

### Understanding how regenerate constructs the internal proposal distribution family

We will run `regenerate` with a selection of just `:a`. Let's analyze the internal proposal distribution in this case:

```julia
    (trace, weight, retdiff) = regenerate(trace, (0.3,), noargdiff, select(:a));
```

Since the current trace is $t = \{a \mapsto \mbox{true}, b \mapsto \mbox{false}, c \mapsto \mbox{true}\}$, the constraints $u$ that will be passed to the internal proposal are $u = \{b \mapsto \mbox{false}, c \mapsto \mbox{true}\}$ (everything in $t$ but with the mapping for any selected addresses removed).

To compute the internal proposal distribution, we first write down the list of all choice maps $t'$ where $p(t'; x) > 0$, with the probability $p(t'; x)$ listed.

$$
\begin{array}{l|l}
\mbox{Random choice map } t' & \mbox{Probability } p(t'; x)\\
\hline
\{a \mapsto \mbox{true}, b \mapsto \mbox{true}, c \mapsto \mbox{true}\} & 0.3 \cdot 0.6\cdot 0.9\\
\{a \mapsto \mbox{true}, b \mapsto \mbox{true}, c \mapsto \mbox{false}\} & 0.3 \cdot 0.6 \cdot 0.1\\
\{a \mapsto \mbox{true}, b \mapsto \mbox{false}, c \mapsto \mbox{true}\} & 0.3 \cdot 0.4 \cdot 0.2\\
\{a \mapsto \mbox{true}, b \mapsto \mbox{false}, c \mapsto \mbox{false}\} & 0.3 \cdot 0.4 \cdot 0.8\\
\{a \mapsto \mbox{false}, c \mapsto \mbox{true}\} & 0.7 \cdot 0.9\\
\{a \mapsto \mbox{false}, c \mapsto \mbox{false}\} & 0.7 \cdot 0.1
\end{array}
$$

Then, we eliminate any choice maps $t'$ such that $t'(i) \ne u(i)$ for some address $i$ that is contained in both maps $u$ and $t'$.

In particular:

- we eliminate $t' = \{a \mapsto \mbox{true}, b \mapsto \mbox{true}, c \mapsto \mbox{true}\}$ because $t'(b) = \mbox{true} \ne u(b) = \mbox{false}$.

- we eliminate $t' = \{a \mapsto \mbox{true}, b \mapsto \mbox{true}, c \mapsto \mbox{false}\}$ because $t'(c) = \mbox{false} \ne u(c) = \mbox{true}$.

- we eliminate $t' = \{a \mapsto \mbox{true}, b \mapsto \mbox{false}, c \mapsto \mbox{false}\}$ because $t'(c) = \mbox{false} \ne u(c) = \mbox{true}$.

- we eliminate $t' = \{a \mapsto \mbox{false}, c \mapsto \mbox{false}\}$ because $t'(c) = \mbox{false} \ne u(c) = \mbox{true}$.

For the remaining choice maps $t'$ we require that $q(t'; x, u) > 0$. The remaining two choice maps are:

$$
\begin{array}{l}
t' = \{a \mapsto \mbox{true}, b \mapsto \mbox{false}, c \mapsto \mbox{true}\}\\
t' = \{a \mapsto \mbox{false}, c \mapsto \mbox{true}\}
\end{array}
$$

The ancestral sampling algorithm has two possible results, depending on whether the new value for `a` was `true` or `false`. It will sample `a = true` with probability 0.3 and `a = false` with probability 0.7. If `a` is sampled to be `true`, then the existing value of `b = false` will always be kept. In both cases, the previous value of `c = true` is kept. Therefore, the internal proposal distribution is:

$$
\begin{array}{l|l}
\mbox{Random choice map } t' & q(t'; x, u)\\
\hline
\{a \mapsto \mbox{true}, b \mapsto \mbox{false}, c \mapsto \mbox{true}\} & 0.3\\
\{a \mapsto \mbox{false}, c \mapsto \mbox{true}\} & 0.7
\end{array}
$$

### Reversibility of regenerate

Regenerate has the useful property, that for selected addresses $I$ (e.g. $I = \{a\}$) and initial choice map $t$ where $p(t; x) > 0$, if regenerate has nonzero probability of producing a new choice map $t'$ from $t$ and $I$, then regenerate also has a nonzero probability of producing choice map $t$ from $t'$ and the same set of selected addresses $I$.

*Challenge: convince yourself, or prove, that this is the case.*

### Understanding the weight returned by regenerate

The weight returned by `regenerate`, for selected addresses $I$ is:

$$\log \frac{p(t'; x')q(t; u', x)}{p(t; x) q(t'; u, x')}$$

where $u$ is the restriction of $t$ to the complement of $I$, and where $u'$ is the restriction of $t'$ to the complement of $I$.

We will now manually compute what the weights should be for the two possible transitions from $t = \{a \mapsto \mbox{true}, b \mapsto \mbox{false}, c \mapsto \mbox{true}\}$, with $I = \{a\}$.


First, consider $t' = \{a \mapsto \mbox{true}, b \mapsto \mbox{false}, c \mapsto \mbox{true}\}$. In this case, $u' = \{b \mapsto \mbox{false}, c \mapsto \mbox{true}\}$. The internal proposal distribution in this case is:

$$
\begin{array}{l|l}
\mbox{Random choice map }t & q(t; x, u')\\
\hline
\{a \mapsto \mbox{true}, b \mapsto \mbox{false}, c \mapsto \mbox{true}\} & 0.3\\
\{a \mapsto \mbox{false}, c \mapsto \mbox{true}\} & 0.7
\end{array}
$$

Therefore, the weight for the transition from $t = \{a \mapsto \mbox{true}, b \mapsto \mbox{false}, c \mapsto \mbox{true}\}$ to $t' = \{a \mapsto \mbox{true}, b \mapsto \mbox{false}, c \mapsto \mbox{true}\}$ is:

$$
\log \frac{p(t'; x')q(t; u', x)}{p(t; x) q(t'; u, x')}
= \log \frac{q(t; u', x)}{q(t'; u, x')}
= \log \frac{0.3}{0.3} = 0
$$

Next, we consider $t' = \{a \mapsto \mbox{false}, c \mapsto \mbox{true}\}$. In this case $u' = \{c \mapsto \mbox{true}\}$. The internal proposal distribution in this case is:

$$
\begin{array}{l|l}
\mbox{Random choice map } t & q(t; x, u')\\
\hline
\{a \mapsto \mbox{true}, b \mapsto \mbox{true}, c \mapsto \mbox{true}\} & 0.3 \cdot 0.6\\
\{a \mapsto \mbox{true}, b \mapsto \mbox{false}, c \mapsto \mbox{true}\} & 0.3 \cdot 0.4\\
\{a \mapsto \mbox{false}, c \mapsto \mbox{true}\} & 0.7\\
\end{array}
$$

The weight for the transition from $t = \{a \mapsto \mbox{true}, b \mapsto \mbox{false}, c \mapsto \mbox{true}\}$ to $t' = \{a \mapsto \mbox{false}, c \mapsto \mbox{true}\}$ is:

$$
\log \frac{p(t'; x')q(t; u', x)}{p(t; x) q(t'; u, x')}
= \log \frac{p(t'; x')}{p(t; x)} + \log \frac{q(t; u', x)}{q(t'; u, x')}
= \log \frac{0.7 \cdot 0.9}{0.3 \cdot 0.4 \cdot 0.2} + \log \frac{0.3 \cdot 0.4}{0.7}
$$


```julia
log((0.7 * 0.9)/(0.3 * 0.4 * 0.2)) + log((0.3 * 0.4)/(0.7))
```




    1.504077396776274



Now that we've done all this work, let's check it against Gen.

Run the cell enough times to sample both of the transitions, and confirm that the weights match with our calculations:


```julia
trace, weight = generate(foo, (0.3,), choicemap((:a, true), (:b, false), (:c, true)));
(trace, weight, retdiff) = regenerate(trace, (0.3,), (NoChange(),), select(:a));
println(get_choices(trace))
println("weight: $weight");
```

    │
    ├── :a : false
    │
    └── :c : true
    
    weight: 1.504077396776274


Aren't we glad this is automated by Gen!

### Approaching irreducibility

Exercise: Draw a graph in which each random choice map $t$ where $p(t; x) > 0$ is a node, and where there are directed edges from $t$ to $t'$ if applying `regenerate` to $t$ with selection $\{a\}$ can produce trace $t'$. Do the same for selections $\{b\}$ and $\{c\}$. What about selection $\{a,b,c\}$?
