###################
# Trace interface #
###################

"""
    get_args(trace)

Return the argument tuple for a given execution.
"""
function get_args end

"""
    get_retval(trace)

Return the return value of the given execution.
"""
function get_retval end

"""
    get_assmt(trace)

Return a value implementing the assignment interface
"""
function get_assmt end

"""
    get_score(trace)

**Basic case**

Return \$P(t; x)\$

**General case**

Return \$P(r, t; x) / Q(r; tx, t)\$
"""
function get_score end

"""
    gen_fn::GenerativeFunction = get_gen_fn(trace)

Return the generative function that produced the given trace.
"""
function get_gen_fn end

export get_args
export get_retval
export get_assmt
export get_score
export get_gen_fn

######################
# GenerativeFunction #
######################

"""
GenerativeFunction with return value type T and trace type U
"""
abstract type GenerativeFunction{T,U} end

get_return_type(::GenerativeFunction{T,U}) where {T,U} = T
get_trace_type(::GenerativeFunction{T,U}) where {T,U} = U

"""
    bools::Tuple = has_argument_grads(gen_fn::GenerativeFunction)

Return a tuple of booleans indicating whether a gradient is available for each of its arguments.
"""
function has_argument_grads end

"""
    req::Bool = accepts_output_grad(gen_fn::GenerativeFunction)

Return a boolean indicating whether the return value is dependent on any of the *gradient source elements* for any trace.

The gradient source elements are:

- Any argument whose position is true in `has_argument_grads`

- Any static parameter

- Random choices made at a set of addresses that are selectable by `backprop_trace`.
"""
function accepts_output_grad end


"""
    (trace::U, weight) = initialize(gen_fn::GenerativeFunction{T,U}, args::Tuple,
                                    assmt::Assignment)

Return a trace of a generative function that is consistent with the given
assignment.

**Basic case**

Given arguments \$x\$ (`args`) and assignment \$u\$ (`assmt`), sample \$t \\sim
Q(\\cdot; u, x)\$ and return the trace \$(x, t)\$ (`trace`).  Also return the
weight (`weight`):
```math
\\frac{P(t; x)}{Q(t; u, x)}
```

**General case**

Identical to the basic case, except that we also sample \$r \\sim Q(\\cdot; x,
t)\$, the trace is \$(x, t, r)\$ and the weight is:
```math
\\frac{P(t; x)}{Q(t; u, x)}
\\cdot \\frac{P(r; x, t)}{Q(r; x, t)}
```
"""
function initialize(gen_fn::GenerativeFunction, args::Tuple,
                    assmt::Assignment)
    error("Not implemented")
end

function initialize(gen_fn::GenerativeFunction, args::Tuple)
    initialize(gen_fn, args, EmptyAssignment())
end

"""
    weight = project(trace::U, selection::AddressSet)

Estimate the probability that the selected choices take the values they do in a
trace. 

**Basic case**

Given a trace \$(x, t)\$ (`trace`) and a set of addresses \$A\$ (`selection`),
let \$u\$ denote the restriction of \$t\$ to \$A\$. Return the weight
(`weight`):
```math
\\frac{P(t; x)}{Q(t; u, x)}
```

**General case**

Identical to the basic case except that the previous trace is \$(x, t, r)\$ and
the weight is:
```math
\\frac{P(t; x)}{Q(t; u, x)}
\\cdot \\frac{P(r; x, t)}{Q(r; x, t)}
```
"""
function project(trace, selection::AddressSet)
    error("Not implemented")
end

"""
    (assmt, weight, retval) = propose(gen_fn::GenerativeFunction, args::Tuple)

Sample an assignment and compute the probability of proposing that assignment.

**Basic case**

Given arguments (`args`), sample \$t' \\sim P(\\cdot; x)\$, and return \$t\$
(`assmt`) and the weight (`weight`) \$P(t; x)\$.

**General case**

Identical to the basic case, except that we also sample \$r \\sim P(\\cdot; x,
t)\$, and the weight is:
```math
P(t; x)
\\cdot \\frac{P(r; x, t)}{Q(r; x, t)}
```
"""
function propose(gen_fn::GenerativeFunction, args::Tuple)
    error("Not implemented")
end

"""
    (weight, retval) = assess(gen_fn::GenerativeFunction, args::Tuple, assmt::Assignment)

Return the probability of proposing an assignment

**Basic case**

Given arguments \$x\$ (`args`) and an assignment \$t\$ (`assmt`) such that
\$P(t; x) > 0\$, return the weight (`weight`) \$P(t; x)\$.  It is an error if
\$P(t; x) = 0\$.

**General case**

Identical to the basic case except that we also sample \$r \\sim Q(\\cdot; x,
t)\$, and the weight is:
```math
P(t; x)
\\cdot \\frac{P(r; x, t)}{Q(r; x, t)}
```
"""
function assess(gen_fn::GenerativeFunction, args::Tuple, assmt::Assignment)
    (trace, weight) = initialize(gen_fn, args, assmt)
    (weight, get_retval(trace))
end

"""
    (new_trace, weight, discard, retdiff) = force_update(args::Tuple, argdiff, trace,
                                                         assmt::Assignment)

Update a trace by changing the arguments and/or providing new values for some
existing random choice(s) and values for any newly introduced random choice(s).

**Basic case**

Given a previous trace \$(x, t)\$ (`trace`), new arguments \$x'\$ (`args`), and
an assignment \$u\$ (`assmt`), return a new trace \$(x', t')\$ (`new_trace`)
that is consistent with \$u\$.  The values of choices in \$t'\$ are
deterministically copied either from \$t\$ or from \$u\$ (with \$u\$ taking
precedence).  All choices in \$u\$ must appear in \$t'\$.  Also return an
assignment \$v\$ (`discard`) containing the choices in \$t\$ that were
overwritten by values from \$u\$, and any choices in \$t\$ whose address does
not appear in \$t'\$.  Also return the weight (`weight`):
```math
\\frac{P(t'; x')}{P(t; x)}
```

**General case**

Identical to the basic case except that the previous trace is \$(x, t, r)\$,
the new trace is \$(x', t', r')\$ where \$r' \\sim Q(\\cdot; x', t')\$, and the
weight is:
```math
\\frac{P(t'; x')}{P(t; x)}
\\cdot \\frac{P(r'; x', t') Q(r; x, t)}{P(r; x, t) Q(r'; x', t')}
```
"""
function force_update(args::Tuple, argdiff, trace, assmt::Assignment)
    error("Not implemented")
end

"""
    (new_trace, weight, discard, retdiff) = fix_update(args::Tuple, argdiff, trace,
                                                       assmt::Assignment)

Update a trace, by changing the arguments and/or providing new values for some
existing random choice(s).

**Basic case**

Given a previous trace \$(x, t)\$ (`trace`), new arguments \$x'\$ (`args`), and
an assignment \$u\$ (`assmt`), return a new trace \$(x', t')\$ (`new_trace`)
that is consistent with \$u\$.  Let \$u + t\$ denote the merge of \$u\$ and
\$t\$ (with \$u\$ taking precedence).  Sample \$t' \\sim Q(\\cdot; u + t, x)\$.
All addresses in \$u\$ must appear in \$t\$ and in \$t'\$.  Also return an
assignment \$v\$ (`discard`) containing the values from \$t\$ for addresses in
\$u\$.  Also return the weight (`weight`):
```math
\\frac{P(t'; x')}{P(t; x)} \\cdot \\frac{Q(t; v + t', x)}{Q(t'; u + t, x')}
```

**General case**

Identical to the basic case except that the previous trace is \$(x, t, r)\$,
the new trace is \$(x', t', r')\$ where \$r' \\sim Q(\\cdot; x', t')\$, and the
weight is:
```math
\\frac{P(t'; x')}{P(t; x)}
\\cdot \\frac{Q(t; v + t', x)}{Q(t'; u + t, x')}
\\cdot \\frac{P(r'; x', t') Q(r; x, t)}{P(r; x, t) Q(r'; x', t')}
```
"""
function fix_update(args::Tuple, argdiff, trace, assmt::Assignment)
    error("Not implemented")
end

"""
    (new_trace, weight, retdiff) = free_update(args::Tuple, argdiff, trace,
                                               selection::AddressSet)

Update a trace by changing the arguments and/or randomly sampling new values
for selected random choices.

**Basic case**

Given a previous trace \$(x, t)\$ (`trace`), new arguments \$x'\$ (`args`), and
a set of addresses \$A\$ (`selection`), return a new trace \$(x', t')\$
(`new_trace`) such that \$t'\$ agrees with \$t\$ on all addresses not in \$A\$
(\$t\$ and \$t'\$ may have different sets of addresses).  Let \$u\$ denote the
restriction of \$t\$ to the complement of \$A\$.  Sample \$t' \\sim Q(\\cdot;
u, x')\$.  Return the new trace \$(x', t')\$ (`new_trace`) and the weight
(`weight`):
```math
\\frac{P(t'; x')}{P(t; x)}
\\cdot \\frac{Q(t; u', x)}{Q(t'; u, x')}
```
where \$u'\$ is the restriction of \$t'\$ to the complement of \$A\$.

**General case**

Identical to the basic case except that the previous trace is \$(x, t, r)\$,
the new trace is \$(x', t', r')\$ where \$r' \\sim Q(\\cdot; x', t')\$, and the
weight is:
```math
\\frac{P(t'; x')}{P(t; x)}
\\cdot \\frac{Q(t; u', x)}{Q(t'; u, x')}
\\cdot \\frac{P(r'; x', t') Q(r; x, t)}{P(r; x, t) Q(r'; x', t')}
```
"""
function free_update(args::Tuple, argdiff, trace, selection::AddressSet)
    error("Not implemented")
end

"""
    (new_trace, weight, retdiff) = extend(args::Tuple, argdiff, trace, assmt::Assignment)

Extend a trace with new random choices by changing the arguments.

**Basic case**

Given a previous trace \$(x, t)\$ (`trace`), new arguments \$x'\$ (`args`), and
an assignment \$u\$ (`assmt`) that shares no addresses with \$t\$, return a new
trace \$(x', t')\$ (`new_trace`) such that \$t'\$ agrees with \$t\$ on all
addresses in \$t\$ and \$t'\$ agrees with \$u\$ on all addresses in \$u\$.
Sample \$t' \\sim Q(\\cdot; t + u, x')\$. Also return the weight (`weight`):
```math
\\frac{P(t'; x')}{P(t; x) Q(t'; t + u, x')}
```

**General case**

Identical to the basic case except that the previous trace is \$(x, t, r)\$,
and we also sample \$r' \\sim Q(\\cdot; t', x)\$, the new trace is \$(x', t',
r')\$, and the weight is:
```math
\\frac{P(t'; x')}{P(t; x) Q(t'; t + u, x')}
\\cdot \\frac{P(r'; x', t') Q(r; x, t)}{P(r; x, t) Q(r'; x', t')}
```
"""
function extend(args::Tuple, argdiff, trace, assmt::Assignment)
    error("Not implemented")
end

"""
    arg_grads = backprop_params(trace, retgrad)

Increment gradient accumulators for parameters by the gradient of the
log-probability of the trace.

**Basic case**

Given a previous trace \$(x, t)\$ (`trace`) and a gradient with respect to the
return value \$∇_y J\$ (`retgrad`), return the following gradient (`arg_grads`)
with respect to the arguments \$x\$:
```math
∇_x \\left( \\log P(t; x) + J \\right)
```
Also increment the gradient accumulators for the static parameters \$Θ\$ of
the function by:
```math
∇_Θ \\left( \\log P(t; x) + J \\right)
```

**General case**

Not yet formalized.
"""
function backprop_params(trace, retgrad)
    error("Not implemented")
end

"""
    (arg_grads, choice_values, choice_grads) = backprop_trace(trace, selection::AddressSet,
                                                              retgrad)

**Basic case**

Given a previous trace \$(x, t)\$ (`trace`) and a gradient with respect to the
return value \$∇_y J\$ (`retgrad`), return the following gradient (`arg_grads`)
with respect to the arguments \$x\$:
```math
∇_x \\left( \\log P(t; x) + J \\right)
```
Also given a set of addresses \$A\$ (`selection`) that are continuous-valued
random choices, return the folowing gradient (`choice_grads`) with respect to
the values of these choices:
```math
∇_A \\left( \\log P(t; x) + J \\right)
```
Also return the assignment (`choice_values`) that is the restriction of \$t\$ to \$A\$.

**General case**

Not yet formalized.
"""
function backprop_trace(trace, selection::AddressSet, retgrad)
    error("Not implemented")
end

export GenerativeFunction
export has_argument_grads
export accepts_output_grad
export initialize
export project
export propose
export assess
export force_update
export fix_update
export free_update
export extend
export backprop_params
export backprop_trace
