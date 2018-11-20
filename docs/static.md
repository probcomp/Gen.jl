# Static Information Flow Intermediate Representation

directed acyclic graph


types of nodes:

- argument node

- julia node

    > the julia expression (symbols in expressions are replaced with something?)

- random choice node

    > the distribution being sampled from (must be statically inferrable)

    > the list of nodes whose return values give the input

    > the address (a unique symbol)

- generative function call node

    > the generative function being sampled from (must be statically inferrable)

    > the list of nodes whose return values give the input

    > the address (a unique symbol)

- diff julia node

- argdiff node (unique)

- retdiff node (unique)

- call argdiff node

- choicediff node

- calldiff node

- retdiff node


the IR has:

- all the nodes

- a list of the argument nodes

- the return value node



Q: do nodes store the name of the left-hand-side assignment, or something else?


what gets stored in the trace?

- argument node - value

- julia node - value

- random choice node - value

- generative function call node - subtrace (which stores its args and value)

