# Selections

A *selection* is a set of addresses.
Users typically construct selections and pass them to Gen inference library methods.

There are various concrete types for selections, each of which is a subtype of `AddressSet`.
One such concrete type is `DynamicAddressSet`, which users can populate using `Base.push!`, e.g.:
```julia
sel = DynamicAddressSet()
push!(sel, :x)
push!(sel, "foo")
push!(sel, :y => 1 => :z)
```
There is also the following syntactic sugar:
```julia
sel = select(:x, "foo", :y => 1 => :z)
```
