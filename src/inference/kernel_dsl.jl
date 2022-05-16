using MacroTools: @capture, postwalk, unblock, rmlines, flatten

function check_observations(choices::ChoiceMap, observations::ChoiceMap)
    for (key, value) in get_values_shallow(observations)
        !has_value(choices, key) && error("Check failed: observed choice at $key not found")
        choices[key] != value && error("Check failed: value of observed choice at $key changed")
    end
    for (key, submap) in get_submaps_shallow(observations)
        check_observations(get_submap(choices, key), submap)
    end
end

is_custom_primitive_kernel(::Function) = false
check_is_kernel(::Function) = false

const _is_custom_primitive_kernel = GlobalRef(Gen, :is_custom_primitive_kernel)
const _check_is_kernel = GlobalRef(Gen, :check_is_kernel)
const _check_observations = GlobalRef(Gen, :check_observations)
const _reversal = GlobalRef(Gen, :reversal)

"""
    @pkern function k(trace, args...; 
                      check=false, observations=EmptyChoiceMap())
        ...
        return trace
    end

Declare a Julia function as a primitive stationary kernel.

The first argument of the function should be a trace, and the return value of the function should be a trace.
There should be keyword arguments check and observations.
"""
macro pkern(ex)
    @capture(ex, function f_(args__) body_ end) || error("expected a function")
    quote
        function $(f)($(args...))
            $body
        end
        $(is_custom_primitive_kernel)(::typeof($f)) = true
        $(_check_is_kernel)(::typeof($f)) = true
    end
end

function expand_kern_ex(ex)
    @capture(ex, function f_(T_, toplevel_args__) body_ end) || error("expected kernel syntax: function my_kernel(trace, args...) (...) end")
    trace = T

    body = postwalk(body) do x
        if @capture(x, for idx_ in range_ body_ end)

            # for loops
            quote
                loop_range = $(range)
                for $(idx) in loop_range
                    $body
                end
                check && (loop_range != $(range)) && error("Check failed in loop")
            end

        elseif @capture(x, if cond_ body_ end)

            # if ... end
            quote
                cond = $(cond)
                if cond
                    $body
                end
                check && (cond != $(cond)) && error("Check failed in if-end")
            end

        elseif @capture(x, let var_ = rhs_; body_ end)

            # let
            quote
                rhs = $(rhs)
                let $(var) = rhs
                    $body
                end
                check && (rhs != $(rhs)) && error("Check failed in let")
            end

        elseif @capture(x, let idx_ ~ dist_(args__); body_ end)

            # mixture
            quote
                dist = $(dist)
                args = ($(args...),)
                let $(idx) = dist($(args...))
                    $body
                end
                check && (dist != $(dist)) && error("Check failed in mixture (distribution)")
                check && (args != ($(args...),)) && error("Check failed in mixture (arguments)")
            end

        elseif @capture(x, T_ ~ k_(T_, args__))
            # applying a kernel
            quote
                check && $(_check_is_kernel)($(k))
                $(T) = $(k)($(T), $(args...),
                            check=check, observations=observations)[1]
            end

        else

            # leave it as is
            x
        end
    end

    ex = quote
        function $(f)($(trace)::Trace, $(toplevel_args...);
                      check=false, observations=EmptyChoiceMap())
            $body
            check && $(_check_observations)(get_choices($(trace)), observations)
            metadata = nothing
            ($(trace), metadata)
        end
        $(_check_observations)(::typeof($f)) = true
    end

    ex, f
end

"""
    k2 = reversal(k1)

Return the reversal kernel for a given kernel.
"""
function reversal(f)
    check_is_kernel(f)
    error("Reversal for kernel $f is not defined")
end

"""
    @rkern k1 : k2

Declare that two primitive stationary kernels are reversals of one another.

The two kernels must have the same argument type signatures.
"""
macro rkern(ex)
    @capture(ex, k_ : l_) || error("expected a pair of functions")
    quote
        $(_is_custom_primitive_kernel)($k) || error("first function is not a custom primitive kernel")
        $(_is_custom_primitive_kernel)($l) || error("second function is not a custom primitive kernel")
        $(_reversal)(::typeof($k)) = $(l)
        $(_reversal)(::typeof($l)) = $(k)
    end
end


function reversal_ex(ex)
    @capture(ex, function f_(toplevel_args__) body_ end) || error("expected a function")

    # modify the body
    body = postwalk(body) do x
        if @capture(x, for idx_ in range_ body_ end)

            # for loops - reverse the order of loop indices
            quote
                for $idx in reverse($range)
                    $body
                end
            end

        elseif @capture(x, T_ ~ k_(T_, args__))

            # applying a kernel - apply the reverse kernel
            quote
                check && $(_check_is_kernel)($(_reversal)($k))
                $(T) = $(_reversal)($k)($(T), $(args...),
                                  check=check, observations=observations)[1]
            end

        elseif isa(x, Expr) && x.head == :block

            # a sequence of things -- reverse the order
            Expr(:block, reverse(x.args)...)

        else

            # leave it as is
            x
        end
    end

    # change the name
    rev = gensym("reverse_kernel")
    ex = quote
        function $(rev)($(toplevel_args...))
            $body
        end
    end

    ex, rev
end

function toplevel(ex::Expr)
    kern_ex, kern = expand_kern_ex(ex)
    rev_kern_ex, rev_kern = reversal_ex(ex)
    rev_kern_ex, _ = expand_kern_ex(rev_kern_ex)
    expr = quote
        # define forward kerel
        $kern_ex

        # define reversal kernel
        $rev_kern_ex

        # bind the reversals for both
        $(_reversal)(::typeof($kern)) = $(rev_kern)
        $(_reversal)(::typeof($rev_kern)) = $(kern)
    end
    expr = postwalk(flatten ∘ unblock ∘ rmlines, expr)
    expr
end

"""
    @kern function k(trace, args...)
        ...
    end

Construct a composite MCMC kernel.

The resulting object is a Julia function that is annotated as a composite MCMC kernel, and can be called as a Julia function or applied within other composite kernels.
"""
macro kern(ex)
    expr = toplevel(ex)
    esc(expr)
end

export @pkern, @rkern, @kern, reversal
