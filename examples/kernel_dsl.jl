using MacroTools

macro kern(ex)
    check = gensym()
    trace = gensym()

    @capture(ex, function f_(args__) body_ end) || error("expected a function")

    ex = quote
        function $(esc(f))(@tr(), $(args...), $check = false)
            $body
            @tr()
        end
    end

    # replace @tr() with trace gensym
    ex = MacroTools.postwalk(ex) do x
        @capture(x, @tr()) ? trace : x
    end

    # for loops
    ex = MacroTools.postwalk(ex) do x
        @capture(x, for idx_ in range_ body_ end) || return x
        quote
            loop_range = $range
            for $idx in loop_range
                $body
            end
            $check && (loop_range != $range) && error("Check failed in loop")
        end
    end

    # if .. end
    ex = MacroTools.postwalk(ex) do x
        @capture(x, if cond_ body_ end) || return x
        quote
            cond = $cond
            if cond
                $body
            end
            $check && (cond != $cond) && error("Check failed in if-end")
        end
    end

    # let
    ex = MacroTools.postwalk(ex) do x
        @capture(x, let var_ = rhs_; body_ end) || return x
        quote
            rhs = $rhs
            let $var = rhs
                $body
            end
            $check && (rhs != $rhs) && error("Check failed in let")
        end
    end

    # mixture
    ex = MacroTools.postwalk(ex) do x
        @capture(x, let idx_ ~ dist_(args__); body_ end) || return x
        quote
            dist = $dist
            args = ($(args...),)
            let $idx = dist($(args...))
                $body
            end
            $check && (dist != $dist) && error("Check failed in mixture (distribution)")
            $check && (args != ($(args...),)) && error("Check failed in mixture (arguments)")
        end
    end

    # @app statements
    ex = MacroTools.postwalk(ex) do x
        if @capture(x, @app K_(args__))
            quote $trace = $(esc(K))($trace, $(args...)) end
        else
            x
        end
    end
end
