###############################
# intermediate representation #
###############################

include("intermediate_repr.jl")


##########################################
# generation of trace data types from IR #
##########################################

include("trace.jl")


#########################
# basic block generator #
#########################

abstract type StaticDataFlowGenerator{T,U} <: Generator{T,U} end

# a method on the generator type that is executed during expansion of
# generator API generated functions
function get_ir end
function get_grad_fn end

function generate_generator_type(ir::DataFlowIR, trace_type::Symbol, name::Symbol, node_to_gradient)
    generator_type = gensym("StaticDataFlowGenerator_$name")
    retval_type = ir.output_node === nothing ? :Nothing : ir.output_node.typ
    defn = esc(quote
        struct $generator_type <: Gen.StaticDataFlowGenerator{$retval_type, $trace_type}
            params_grad::Dict{Symbol,Any}
            params::Dict{Symbol,Any}
        end
        $generator_type() = $generator_type(Dict{Symbol,Any}(), Dict{Symbol,Any}())

        (gen::$generator_type)(args...) = get_call_record(simulate(gen, args)).retval
        Gen.get_ir(::Type{$generator_type}) = $(QuoteNode(ir))
        #Gen.render_graph(::$generator_type, fname) = Gen.render_graph(Gen.get_ir($generator_type), fname)
        Gen.get_trace_type(::Type{$generator_type}) = $trace_type
        function Gen.get_static_argument_types(::$generator_type)
            [node.typ for node in Gen.get_ir($generator_type).arg_nodes]
        end
        Gen.accepts_output_grad(::$generator_type) = $(QuoteNode(ir.output_ad))
        Gen.has_argument_grads(::$generator_type) = $(QuoteNode(ir.args_ad))
        Gen.get_grad_fn(::Type{$generator_type}, node::Gen.JuliaNode) = $(QuoteNode(node_to_gradient))[node]
    end)
    (defn, generator_type)
end

function is_differentiable(typ::Type)
    typ <: AbstractFloat || typ <: AbstractArray{T} where {T <: AbstractFloat}
end

# TODO refactor and simplify:
function generate_gradient_fn(node::JuliaNode, gradient_fn::Symbol)
    if isa(node.expr_or_value, Expr) || isa(node.expr_or_value, Symbol)
        input_nodes = node.input_nodes
        inputs_do_ad = map((in_node) -> is_differentiable(get_type(in_node)), node.input_nodes)
        untracked_inputs = [gensym("untracked_$(in_node.name)") for in_node in node.input_nodes]
        maybe_tracked_inputs = [in_node.name for in_node in node.input_nodes]
        track_stmts = Expr[]
        grad_exprs = Expr[]
        grad_exprs_noop = Expr[]
        tape = gensym("tape")
        for (untracked, maybe_tracked, do_ad) in zip(untracked_inputs, maybe_tracked_inputs, inputs_do_ad)
            if do_ad
                push!(track_stmts, quote $maybe_tracked = ReverseDiff.track($untracked, $tape) end)
                push!(grad_exprs, quote ReverseDiff.deriv($maybe_tracked) end)
                push!(grad_exprs_noop, quote zero($untracked) end)
            else
                push!(track_stmts, quote $maybe_tracked = $untracked end)
                push!(grad_exprs, quote nothing end)
                push!(grad_exprs_noop, quote nothing end)
            end
        end
        output_grad = gensym("output_grad")
        given_output_value = gensym("given_output_value")
        output_value_maybe_tracked = gensym("output_value_maybe_tracked")
        err_msg = QuoteNode("julia expression was not differentiable: $(node.expr_or_value)")
        quote
            function $gradient_fn($output_grad, $given_output_value, $(untracked_inputs...))
                $tape = ReverseDiff.InstructionTape()
                $(track_stmts...)
                $output_value_maybe_tracked = $(node.expr_or_value)
                @assert isapprox(ReverseDiff.value($output_value_maybe_tracked), $given_output_value)
                if $output_grad !== nothing
                    if ReverseDiff.istracked($output_value_maybe_tracked)
                        ReverseDiff.deriv!($output_value_maybe_tracked, $output_grad)
                        ReverseDiff.reverse_pass!($tape)
                        return ($(grad_exprs...),)
                    else
                        # the output value was not tracked

                        # this could indicate that the expresssion was not
                        # differentiable, which should probably be an error

                        # or, it could indicate that the expression was a constant

                        # TODO revisit

                        return ($(grad_exprs_noop...),)
                    end
                else
                    # output_grad is nothing (i.e. not a floating point value)
                    return ($(grad_exprs_noop...),)
                end
            end
        end
    else
        # it is a constant value
        @assert length(node.input_nodes) == 0
        quote
            $gradient_fn(output_grad, given_output_value) = ()
        end
    end
end

function generate_gradient_functions(ir::DataFlowIR)
    gradient_function_defns = Expr[]
    node_to_gradient = Dict{JuliaNode,Symbol}()
    for node::JuliaNode in filter((node) -> isa(node, JuliaNode), ir.all_nodes)
        gradient_fn = gensym("julia_grad_$(node.output.name)")
        gradient_fn_defn = esc(generate_gradient_fn(node, gradient_fn))
        push!(gradient_function_defns, gradient_fn_defn)
        node_to_gradient[node] = gradient_fn
    end
    (gradient_function_defns, node_to_gradient)
end

#####################
# static parameters #
#####################

# V1: just use a dictionary
# V2: create specialized fields.

# note that parameters will be cached (as specialized fields) in the trace;
# user will need to use assess() after changing the parameters to get a trace
# that has the new values of the parameters, before doing e.g. backprop()

# for V1, just during simulate and assess, the parameters will be read from
# dictionaries

function set_param!(gf::StaticDataFlowGenerator, name::Symbol, value)
    gf.params[name] = value
end

function get_param(gf::StaticDataFlowGenerator, name::Symbol)
    gf.params[name]
end

function get_param_grad(gf::StaticDataFlowGenerator, name::Symbol)
    gf.params_grad[name]
end

function zero_param_grad!(gf::StaticDataFlowGenerator, name::Symbol)
    gf.params_grad[name] = zero(gf.params[name])
end

function init_param!(gf::StaticDataFlowGenerator, name::Symbol, value)
    set_param!(gf, name, value)
    zero_param_grad!(gf, name)
end


######################
# change propagation #
######################

"""
Example: MaskedArgChange{Tuple{Val{:true},Val{:false}},Something}(something)
"""
struct MaskedArgChange{T <: Tuple,U}
    info::U
end

# TODO make the type parameter U part of the StaticDataFlowGenerator type parameter?
get_change_type(::StaticDataFlowGenerator) = MaskedArgChange

function mask(bits...)
    parameters = map((bit) -> Val{bit}, bits)
    MaskedArgChange{Tuple{parameters...},Nothing}(nothing)
end

export MaskedArgChange, mask



###############################
# generator interface methods #
###############################

include("simulate.jl")
include("assess.jl")
include("generate.jl")
include("update.jl")
include("backprop.jl")
