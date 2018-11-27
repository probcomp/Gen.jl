abstract type StaticIRNode end
abstract type RegularNode <: StaticIRNode end
abstract type DiffNode <: StaticIRNode end

#################
# regular nodes #
#################

# NOTE: statically inferring types is possible in some cases, but not yet
# implemented. this might allow us to simplify the distribution and generative
# function interfaces (by removing the concrete/static type getters)

struct ArgumentNode <: RegularNode
    # variable name (from the signature, defaults to a gensym)
    name::Symbol

    # optional declared type (default is Any)
    typ::Type
end

struct ConstantNode <: RegularNode
    value::Any
end

struct JuliaNode <: RegularNode
    # contains symbols, some of which resolve to the local scope
    expr::Expr

    # map from local scope symbols to node
    inputs::Dict{Symbol,RegularNode}

    # variable name (from the LHS of assignment or defaults to a gensym)
    name::Symbol

    # optional declared type (from LHS of assignment; default is Any)
    typ::Type
end

struct RandomChoiceNode <: RegularNode
    dist::Distribution
    inputs::Vector{RegularNode}
    addr::Symbol

    # variable name (from the LHS of assignment or defaults to a gensym)
    name::Symbol

    # optional declared type (default is Any)
    typ::Type
end

struct GenerativeFunctionCallNode <: RegularNode
    generative_function::GenerativeFunction
    inputs::Vector{RegularNode}
    addr::Symbol
    argdiff::StaticIRNode

    # variable name (from the LHS of assignment or defaults to a gensym)
    name::Symbol

    # optional declared type (default is Any)
    typ::Type
end

##############
# diff nodes #
##############

struct DiffJuliaNode <: DiffNode
    # contains gensyms for each symbol in local scope
    expr::Expr

    # map from local scope symbols to node
    # (may be regular or diff nodes)
    inputs::Dict{Symbol,StaticIRNode}

    # variable name (from the LHS of assignment; defaults to a gensym)
    name::Symbol

    # optional declared type (default is Any)
    typ::Type
end

struct ReceivedArgDiffNode <: DiffNode
    # variable name (from the LHS of assignment; defaults to a gensym)
    name::Symbol

    # declared type (default is Any)
    # this is just a type assertion (not stored in the trace)
    typ::Type

    # syntax:
    # @diff x::T = @argdiff()

    # defaults to declared_type = Any
    # @diff _::Any = @argdiff()
end

struct ChoiceDiffNode <: DiffNode
    choice_node::RandomChoiceNode

    # variable name (from the LHS of assignment; defaults to a gensym)
    name::Symbol

    # declared type (default is Any)
    # this is just a type assertion (not stored in the trace)
    typ::Type
end

struct CallDiffNode <: DiffNode
    call_node::GenerativeFunctionCallNode

    # variable name (from the LHS of assignment; defaults to a gensym)
    name::Symbol

    # declared type (default is Any)
    # this is just a type assertion (not stored in the trace)
    typ::Type
end

###############################
# intermediate representation #
###############################

struct StaticIR
    #nodes::Set{StaticIRNode} # ? needed ?
    arg_nodes::Vector{ArgumentNode}
    choice_nodes::Vector{RandomChoiceNode}
    call_nodes::Vector{GenerativeFunctionCallNode}
    return_node::RegularNode
    #retdiff_node::StaticIRNode
    #received_argdiff_node::ReceivedArgDiffNode
end
