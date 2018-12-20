#######################
# selection functions #
#######################

struct SelectionFunction
    julia_function::Function
end

macro sel(ast)
    if ast.head != :function
        error("syntax error at $(ast) in $(ast.head)")
    end
    if length(ast.args) != 2
        error("syntax error at $(ast) in $(ast.args)")
    end
    signature = ast.args[1]
    body = ast.args[2]
    if signature.head != :call
        error("syntax error at $(ast) in $(signature)")
    end
    function_name = signature.args[1]
    args = signature.args[2:end]
    escaped_args = map(esc, args)
    fn_args = [esc(state), escaped_args...]
    Expr(:block,
        Expr(:(=), 
            esc(function_name),
            Expr(:call, :SelectionFunction,
                Expr(:function, Expr(:tuple, fn_args...), esc(body)))))
end

macro select(addr)
    Expr(:call, :select_addr, esc(state), esc(addr))
end

struct SelectionState{T}
    assignment::T
    selection::DynamicAddressSet
    visitor::AddressVisitor
end

function SelectionState(assignment::T) where {T}
    SelectionState(assignment, DynamicAddressSet(), AddressVisitor())
end

function exec(fn::SelectionFunction, state::SelectionState, args::Tuple)
    fn.julia_function(state, args...)
end

# NOTE: unlike for @bijective and @gen functions, invoking @addr
# changes both the namespace of @read (and @select)

function addr(state::SelectionState, fn::SelectionFunction, args, addr)
    visit!(state.visitor, addr)
    if has_internal_node(state.assignment, addr)
        node = get_internal_node(state.assignment, addr)
    else
        node = EmptyAssignment()
    end
    if has_internal_node(state.selection, addr) || has_leaf_node(state.selection, addr)
        error("Address $addr, or an address under $addr, was already selected")
    end
    (selection, value) = select(fn, args, node)
    set_internal_node!(state.selection, addr, selection)
    value
end

function read(state::SelectionState, addr)
    get_leaf_node(state.assignment, addr)
end

function select_addr(state::SelectionState, addr)
    visit!(state.visitor, addr)
    push!(state.selection, addr)
end

function splice(state::SelectionState, fn::SelectionFunction, args::Tuple)
    exec(fn, state, args)
end

function select(fn::SelectionFunction, args::Tuple, assignment::T) where {T}
    state = SelectionState(assignment)
    value = exec(fn, state, args)
    (state.selection, value)
end


# alternative syntax for constructing selection from a flat list of addresses

function select(addrs...)
    set = DynamicAddressSet()
    for addr in addrs
        push!(set, addr)
    end
    return set
end

export @sel
export @select
export select
