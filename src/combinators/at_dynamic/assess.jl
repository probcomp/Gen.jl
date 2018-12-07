function assess(gen::AtDynamic{T,U,K}, args, constraints) where {T,U,K}
    (key::K, kernel_args) = args
    if length(get_internal_nodes(constraints)) + length(get_leaf_nodes(constraints)) > 1
        error("Not all constraints were consumed")
    end
    kernel_constraints = get_internal_node(constraints, key)
    assess(gen.kernel, kernel_args, kernel_constraints)
end
