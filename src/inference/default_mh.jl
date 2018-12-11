function default_mh(trace, selection::AddressSet)
    args = get_args(trace)
    (new_trace, weight) = free_update(args, noargdiff, trace, selection)
    if log(rand()) < weight
        # accept
        return new_trace
    else
        # reject
        return trace
    end
end

export default_mh
