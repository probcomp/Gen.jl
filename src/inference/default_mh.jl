function default_mh(selection::AddressSet, trace)
    args = get_args(trace)
    (new_trace, weight) = free_update(args, noargdiff, trace, selection)
    if log(rand()) < weight
        # accept
        return (new_trace, true)
    else
        # reject
        return (trace, false)
    end
end

export default_mh
