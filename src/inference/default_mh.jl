function default_mh(model::GenerativeFunction{T,U}, selection::AddressSet,
                    trace::U) where {T,U}
    args = get_call_record(trace).args
    (new_trace, weight) = regenerate(model, args, noargdiff, trace, selection)
    if log(rand()) < weight
        # accept
        return (new_trace, true)
    else
        # reject
        return (trace, false)
    end
end

export default_mh
