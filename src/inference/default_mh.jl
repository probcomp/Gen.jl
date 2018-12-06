function default_mh(model::GenerativeFunction, selection::AddressSet, trace; verbose=false)
    model_args = get_call_record(trace).args
    (new_trace, weight) = regenerate(model, model_args, noargdiff, trace, selection)
    if log(rand()) < weight
        verbose && println("accept")
        # accept
        return new_trace
    else
        verbose && println("reject")
        # reject
        return trace
    end
end

export default_mh
