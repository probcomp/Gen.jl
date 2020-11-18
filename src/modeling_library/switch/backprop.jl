@inline choice_gradients(trace::SwitchTrace{T}, selection::Selection, retval_grad) where T = (nothing, choice_gradients(getfield(trace, :branch), selection, retval_grad)...)
@inline accumulate_param_gradients(trace::SwitchTrace{T}, retval_grad) where {T} = (nothing, accumulate_param_gradients(getfield(trace, :branch), retval_grad)...)
