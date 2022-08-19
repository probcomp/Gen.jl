@inline choice_gradients(trace::SwitchTrace{T}, selection::Selection, retval_grad) where T = choice_gradients(getfield(trace, :branch), selection, retval_grad)
@inline accumulate_param_gradients!(trace::SwitchTrace{T}, retval_grad, scale_factor = 1.) where {T} = accumulate_param_gradients!(getfield(trace, :branch), retval_grad, scale_factor)
