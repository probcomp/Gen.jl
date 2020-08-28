struct PFCombinatorTrace
    args::Tuple
    gen_fn::Any
    emission_choices::DynamicChoiceMap
    log_ml_est::Float64
end

Gen.get_args(tr::PFCombinatorTrace) = tr.args
Gen.get_retval(tr::PFCombinatorTrace) = nothing
Gen.get_gen_fn(tr::PFCombinatorTrace) = tr.gen_fn
Gen.get_score(tr::PFCombinatorTrace) = tr.log_ml_est
Gen.get_choices(tr::PFCombinatorTrace) = tr.emission_choices

struct ParticleFilterCombinator <: GenerativeFunction{Any,PFCombinatorTrace}
    combined::GenerativeFunction
    num_particles::Int
end

function ParticleFilterCombinator(init, dynamics, emission, num_particles::Int)

    @gen function kernel(t::Int, prev_state,
            dynamics_args::Tuple, emission_args::Tuple)
        state = @trace(dynamics(t+1, prev_state, dynamics_args...), :dynamics)
        @trace(emission(t+1, state, emission_args...), :emission)
        return state
    end

    chain = Unfold(kernel)

    @gen function combined(T::Int, init_args::Tuple, dynamics_args::Tuple, emission_args::Tuple)
        init_state = @trace(init(init_args...), :init)
        @trace(emission(1, init_state, emission_args...), :init_emission)
        @trace(chain(T, init_state, dynamics_args, emission_args...), :chain, @argdiff())
    end

    ParticleFilterCombinator(combined, num_particles)
end


function run_particle_filter(gen_fn::ParticleFilterCombinator, args::Tuple, choices::ChoiceMap, num_particles::Int)
    (T::Int, initial_args, dynamics_args, emission_args) = args
    init_obs = choicemap()
    set_submap!(init_obs, :init_emission, get_submap(choices, 1))
    init_args = (0, initial_args, dynamics_args, emission_args)
    state = initialize_particle_filter(gen_fn.combined,
        init_args, init_obs, num_particles)
    argdiff = UnfoldCustomArgDiff(false, false)
    for t=2:T
        new_args = (t-1, initial_args, dynamics_args, emission_args)
        maybe_resample!(state)
        obs = choicemap()
        set_submap!(obs, :chain => t-1 => :emission, get_submap(choices, t))
        particle_filter_step!(state, new_args, argdiff, obs)
    end
    log_ml_estimate(state)
end

function Gen.generate(gen_fn::ParticleFilterCombinator, args::Tuple, choices::ChoiceMap)
    log_ml_est = run_particle_filter(gen_fn, args, choices, gen_fn.num_particles)
    trace = PFCombinatorTrace(args, gen_fn, choices, log_ml_est)
    (trace, log_ml_est)
end

function Gen.update(trace::PFCombinatorTrace, args::Tuple, argdiff, choices::ChoiceMap)
    if !isempty(choices)
        error("Not implemented")
    end
    gen_fn = get_gen_fn(trace)
    new_log_ml_est = run_particle_filter(gen_fn, args, trace.emission_choices, gen_fn.num_particles)
    new_trace = PFCombinatorTrace(args, gen_fn, trace.emission_choices, new_log_ml_est)
    weight = new_log_ml_est - trace.log_ml_est
    (new_trace, weight, DefaultRetDiff(), EmptyChoiceMap())
end

function Gen.regenerate(trace::PFCombinatorTrace, args::Tuple, argdiff, selection::Selection)
    if !isempty(selection)
        error("Not implemented")
    end
    gen_fn = get_gen_fn(trace)
    new_log_ml_est = run_particle_filter(gen_fn, args, trace.emission_choices, gen_fn.num_particles)
    new_trace = PFCombinatorTrace(args, gen_fn, trace.emission_choices, new_log_ml_est)
    weight = new_log_ml_est - trace.log_ml_est
    (new_trace, weight, DefaultRetDiff(), EmptyChoiceMap())
end

function Gen.extend(::PFCombinatorTrace, ::Tuple, argdiff, ::ChoiceMap)
    error("Not implemented")
end

function Gen.choice_gradients(::PFCombinatorTrace, ::Selection)
    error("Not implemented")
end

function Gen.accumulate_param_gradients!(::PFCombinatorTrace,
    retgrad, scale_factor)
    error("Not implemented")
end

function Gen.propose(::ParticleFilterCombinator, args::Tuple)
    error("Not implemented")
end

function Gen.assess(::ParticleFilterCombinator, args::Tuple, choices::ChoiceMap)
    error("Not implemented")
end

function Gen.project(tr::PFCombinatorTrace, ::EmptySelection)
    # q(t; u=empty, x) = simulate forward from model
    # q(r; x, t) = run sMC
    # therefore, the result is SMC-ML-est / forward probability
    #tr.log_ml_est - tr.model_project
    0.
end
