########
# MCMC #
########

# TODO add discard_proto as an argument?
function mh(model::Generator, proposal::Generator, proposal_args::Tuple, trace)
    model_args = get_call_record(trace).args
    forward_trace = simulate(proposal, proposal_args, Some(get_choices(trace)))
    forward_score = get_call_record(forward_trace).score
    constraints = get_choices(forward_trace)
    (new_trace, weight, discard) = update(
        model, model_args, NoChange(), trace, constraints)
    backward_trace = assess(proposal, proposal_args, discard, Some(get_choices(new_trace)))
    backward_score = get_call_record(backward_trace).score
    if log(rand()) < weight - forward_score + backward_score
        # accept
        return new_trace
    else
        # reject
        return trace
    end
end

function mh(model::Generator, selector::SelectionFunction, selector_args::Tuple, trace)
    (selection, _) = select(selector, selector_args, get_choices(trace))
    model_args = get_call_record(trace).args
    (new_trace, weight) = regenerate(model, model_args, NoChange(), trace, selection)
    if log(rand()) < weight
        # accept
        return new_trace
    else
        # reject
        return trace
    end
end

# TODO add mh where we use a separate forward and backward proposal?
# TODO add mh where we use resimulation_update() (i.e. propose constrained changes)

#const OBS = "obs"
const MODEL = "model"
const PROPOSAL = "prop"

export MODEL, PROPOSAL

function rjmcmc(model, forward, forward_args, backward, backward_args,
                injective, injective_args, trace, correction)
    model_args = get_call_record(trace).args
    model_score = get_call_record(trace).score
    forward_trace = simulate(forward, forward_args, Some(get_choices(trace)))
    forward_score = get_call_record(forward_trace).score
    input = pair(get_choices(trace), get_choices(forward_trace), MODEL, PROPOSAL)
    (output, logabsdet) = apply(injective, injective_args, input)
    (model_constraints, backward_constraints) = unpair(output, MODEL, PROPOSAL)
    new_trace = assess(model, model_args, model_constraints)
    new_model_score = get_call_record(new_trace).score
    backward_trace = assess(backward, backward_args, backward_constraints, Some(get_choices(new_trace)))
    backward_score = get_call_record(backward_trace).score
    alpha = new_model_score - model_score - forward_score + backward_score + logabsdet + correction(new_trace)
    if log(rand()) < alpha
        # accept
        return new_trace
    else
        # reject
        return trace
    end
end

export mh, rjmcmc


#########################################
## maximum a-posteriori (MAP) inference #
#########################################

# TODO add custom gradient/values proto trie as an argument?
"""
Backtracking gradient ascent for MAP inference on selected real-valued choices
"""
function map_optimize(model::Generator, selector::SelectionFunction, selector_args::Tuple,
                      trace; max_step_size=0.1, tau=0.5, min_step_size=1e-16, verbose=false)
    model_args = get_call_record(trace).args
    (selection, _) = select(selector, selector_args, get_choices(trace))
    (_, values, gradient) = backprop_trace(model, trace, selection, nothing)
    values_vec = to_array(values)
    gradient_vec = to_array(gradient)
    step_size = max_step_size
    score = get_call_record(trace).score
    while true
        new_values_vec = values_vec + gradient_vec * step_size
        from_array!(values, new_values_vec)
        # TODO discard and weight are not actually needed, there should be a more specialized variant
        (new_trace, _, discard, _) = update(model, model_args, NoChange(), trace,
                                            values, HomogenousTrie{Any,Float64}())
        new_score = get_call_record(new_trace).score
        change = new_score - score
        if verbose
            println("step_size: $step_size, prev score: $score, new score: $new_score, change: $change")
        end
        if change >= 0.
            # it got better, return it
            return new_trace
        elseif step_size < min_step_size
            # it got worse, but we ran out of attempts
            return trace
        end
        
        # try again with a smaller step size
        step_size = tau * step_size
    end
end

# TODO what if we want to do MAP optimization over a transformed variant?
# e.g. for a value that is non-negative (a variance)
# we coul map optimize over all of R and then take an exp()
# answer: use a normal() and then take exp(), so it is log-normal..

export map_optimize

#
#
## names of namespaces
#
#export OBS, MODEL, PROP
#
#function discard_error(discard)
    #error("Some addresses were updated or deleted; got discard trace: $discard")
#end
#
#############################
## Markov Chain Monte Carlo #
#############################
#
#"""
#Resimulation Metropolis-Hastings
#"""
#function mh(model::GenFunction, model_args::Tuple,
            #selector::SelectionFunction, selector_args::Tuple,
            #trace::Trace, score::Float64, aux::Trace)
    #(selection, _) = select(selector, selector_args, trace)
    #regenerate(model, model_args, model_args, selection, trace, aux)
#end
#
#"""
#Metropolis-Hastings using a custom proposal expressed as a generative function
#"""
#function mh_custom(model::GenFunction, model_args::Tuple,
             #proposal::GenFunction, proposal_args::Tuple,
             #prev_model_trace::Trace, prev_model_score::Real)
    #mh_back(model, model_args, proposal, proposal_args,
        #proposal, proposal_args, prev_model_trace, prev_model_score)
#end
#
#"""
#Metropolis-Hastings with different forward and backward proposals
#"""
#function mh_back(model::GenFunction, model_args::Tuple,
             #forward::GenFunction, forward_args::Tuple,
             #backward::GenFunction, backward_args::Tuple,
             #prev_model_trace::Trace, prev_model_score::Real)
#
    ## run forward
    #(forward_trace, forward_score, _) = simulate(
        #forward, forward_args, prev_model_trace)
#
    ## update model trace
    #(new_model_trace, new_model_score, aux, discard_trace, log_weight, val) = update(
        #model, model_args, prev_model_trace, forward_trace)
#
    ## assess backward proposal probability
    #(backward_score, _) = assess(
        #backward, backward_args, discard_trace, new_model_trace)
#
    ## MH acceptance ratio
    #alpha = log_weight - prev_model_score + backward_score - forward_score
#
    #(new_model_trace, new_model_score, aux, alpha, val)
#end
#
#"""
#General (reversible-jump) Metropolis-Hastings
#"""
#function mh_trans(model::GenFunction, model_args::Tuple,
             #forward::GenFunction, forward_args::Tuple,
             #backward::GenFunction, backward_args::Tuple,
             #injection::InjectiveFunction, injection_args::Tuple,
             #prev_model_trace::Trace, prev_model_score::Float64)
#
    ## TODO asymptotically more efficient:
    ## 1) simulate forward proposal, and merge trace with model trace
    ## 2) generate update using transform (this is some small choice trie, not a whole trace)
    ## 3) run update (without arg change) on the model with small choice trie as constraints
    ## 4) assess discard (addresses overwritten or deleted) using backwards kernel
    ## addresses that are retained and not changed are kept (no full trace copy necessary)
    ## TODO the only problem: how do we compute the Jacobian determinant before we
    ## know which addresses are to be copied and which are to be deleted? it
    ## seems like there should be an efficient solution to this problem.
#
    ## run forward
    #(forward_trace, forward_score, _) = simulate(
        #forward, forward_args, prev_model_trace)
#
    ## merge previous trace and forward trace
    #merged_trace = pair(prev_model_trace, forward_trace, MODEL, PROP)
#
    ## apply injection
    #(injection_trace, injection_score) = apply(injection, injection_args, merged_trace)
#
    ## split
    #(new_model_trace, backward_trace) = unpair(injection_trace, MODEL, PROP)
#
    ## assess backward
    #(backward_score, _) = assess(
        #backward, backward_args, backward_trace, new_model_trace)
#
    ## assess new model score (NOTE: assess will return an actual trace, which we can then use later)
    #(new_score, aux, val) = assess(model, model_args, new_model_trace)
#
                        #
    ## MH acceptance ratio
    #alpha = (new_score - prev_model_score + backward_score - forward_score
                  #+ injection_score)
#
    #(new_model_trace, new_score, aux, alpha, val)
#end
#
#
#"""
#Accept or reject according to Metropolis-Hastings rule.
#"""
#function mh_accept_reject(alpha::Float64,
                          #prev_trace::Trace, prev_score::Float64, prev_aux::Trace, 
                          #new_trace::Trace, new_score::Float64, new_aux::Trace)
    #if log(rand()) <= alpha
        #(new_trace, new_score, new_aux, true)
    #else
        #(prev_trace, prev_score, prev_aux, false)
    #end
#end
#
#
###########################
## Sequential Monte Carlo #
###########################
#
#"""
#Sequential Monte Carlo using forward simulation as proposal.
#
#Trace must be a trace of the model for different arguments.
#The model may visit new addresses, but must visit all existing addresses.
#"""
#function smc(model::GenFunction, args::Tuple,
             #observes::Trace, trace::Trace, score::Float64)
    #extend(model, args, trace, observes)
#end
#
#function smc_assess(model::GenFunction, args::Tuple,
                    #observes::Trace, trace::Trace, score::Float64, new_trace::Trace)
    #ungenerate(model, args, observes, trace)
#end
#
#
#"""
#Sequential Monte Carlo with custom proposal.
#
#Trace must be a trace of the model for different arguments.
#The model may visit new addresses, but must visit all existing addresses.
#"""
#function smc_custom(model::GenFunction, args::Tuple,
              #proposal::GenFunction, proposal_args::Tuple, 
              #observes::Trace, trace::Trace, score::Float64)
#
    ## sample from proposal distribution, and compute proposal score
    #(proposal_trace, proposal_score, _) = simulate(proposal, proposal_args, trace)
#
    ## compute model score
    #new_trace = merge(observes, proposal_trace)
    #(new_score, aux, val) = assess(gf, args, new_trace)
#
    ## TODO use:
    ##constraints = merge(observes, proposal_trace)
    ##(new_trace, new_aux, new_score) = extend(gf, args, trace, aux, contraints)
#
    ## log incremental importance weight
    #log_weight = new_score - score - proposal_score
#
    #(new_trace, new_score, aux, log_weight, val)
#end
#
#
#"""
#Sequential Monte Carlo with custom proposal and custom backward kernel.
#
#Trace must be a trace of the model for different arguments.
#The model may visit new addresses.
#"""
#function smc_back(model::GenFunction, args::Tuple,
              #forward::GenFunction, forward_args::Tuple, 
              #backward::GenFunction, backward_args::Tuple, 
              #observes::Trace, trace::Trace, score::Float64)
    #forward_read_trace = pair(observes, trace, OBS, MODEL)
    #(forward_trace, forward_score, _) = simulate(forward, forward_args, forward_read_trace)
    #merged_trace = merge(observes, forward_trace)
    #(new_trace, backward_score) = project(backward, backward_args, merged_trace)
    #(new_score, aux, val) = assess(model, args, new_trace)
    #log_weight = regenerate_log_weight - proposal_score + backward_score
    #(new_trace, new_score, aux, log_weight, val)
#end
#
#
#"""
#General Sequential Monte Carlo with custom proposal, backward kernel, and
#change-of-variables injectionation.
#
#Trace may be a trace of a different model.
#The model may visit new addresses.
#"""
#function smc_trans(model::GenFunction, model_args::Tuple,
              #forward::GenFunction, forward_args::Tuple,
              #backward::GenFunction, backward_args::Tuple,
              #injection::InjectiveFunction, injection_args::Tuple,
              #observes::Trace, trace::Trace, score::Float64)
#
    ## run forward
    #forward_read_trace = pair(observes, trace, OBS, MODEL)
    #(forward_trace, forward_score, _) = simulate(forward, forward_args, forward_read_trace)
#
    ## merge previous trace and forward trace
    #merged_trace = pair(trace, forward_trace, MODEL, PROP)
#
    ## apply injection
    #(injection_trace, injection_score) = apply(injection, injection_args, merged_trace)
#
    ## split
    #(new_model_trace, backward_trace) = unpair(injection_trace, MODEL, PROP)
#
    ## assess backward
    #(backward_score, _) = assess(backward, backward_args, backward_trace, new_model_trace)
#
    ## add observes
    #new_model_trace = merge(new_model_trace, observes)
#
    ## assess new model score
    #(new_score, aux, val) = assess(model, model_args, new_model_trace)
                        #
    ## log importance weight
    #log_weight = (new_score - score + backward_score - forward_score
                  #+ injection_score)
#
    #(new_model_trace, new_score, aux, log_weight, val)
#end
#
#
########################
## Importance Sampling #
########################
#
#"""
#Do importance sampling using forward evaluation as the proposal.
#"""
#function imp(model::GenFunction, args::Tuple, observes::Trace)
    #generate(model, args, observes)
#end
#
#function imp_assess(model::GenFunction, args::Tuple, observes::Trace, trace::Trace)
    #ungenerate(model, args, observes, trace)
#end
#
#function imp_custom(model::GenFunction, model_args::Tuple, 
              #proposal::GenFunction, proposal_args::Tuple, 
              #observes::Trace)
#
    ## sample from importance distribution
    #(proposal_trace, proposal_score, _) = simulate(
        #proposal, proposal_args, observes)
#
    ## compute model score
    #trace = merge(proposal_trace, observes) # TODO how does this work with basic blocks?
    #(score, aux, val) = assess(model, model_args, trace)
#
    ## compute log importance weight
    #log_weight = score - proposal_score
    #
    #(trace, score, aux, log_weight, val)
#end
#
#
#function imp_back(model::GenFunction, model_args::Tuple, 
              #proposal::GenFunction, proposal_args::Tuple, 
              #backward::GenFunction, backward_args::Tuple, 
              #observes::Trace)
    #(proposal_trace, proposal_score) = simulate(proposal, proposal_score, observes)
    #merged = merge(observes, proposal_trace)
    #(trace, backward_score) = project(backward, backward_args, merged)
    #(score, aux, val) = assess(model, model_args, trace)
    #log_weight = score - proposal_score + backward_score
    #(trace, new_score, aux, log_weight, val)
#end
#
#
#function imp_trans(model::GenFunction, model_args::Tuple, 
              #proposal::GenFunction, proposal_args::Tuple, 
              #backward::GenFunction, backward_args::Tuple,
              #injection::InjectiveFunction, injection_args::Tuple,
              #observes::Trace)
    #(proposal_trace, proposal_score) = simulate(proposal, proposal_args, observes)
    #(injection_trace, injection_score) = apply(injection, injection_args, proposal_trace)
    #merged = merge(injection_trace, observes)
    #(trace, backward_score) = project(backward, backward_args, merged)
    #(score, aux, val) = assess(model, model_args, trace)
    #log_weight = (score + backward_score - proposal_score + injection_score)
    #(trace, score, aux, log_weight, val)
#end



#export mh, mh_custom, mh_back, mh_trans, mh_accept_reject
#export imp, imp_assess, imp_custom, imp_back, imp_trans
#export smc, smc_assess, smc_custom, smc_back, smc_trans
#export transform
