using LinearAlgebra: dot

struct Tree
  val_left :: ChoiceMap
  momenta_left :: ChoiceMap
  val_right :: ChoiceMap
  momenta_right :: ChoiceMap
  val_sample :: ChoiceMap
  n :: Int
  weight :: Float64
  stop :: Bool
  diverging :: Bool
end

function u_turn(values_left, values_right, momenta_left, momenta_right)
  return (dot(values_left - values_right, momenta_right) >= 0) &&
  (dot(values_right - values_left, momenta_left) >= 0)
end

function leapfrog(values_trie, momenta_trie, eps, integrator_state)
  selection, retval_grad, trace = integrator_state

  (trace, _, _) = update(trace, values_trie)
  (_, _, gradient_trie) = choice_gradients(trace, selection, retval_grad)

  # half step on momenta
  momenta_trie = add_choicemaps(momenta_trie, scale_choicemap(gradient_trie, eps / 2))

  # full step on positions
  values_trie = add_choicemaps(values_trie, scale_choicemap(momenta_trie, eps))

  # get new gradient
  (trace, _, _) = update(trace, values_trie)
  (_, _, gradient_trie) = choice_gradients(trace, selection, retval_grad)

  # half step on momenta
  momenta_trie = add_choicemaps(momenta_trie, scale_choicemap(gradient_trie, eps / 2))
  return values_trie, momenta_trie, get_score(trace)
end

function build_root(val, momenta, eps, direction, weight_init, integrator_state)
  val, momenta, lp = leapfrog(val, momenta, direction * eps, integrator_state)
  weight = lp + assess_momenta(to_array(momenta, Float64))

  diverging = weight - weight_init > 1000

  return Tree(val, momenta, val, momenta, val, 1, weight, false, diverging)
end

function merge_trees(tree_left, tree_right)
  # multinomial sampling
  if log(rand()) < tree_right.weight - tree_left.weight
    sample = tree_right.val_sample
  else
    sample = tree_left.val_sample
  end

  weight = logsumexp(tree_left.weight, tree_right.weight)
  n = tree_left.n + tree_right.n

  stop = tree_left.stop || tree_right.stop || u_turn(to_array(tree_left.val_left, Float64),
                                                     to_array(tree_right.val_right, Float64),
                                                     to_array(tree_left.momenta_left, Float64),
                                                     to_array(tree_right.momenta_right, Float64))
  diverging = tree_left.diverging || tree_right.diverging

  return Tree(tree_left.val_left, tree_left.momenta_left, tree_right.val_right,
    tree_right.momenta_right, sample, n, weight, stop, diverging)
end

function build_tree(val, momenta, depth, eps, direction, weight_init, integrator_state)
  if depth == 0
    return build_root(val, momenta, eps, direction, weight_init, integrator_state)
  end

  tree = build_tree(val, momenta, depth - 1, eps, direction, weight_init, integrator_state)

  if tree.stop || tree.diverging
    return tree
  end

  if direction == 1
    other_tree = build_tree(tree.val_right, tree.momenta_right, depth - 1, eps, direction,
                            weight_init, integrator_state)
    return merge_trees(tree, other_tree)
  else
    other_tree = build_tree(tree.val_left, tree.momenta_left, depth - 1, eps, direction,
                            weight_init, integrator_state)
    return merge_trees(other_tree, tree)
  end
end

"""
    (new_trace, sampler_statistics) = nuts(
        trace, selection::Selection;eps=0.1,
        max_treedepth=15, check=false, observations=EmptyChoiceMap())

Apply a Hamiltonian Monte Carlo (HMC) update with a No U Turn stopping criterion that proposes new values for the selected addresses, returning the new trace (which is equal to the previous trace if the move was not accepted) and a `Bool` indicating whether the move was accepted or not..

The NUT sampler allows for sampling trajectories of dynamic lengths, removing the need to specify the length of the trajectory as a parameter.
The sample will be returned early if the height of the sampled tree exceeds `max_treedepth`.

`sampler_statistics` is a struct containing the following fields:
    - depth: the depth of the trajectory tree
    - n: the number of samples in the trajectory tree
    - sum_alpha: the sum of the individual mh acceptance probabilities for each sample in the tree
    - n_accept: how many intermediate samples were accepted
    - accept: whether the sample was accepted or not

# References
Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. URL: https://doi.org/10.48550/arXiv.1701.02434
Hoffman, M. D., & Gelman, A. (2022). The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. URL: https://arxiv.org/abs/1111.4246
"""
function nuts(
  trace::Trace, selection::Selection; eps=0.1, max_treedepth=15,
  check=false, observations=EmptyChoiceMap())
    prev_model_score = get_score(trace)
    retval_grad = accepts_output_grad(get_gen_fn(trace)) ? zero(get_retval(trace)) : nothing

    # values needed for a leapfrog step
    (_, values_trie, _) = choice_gradients(trace, selection, retval_grad)

    momenta = sample_momenta(length(to_array(values_trie, Float64)))
    momenta_trie = from_array(values_trie, momenta)
    prev_momenta_score = assess_momenta(momenta)

    weight_init = prev_model_score + prev_momenta_score

    integrator_state = (selection, retval_grad, trace)

    tree = Tree(values_trie, momenta_trie, values_trie, momenta_trie, values_trie, 1, -Inf, false, false)

    direction = 0
    depth = 0
    stop = false
    while depth < max_treedepth
      direction = rand([-1, 1])

      if direction == 1 # going right
        other_tree = build_tree(tree.val_right, tree.momenta_right, depth, eps, direction,
                                weight_init, integrator_state)
        tree = merge_trees(tree, other_tree)
      else # going left
        other_tree = build_tree(tree.val_left, tree.momenta_left, depth, eps, direction,
                                weight_init, integrator_state)
        tree = merge_trees(other_tree, tree)
      end

      stop = stop || tree.stop || tree.diverging
      if stop
        break
      end
      depth += 1
    end

    (new_trace, _, _) = update(trace, tree.val_sample)
    check && check_observations(get_choices(new_trace), observations)

    # assess new model score (negative potential energy)
    new_model_score = get_score(new_trace)

    # assess new momenta score (negative kinetic energy)
    if direction == 1
      new_momenta_score = assess_momenta(to_array(tree.momenta_right, Float64))
    else
      new_momenta_score = assess_momenta(to_array(tree.momenta_left, Float64))
    end

    # accept or reject
    alpha = new_model_score + new_momenta_score - weight_init
    if log(rand()) < alpha
      return (new_trace, true)
    else
      return (trace, false)
    end
end

export nuts

