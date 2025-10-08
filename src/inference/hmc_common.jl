function sample_momenta(n::Int)
  Float64[random(normal, 0, 1) for _=1:n]
end

function assess_momenta(momenta)
  logprob = 0.
  for val in momenta
      logprob += logpdf(normal, val, 0, 1)
  end
  logprob
end

function add_choicemaps(a::ChoiceMap, b::ChoiceMap)
  out = choicemap()

  for (name, val) in get_values_shallow(a)
    out[name] = val + b[name]
  end

  for (name, submap) in get_submaps_shallow(a)
    out.internal_nodes[name] = add_choicemaps(submap, get_submap(b, name))
  end

  return out
end

function scale_choicemap(a::ChoiceMap, scale)
  out = choicemap()

  for (name, val) in get_values_shallow(a)
    out[name] = val * scale
  end

  for (name, submap) in get_submaps_shallow(a)
    out.internal_nodes[name] = scale_choicemap(submap, scale)
  end

  return out
end
