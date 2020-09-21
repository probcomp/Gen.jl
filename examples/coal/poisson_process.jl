# piecewise homogenous Poisson process

# n intervals - n + 1 bounds
# (b_1, b_2]
# (b_2, b_3]
# ..
# (b_n, b_{n+1}]

function compute_total(bounds, rates)
    num_intervals = length(rates)
    if length(bounds) != num_intervals + 1
        error("Number of bounds does not match number of rates")
    end
    total = 0.
    bounds_ascending = true
    for i=1:num_intervals
        lower = bounds[i]
        upper = bounds[i+1]
        rate = rates[i]
        len = upper - lower
        if len <= 0
            bounds_ascending = false
        end
        total += len * rate
    end
    (total, bounds_ascending)
end

struct PiecewiseHomogenousPoissonProcess <: Distribution{Vector{Float64}} end
const piecewise_poisson_process = PiecewiseHomogenousPoissonProcess()

function Gen.logpdf(::PiecewiseHomogenousPoissonProcess, x::Vector{Float64}, bounds::Vector{Float64}, rates::Vector{Float64})
    cur = 1
    upper = bounds[cur+1]
    lpdf = 0.
    for xi in sort(x)
        if xi < bounds[1] || xi > bounds[end]
            error("x ($xi) lies outside of interval")
        end
        while xi > upper
            cur += 1
            upper = bounds[cur+1]
        end
        lpdf += log(rates[cur])
    end
    (total, bounds_ascending) = compute_total(bounds, rates)
    if bounds_ascending
        lpdf - total
    else
        -Inf
    end
end

function Gen.random(::PiecewiseHomogenousPoissonProcess, bounds::Vector{Float64}, rates::Vector{Float64})
    x = Vector{Float64}()
    num_intervals = length(rates)
    for i=1:num_intervals
        lower = bounds[i]
        upper = bounds[i+1]
        rate = (upper - lower) * rates[i]
        n = random(poisson, rate)
        for j=1:n
            push!(x, random(uniform_continuous, lower, upper))
        end
    end
    x
end
