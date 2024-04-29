# [Hidden Markov Models] (@id hmm)

```@example hmm_tutorial
using Gen
using GLMakie
import GLMakie: scatter!, plot
using LinearAlgebra
GLMakie.activate!(inline=true)
```

```@example hmm_tutorial
@gen function kernel(n, latent)
    x ~ normal(latent, 1.0) 
    y ~ normal(x, 1.0)
    return y
end

hmm_model = Unfold(kernel)
```

# Plot this trajectory multiple times...
```@example hmm_tutorial
traces = [simulate(hmm_model, (0,0)) for i in 1:1000]
```

```@example hmm_tutorial
struct EField
    positions::Matrix{Float64}
    charges::Vector{Float64}
    k::Float64
    function EField(positions, charges, k=1.0)
        size(positions, 1) != length(charges) && throw("Number of positions does not match number of charges.")
        new(positions, charges, k)
    end
end

function force(field::EField, charge::Float64, particle::AbstractVector)
    r_vec = field.positions .- particle'
    r = sqrt.(sum(r_vec .^ 2; dims=2)).^3
    forces = field.charges * charge
    forces = r_vec .* (forces ./ r )
    total = sum(forces; dims=1)
    -vec(total)
end

function (field::EField)(charge, position)
    force(field, charge, position)  
end
```

```@example hmm_tutorial
positions = [
    1.0 0;
    0.0 1.0;
    -1.0 0;
    0 -1.0
]
charges = [
    1.0, 
    1.0,
    1.0,
    1.0
]
field = EField(positions, charges)
forces = force(field, 1.0, [0.0, 1.000])
field(1.0, [0,0])
```

```@example hmm_tutorial
scatter!(ax, field::EField) = scatter!(ax, field.positions[:,1], field.positions[:,2], marker=:circle, markersize=20)

function plot(ax,field::EField)
    scatter!(ax, field)
    if length(field.charges) == 1 # Set lengths manually
        x, y = field.positions[1,:]
        x_min, x_max = x-0.5, x+0.5
        y_min, y_max = y-0.5, y+0.5
    else
        x_min, y_min = minimum(field.positions; dims=1)
        x_max, y_max = maximum(field.positions; dims=1)
    end
    xs = range(x_min, x_max, 10)
    ys = range(y_min, y_max, 10)
    charges = [field(1.0, [x, y]) for x in xs, y in ys]
    normalize!.(charges)
    charges = stack(charges) ./ 10
    u = charges[1,:,:]
    v = charges[2,:,:]

    arrows!(ax, xs, ys, u, v)
end
```

```@example hmm_tutorial
function trajectory(field, charge, position; N=1, dt=0.01, velocity = [0.00, 0.00])
    init = position
    path = Matrix{Float64}(undef, N, 2)
    path[1,:] = init

    for i in 2:N
        a = field(charge, init)
        # println("a ", a)
        velocity = velocity + dt*a
        # println("v ", v)
        init = 0.5 * dt^2 * a + velocity * dt + init
        path[i,:] = init
    end
    path
end
```

```@example hmm_tutorial
f = Figure(size=(500,500))
ga = f[1,1] = GridLayout()
ax = Axis(ga[1,1])
plot(ax, field)
# display(trajectory(field, 0.1, [0.0, 0.5],20))
traj = trajectory(field, -1.0, [0.0, 0.0], N=300, velocity=[0.02, 0.01])
# println(traj)
lines!(ax, traj, color=:red)
f
```