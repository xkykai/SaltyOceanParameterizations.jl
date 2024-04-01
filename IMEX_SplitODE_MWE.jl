using LinearAlgebra
using DiffEqBase
using OrdinaryDiffEq: SplitODEProblem, solve, IMEXEuler
import SciMLBase
using Lux, OptimizationOptimisers, ComponentArrays, Random, SciMLSensitivity
using Optimization
using Printf
import OrdinaryDiffEq

n = 32
zC = collect(1.:1:32)
Δz = zC[2] - zC[1]

rng = Random.default_rng()

NN = Chain(Dense(33, 10, leakyrelu), Dense(10, 31))
ps, st = Lux.setup(rng, NN)

ps = ps |> ComponentArray .|> Float64
ps .*= 0

function Dᶜ(N, Δ)
    D = zeros(N, N+1)
    for k in 1:N
        D[k, k]   = -1.0
        D[k, k+1] =  1.0
    end
    D .= 1/Δ .* D
    return D
end

function Dᶠ(N, Δ)
    D = zeros(N+1, N)
    for k in 2:N
        D[k, k-1] = -1.0
        D[k, k]   =  1.0
    end
    D .= 1/Δ .* D
    return D
end

D_center = Dᶜ(n, Δz)
D_face = Dᶠ(n, Δz)

u0 = collect(range(0, 1, length=n))

function compute_diffusivity(∂u∂z)
    return ifelse(∂u∂z < 0, 0.2, 1e-5)
end

α_initial = compute_diffusivity.(D_face * u0)

D = Tridiagonal(D_center * (α_initial .* D_face))

params = (top=3e-3, bottom=0., NN=NN, st=st, f=1e-4)

function rhs(u, p, t)
    x′ = vcat(u, params.f)

    residual_flux = vcat(params.bottom, first(params.NN(x′, p.ps, params.st)), params.top)

    du = -D_center * residual_flux
    return du
end

function update_diffusivity(A, u, p, t)
    ∂u∂z = D_face * u
    α = compute_diffusivity.(∂u∂z)
    return Tridiagonal(D_center * (α .* D_face))
end

D2 = SciMLBase.MatrixOperator(D, update_func=update_diffusivity)

ps_training = ComponentArray(;ps)

times = collect(0:0.1:1)
tspan = (times[1], times[end])

prob = SplitODEProblem(D2, rhs, u0, tspan, ps_training)
supertype(typeof(prob.f))

alg = IMEXEuler()
# alg = OrdinaryDiffEq.ImplicitEuler()
println("Solving...")
sol = solve(prob, alg, dt = 1e-3, saveat = times)

truth = rand(32, 11)

function loss(p)
    prob = SplitODEProblem(D2, rhs, u0, tspan, p)
    # sensealg = ForwardDiffSensitivity()
    sensealg = BacksolveAdjoint(autojacvec=false)
    # sensealg = InterpolatingAdjoint()
    sol = Array(solve(prob, alg; dt = 1e-3, saveat = times, sensealg))
    return sum(abs2, sol - truth)
end

iter = 0
maxiter = 3

callback = function (p, l)
    @printf("loss total %6.10e\n", l,)
    return false
end

loss(ps_training)

# adtype = Optimization.AutoZygote()
adtype = Optimization.AutoReverseDiff()
# adtype = Optimization.AutoForwardDiff()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ps_training)
# @time res = Optimization.solve(optprob, OptimizationOptimisers.Adam(1e-5), callback=callback, maxiters=20)

Optimization.solve(optprob, OptimizationOptimisers.Adam(1e-5), callback=callback, maxiters=2)
