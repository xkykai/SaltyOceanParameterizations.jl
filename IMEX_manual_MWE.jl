using LinearAlgebra
using DiffEqBase
import SciMLBase
using Lux, ComponentArrays, Random
using Optimization
using Printf
using Enzyme

n = 32
zC = collect(1.:1:32)
Δz = zC[2] - zC[1]

rng = Random.default_rng()

NN = Chain(Dense(33, 100, leakyrelu), Dense(100, 31))
ps, st = Lux.setup(rng, NN)

ps = ps |> ComponentArray .|> Float64
# ps .*= 0

truth_data = rand(31)
u₀ = rand(33)

NN(u₀, ps, st)

function one_forward_pass_loss(ps, truth_data, u₀, NN, st)
    pred = NN(u₀, ps, st)[1]
    return sum(abs2, pred .- truth_data)
end

dps = deepcopy(ps) .= 0
autodiff(Enzyme.Reverse, one_forward_pass_loss, Active, Duplicated(ps, dps), Const(truth_data), Const(u₀), Const(NN), Const(st))

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

const D_center = Dᶜ(n, Δz)
const D_face = Dᶠ(n, Δz)

u0 = collect(range(0, 1, length=n))

function compute_diffusivity(∂u∂z)
    return ifelse(∂u∂z < 0, 0.2, 1e-5)
end

α_initial = compute_diffusivity.(D_face * u0)

D = Tridiagonal(D_center * (α_initial .* D_face))

const params = (top=3e-3, bottom=0., NN=NN, st=st, f=1e-4)

function rhs(u, p, params)
    x′ = vcat(u, params.f)

    residual_flux = vcat(params.bottom, first(params.NN(x′, p, params.st)), params.top)

    du = -D_center * residual_flux
    return du
end

function solve_equation(ps, params, u0)
    Δt = 1e-1
    sol = zeros(32, 100)
    sol[:, 1] .= u0
    u = deepcopy(u0)

    for i in 2:100
        ∂u∂x = D_face * u
        α = compute_diffusivity.(∂u∂x)
        D = Tridiagonal(D_center * (α .* D_face))
        # D = D_center * (α .* D_face)

        RHS = rhs(u, ps, params)

        sol[:, i] .= (I - Δt .* D) \ (u .+ Δt .* RHS)
        u .= sol[:, i]
    end

    return sol[:, 1:10:100]
    # return sol
end

truth = rand(32, 10)

solve_equation(ps, params, u0)

function loss_implicitstep(ps, truth, params, u0)
    return sum(abs2, solve_equation(ps, params, u0) .- truth)
end

loss_implicitstep(ps, truth, params, u0)

dps = deepcopy(ps) .= 0
autodiff(Enzyme.Reverse, loss_implicitstep, Active, Duplicated(ps, dps), Const(truth), Const(params), Duplicated(u0, deepcopy(u0)))

truths = [rand(32, 10) for _ in 1:2]
u0s = [rand(32) for _ in 1:2]

function loss_multipleics(ps, truths, params, u0s)
    losses = [sum(abs2, solve_equation(ps, params, u0) - truth) for (truth, u0) in zip(truths, u0s)]
    return sum(losses)
end

loss_multipleics(ps, truths, params, u0s)

dps = deepcopy(ps) .= 0
autodiff(Enzyme.Reverse, loss_multipleics, Active, DuplicatedNoNeed(ps, dps), DuplicatedNoNeed(truths, deepcopy(truths)), Const(params), DuplicatedNoNeed(u0s, deepcopy(u0s)))

# function solve_equation(ps, params)
#     Δt = 1e-1
#     t = 0.
#     index = 2
#     times = 0.:10:50
#     sol = zeros(n, 6)
#     sol[:, 1] .= u0

#     u = deepcopy(u0)

#     for i in 1:500
#         t += Δt

#         ∂u∂x = D_face * u
#         α = compute_diffusivity.(∂u∂x)
#         D = Tridiagonal(D_center * (α .* D_face))
#         RHS = rhs(u, ps, params)

#         u = (I - Δt .* D) \ (u .+ Δt .* RHS)

#         if i % 100 == 0
#             sol[:, index] = u
#             index += 1
#         end
#     end
#     return sol
# end

# sol = rand(32, 6)
# truth = rand(size(sol)...)

# loss(ps, truth, params) = sum(abs2, solve_equation(ps, params) - truth)

# loss(ps_training, truth, params)

# autodiff(Enzyme.Reverse, loss, Active, Duplicated(ps_training, copy(ps_training)), Const(truth), Const(params))

# #%%
# using CairoMakie
# fig = Figure()
# ax = CairoMakie.Axis(fig[1, 1])
# lines!(sol[:, end], zC, label="Final")
# lines!(u0, zC, label="Initial")
# axislegend(position=:lb)
# display(fig)
# #%%

# iter = 0
# maxiter = 3

# callback = function (p, l)
#     @printf("loss total %6.10e\n", l,)
#     return false
# end

# loss(ps_training)

# adtype = Optimization.AutoForwardDiff()
# optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
# optprob = Optimization.OptimizationProblem(optf, ps_training)
# res = Optimization.solve(optprob, OptimizationOptimisers.Adam(1e-5), callback=callback, maxiters=maxiter)