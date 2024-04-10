using LinearAlgebra
using DiffEqBase
import SciMLBase
using Lux, ComponentArrays, Random
using Optimization
using Printf
using Enzyme
using SaltyOceanParameterizations
using SaltyOceanParameterizations.DataWrangling
using SaltyOceanParameterizations: calculate_Ri, local_Ri_ν_convectivetanh_shearlinear, local_Ri_κ_convectivetanh_shearlinear
using Oceananigans
using JLD2
using SeawaterPolynomials.TEOS10
using CairoMakie
using SparseArrays
using Optimisers
using Printf
import Dates

function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

LES_FILE_DIRS = [
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0_QS_-3.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0002_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_0.0_QT_0.0_QS_-3.0e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_0.0_QT_0.0005_QS_0.0_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
]

BASECLOSURE_FILE_DIR = "./training_output/local_diffusivity_convectivetanh_shearlinear_rho_SW_FC_WWSC_SWWC/training_results_5.jld2"

field_datasets = [FieldDataset(FILE_DIR, backend=OnDisk()) for FILE_DIR in LES_FILE_DIRS]

ps_baseclosure = jldopen(BASECLOSURE_FILE_DIR, "r")["u"]

timeframes = [25:10:length(data["ubar"].times) for data in field_datasets]
full_timeframes = [25:length(data["ubar"].times) for data in field_datasets]
train_data = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, timeframes)
coarse_size = 32

# truths = [vcat(data.profile.T.scaled, data.profile.S.scaled, data.profile.ρ.scaled) for data in train_data.data]
truths = [vcat(data.profile.T.scaled, data.profile.S.scaled) for data in train_data.data]

train_data_plot = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, full_timeframes)

params = [(                   f = data.coriolis.unscaled,
                     f_scaled = data.coriolis.scaled,
                            τ = data.times[end] - data.times[1],
                        N_timesteps = length(data.times),
                    scaled_time = (data.times .- data.times[1]) ./ (data.times[end] - data.times[1]),
        scaled_original_time = (plot_data.times .- plot_data.times[1]) ./ (plot_data.times[end] - plot_data.times[1]),
                            zC = data.metadata["zC"],
                            H = data.metadata["original_grid"].Lz,
                            g = data.metadata["gravitational_acceleration"],
                    coarse_size = coarse_size, 
                            Dᶜ = Dᶜ(coarse_size, data.metadata["zC"][2] - data.metadata["zC"][1]),
                            Dᶠ = Dᶠ(coarse_size, data.metadata["zF"][3] - data.metadata["zF"][2]),
                        Dᶜ_hat = Dᶜ(coarse_size, data.metadata["zC"][2] - data.metadata["zC"][1]) .* data.metadata["original_grid"].Lz,
                        Dᶠ_hat = Dᶠ(coarse_size, data.metadata["zF"][3] - data.metadata["zF"][2]) .* data.metadata["original_grid"].Lz,
                            wT = (scaled = (top=data.flux.wT.surface.scaled, bottom=data.flux.wT.bottom.scaled),
                                unscaled = (top=data.flux.wT.surface.unscaled, bottom=data.flux.wT.bottom.unscaled)),
                            wS = (scaled = (top=data.flux.wS.surface.scaled, bottom=data.flux.wS.bottom.scaled),
                                unscaled = (top=data.flux.wS.surface.unscaled, bottom=data.flux.wS.bottom.unscaled)),
                        scaling = train_data.scaling,
                        ) for (data, plot_data) in zip(train_data.data, train_data_plot.data)]

rng = Random.default_rng(123)
wT_NN = Chain(Dense(67, 4, leakyrelu), Dense(4, 31))
wS_NN = Chain(Dense(67, 4, leakyrelu), Dense(4, 31))

ps_wT, st_wT = Lux.setup(rng, wT_NN)
ps_wS, st_wS = Lux.setup(rng, wS_NN)

ps_wT = ps_wT |> ComponentArray .|> Float64
ps_wS = ps_wS |> ComponentArray .|> Float64

ps_wT .*= 1e-5
ps_wS .*= 1e-5

x₀s = [vcat(data.profile.T.scaled[:, 1], data.profile.S.scaled[:, 1]) for data in train_data.data]

ps = ComponentArray(; wT=ps_wT, wS=ps_wS)
NNs = (wT=wT_NN, wS=wS_NN)
sts = (wT=st_wT, wS=st_wS)

function predict_residual_flux(T_hat, S_hat, ∂ρ∂z_hat, p, params, sts, NNs)
    x′_wT = vcat(T_hat, ∂ρ∂z_hat, params.wT.scaled.top, params.f_scaled)
    x′_wS = vcat(S_hat, ∂ρ∂z_hat, params.wS.scaled.top, params.f_scaled)
    
    wT = vcat(0, NNs.wT(x′_wT, p.wT, sts.wT)[1], 0)
    wS = vcat(0, NNs.wS(x′_wS, p.wS, sts.wS)[1], 0)

    return wT, wS
end

predict_residual_flux(x₀s[1][1:32], x₀s[1][33:64], rand(33), ps, params[1], sts, NNs)

function predict_boundary_flux(params)
    wT = vcat(fill(params.wT.scaled.bottom, params.coarse_size), params.wT.scaled.top)
    wS = vcat(fill(params.wS.scaled.bottom, params.coarse_size), params.wS.scaled.top)

    return wT, wS
end

predict_boundary_flux(params[1])

function predict_diffusivities(Ris, ps_baseclosure)
    νs = local_Ri_ν_convectivetanh_shearlinear.(Ris, ps_baseclosure.ν_conv, ps_baseclosure.ν_shear, ps_baseclosure.m, ps_baseclosure.ΔRi)
    κs = local_Ri_κ_convectivetanh_shearlinear.(νs, ps_baseclosure.Pr)
    return νs, κs
end

eos = TEOS10EquationOfState()
Ris = calculate_Ri(zeros(coarse_size), zeros(coarse_size), train_data.data[1].profile.T.unscaled[:, 1], train_data.data[1].profile.S.unscaled[:, 1], params[1].Dᶠ, params[1].g, eos.reference_density, clamp_lims=(-Inf, Inf))

predict_diffusivities(Ris, ps_baseclosure)

function solve_NDE(ps, params, x₀, ps_baseclosure, sts, NNs)
    eos = TEOS10EquationOfState()
    coarse_size = params.coarse_size
    timestep_multiple = 10
    Δt = (params.scaled_time[2] - params.scaled_time[1]) / timestep_multiple
    Nt_solve = (params.N_timesteps - 1) * timestep_multiple + 1
    Dᶜ_hat = params.Dᶜ_hat
    Dᶠ_hat = params.Dᶠ_hat
    Dᶠ = params.Dᶠ

    scaling = params.scaling
    τ, H = params.τ, params.H

    T_hat = deepcopy(x₀[1:coarse_size])
    S_hat = deepcopy(x₀[coarse_size+1:2*coarse_size])
    
    sol = zeros(2*coarse_size, Nt_solve)
    sol[:, 1] .= x₀

    for i in 2:Nt_solve
        T = inv(scaling.T).(T_hat)
        S = inv(scaling.S).(S_hat)

        ρ = TEOS10.ρ.(T, S, 0, Ref(eos))
        ∂ρ∂z_hat = scaling.∂ρ∂z.(Dᶠ * ρ)

        Ris = calculate_Ri(zeros(coarse_size), zeros(coarse_size), ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf))
        _, κs = predict_diffusivities(Ris, ps_baseclosure)

        D = Dᶜ_hat * (-κs .* Dᶠ_hat)

        wT_residual, wS_residual = predict_residual_flux(T_hat, S_hat, ∂ρ∂z_hat, ps, params, sts, NNs)
        wT_boundary, wS_boundary = predict_boundary_flux(params)

        LHS = -τ / H^2 .* D

        T_RHS = - τ / H * scaling.wT.σ / scaling.T.σ .* (Dᶜ_hat * (wT_boundary .+ wT_residual))
        S_RHS = - τ / H * scaling.wS.σ / scaling.S.σ .* (Dᶜ_hat * (wS_boundary .+ wS_residual))

        T_hat .= (I - Δt .* LHS) \ (T_hat .+ Δt .* T_RHS)
        S_hat .= (I - Δt .* LHS) \ (S_hat .+ Δt .* S_RHS)

        sol[1:coarse_size, i] .= T_hat
        sol[coarse_size+1:2*coarse_size, i] .= S_hat
    end

    return sol[:, 1:timestep_multiple:end]
end

sol = solve_NDE(ps, params[1], x₀s[1], ps_baseclosure, sts, NNs)

#%%
fig = Figure()
axT = CairoMakie.Axis(fig[1, 1], xlabel="T", ylabel="z")
axS = CairoMakie.Axis(fig[1, 2], xlabel="S", ylabel="z")

lines!(axT, sol[1:32, 1], params[1].zC, label="initial")
lines!(axT, sol[1:32, end], params[1].zC, label="final")
lines!(axT, truths[1][1:32, end], train_data.data[1].metadata["zC"], label="truth")

lines!(axS, sol[33:64, 1], params[1].zC, label="initial")
lines!(axS, sol[33:64, end], params[1].zC, label="final")
lines!(axS, truths[1][33:64, end], train_data.data[1].metadata["zC"], label="truth")

Legend(fig[2, :], axT, orientation=:horizontal)
display(fig)
#%%
function loss(ps, truth, params, x₀, ps_baseclosure, sts, NNs)
    sol = solve_NDE(ps, params, x₀, ps_baseclosure, sts, NNs)
    return sum(abs2, sol .- truth)
end

loss(ps, truths[1], params[1], x₀s[1], ps_baseclosure, sts, NNs)

dps = deepcopy(ps) .= 0
autodiff(Enzyme.ReverseWithPrimal, 
         loss, 
         Active, 
         Duplicated(ps, dps), 
         Const(truths[1]), 
         Const(params[1]), 
         Duplicated(x₀s[1], deepcopy(x₀s[1])), 
         Const(ps_baseclosure), 
         Const(sts), 
         Const(NNs))

function loss_multipleics(ps, truths, params, x₀s, ps_baseclosure, sts, NNs)
    losses = [sum(abs2, solve_NDE(ps, param, x₀, ps_baseclosure, sts, NNs) .- truth) for (truth, x₀, param) in zip(truths, x₀s, params)]
    return sum(losses)
end

loss_multipleics(ps, truths, params, x₀s, ps_baseclosure, sts, NNs)

autodiff(Enzyme.ReverseWithPrimal, 
         loss_multipleics, 
         Active, 
         Duplicated(ps, dps), 
         DuplicatedNoNeed(truths, deepcopy(truths)), 
         DuplicatedNoNeed(params, deepcopy(params)), 
         DuplicatedNoNeed(x₀s, deepcopy(x₀s)), 
         DuplicatedNoNeed(ps_baseclosure, deepcopy(ps_baseclosure)), 
         Const(sts), 
         Const(NNs))
#%%
rule = Optimisers.Adam()
opt_state = Optimisers.setup(rule, ps)
loss(ps, truths[1], params[1], x₀s[1], ps_baseclosure, sts, NNs)

opt_state, ps = Optimisers.update!(opt_state, ps, dps)

loss(ps, truths[1], params[1], x₀s[1], ps_baseclosure, sts, NNs)

function train_NDE(ps, params, ps_baseclosure, sts, NNs, truths; n_epochs=2, n_batches=2, rule=Optimisers.Adam())
    opt_state = Optimisers.setup(rule, ps)
    dps = deepcopy(ps) .= 0
    wall_clock = [time_ns()]
    maxiter = n_epochs * n_batches
    iter = 1
    for epoch in 1:n_epochs
        for batch in 1:n_batches
            _, l = autodiff(Enzyme.ReverseWithPrimal, 
                            loss, 
                            Active, 
                            Duplicated(ps, dps), 
                            Const(truths[batch]), 
                            Const(params[batch]), 
                            Duplicated(x₀s[batch], deepcopy(x₀s[batch])), 
                            Const(ps_baseclosure), 
                            Const(sts), 
                            Const(NNs))
            opt_state, ps = Optimisers.update!(opt_state, ps, dps)
            @printf("%s, Δt %s, iter %d/%d, loss total %6.10e, max NN weight %6.5e\n",
                     Dates.now(), prettytime(1e-9 * (time_ns() - wall_clock[1])), iter, maxiter, l, 
                     maximum(abs, ps))
            wall_clock = [time_ns()]
            iter += 1
            # @info loss(ps, train_data.data[batch].profile.ρ.scaled, params[batch], x₀s[batch], ps_baseclosure, st, NN)
            dps .= 0
        end
    end
    return ps
end

train_NDE(ps, params, ps_baseclosure, sts, NNs, truths; n_epochs=20, n_batches=1, rule=Optimisers.Adam(1e-4))