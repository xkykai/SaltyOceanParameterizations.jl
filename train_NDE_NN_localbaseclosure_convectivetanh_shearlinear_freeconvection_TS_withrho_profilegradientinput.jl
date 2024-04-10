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
using Statistics

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

truths = [(; T=data.profile.T.scaled, S=data.profile.S.scaled, ρ=data.profile.ρ.scaled, ∂T∂z=data.profile.∂T∂z.scaled, ∂S∂z=data.profile.∂S∂z.scaled, ∂ρ∂z=data.profile.∂ρ∂z.scaled) for data in train_data.data]

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

x₀s = [(; T=data.profile.T.scaled[:, 1], S=data.profile.S.scaled[:, 1]) for data in train_data.data]

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

function predict_boundary_flux(params)
    wT = vcat(fill(params.wT.scaled.bottom, params.coarse_size), params.wT.scaled.top)
    wS = vcat(fill(params.wS.scaled.bottom, params.coarse_size), params.wS.scaled.top)

    return wT, wS
end

function predict_diffusivities(Ris, ps_baseclosure)
    νs = local_Ri_ν_convectivetanh_shearlinear.(Ris, ps_baseclosure.ν_conv, ps_baseclosure.ν_shear, ps_baseclosure.m, ps_baseclosure.ΔRi)
    κs = local_Ri_κ_convectivetanh_shearlinear.(νs, ps_baseclosure.Pr)
    return νs, κs
end

function solve_NDE(ps, params, x₀, ps_baseclosure, sts, NNs, timestep_multiple=10)
    eos = TEOS10EquationOfState()
    coarse_size = params.coarse_size
    Δt = (params.scaled_time[2] - params.scaled_time[1]) / timestep_multiple
    Nt_solve = (params.N_timesteps - 1) * timestep_multiple + 1
    Dᶜ_hat = params.Dᶜ_hat
    Dᶠ_hat = params.Dᶠ_hat
    Dᶠ = params.Dᶠ

    scaling = params.scaling
    τ, H = params.τ, params.H

    T_hat = deepcopy(x₀.T)
    S_hat = deepcopy(x₀.S)
    
    sol_T = zeros(coarse_size, Nt_solve)
    sol_S = zeros(coarse_size, Nt_solve)
    sol_ρ = zeros(coarse_size, Nt_solve)

    sol_T[:, 1] .= T_hat
    sol_S[:, 1] .= S_hat

    for i in 2:Nt_solve
        T = inv(scaling.T).(T_hat)
        S = inv(scaling.S).(S_hat)

        ρ = TEOS10.ρ.(T, S, 0, Ref(eos))
        ρ_hat = scaling.ρ.(ρ)
        sol_ρ[:, i-1] .= ρ_hat

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

        sol_T[:, i] .= T_hat
        sol_S[:, i] .= S_hat
    end

    sol_ρ[:, end] .= ρ = scaling.ρ.(TEOS10.ρ.(inv(scaling.T).(T_hat), inv(scaling.S).(S_hat), 0, Ref(eos)))

    return (; T=sol_T[:, 1:timestep_multiple:end], S=sol_S[:, 1:timestep_multiple:end], ρ=sol_ρ[:, 1:timestep_multiple:end])
end

sol_T, sol_S, sol_ρ = solve_NDE(ps, params[1], x₀s[1], ps_baseclosure, sts, NNs)

#%%
function individual_loss(ps, truth, params, x₀, ps_baseclosure, sts, NNs)
    Dᶠ = params.Dᶠ
    scaling = params.scaling
    sol_T, sol_S, sol_ρ = solve_NDE(ps, params, x₀, ps_baseclosure, sts, NNs)

    T_loss = mean((sol_T .- truth.T).^2)
    S_loss = mean((sol_S .- truth.S).^2)
    ρ_loss = mean((sol_ρ .- truth.ρ).^2)

    T = inv(scaling.T).(sol_T)
    S = inv(scaling.S).(sol_S)
    ρ = inv(scaling.ρ).(sol_ρ)

    ∂T∂z = scaling.∂T∂z.(Dᶠ * T)
    ∂S∂z = scaling.∂S∂z.(Dᶠ * S)
    ∂ρ∂z = scaling.∂ρ∂z.(Dᶠ * ρ)

    ∂T∂z_loss = mean((∂T∂z .- truth.∂T∂z).^2)
    ∂S∂z_loss = mean((∂S∂z .- truth.∂S∂z).^2)
    ∂ρ∂z_loss = mean((∂ρ∂z .- truth.∂ρ∂z).^2)

    return (; T=T_loss, S=S_loss, ρ=ρ_loss, ∂T∂z=∂T∂z_loss, ∂S∂z=∂S∂z_loss, ∂ρ∂z=∂ρ∂z_loss)
end

function loss(ps, truth, params, x₀, ps_baseclosure, sts, NNs, losses_prefactor=(; T=1, S=1, ρ=1, ∂T∂z=1, ∂S∂z=1, ∂ρ∂z=1))
    losses = individual_loss(ps, truth, params, x₀, ps_baseclosure, sts, NNs)
    return sum(values(losses) .* values(losses_prefactor))
end

function compute_loss_prefactor(individual_loss)
    T_loss, S_loss, ρ_loss, ∂T∂z_loss, ∂S∂z_loss, ∂ρ∂z_loss = values(individual_loss)

    T_prefactor = 1
    S_prefactor = T_loss / S_loss
    ρ_prefactor = T_loss / ρ_loss

    ∂T∂z_prefactor = 1
    ∂S∂z_prefactor = ∂T∂z_loss / ∂S∂z_loss
    ∂ρ∂z_prefactor = ∂T∂z_loss / ∂ρ∂z_loss

    profile_loss = T_prefactor * T_loss + S_prefactor * S_loss + ρ_prefactor * ρ_loss
    gradient_loss = ∂T∂z_prefactor * ∂T∂z_loss + ∂S∂z_prefactor * ∂S∂z_loss + ∂ρ∂z_prefactor * ∂ρ∂z_loss

    gradient_prefactor = profile_loss / gradient_loss

    ∂ρ∂z_prefactor *= gradient_prefactor
    ∂T∂z_prefactor *= gradient_prefactor
    ∂S∂z_prefactor *= gradient_prefactor

    return (T=T_prefactor, S=S_prefactor, ρ=ρ_prefactor, ∂T∂z=∂T∂z_prefactor, ∂S∂z=∂S∂z_prefactor, ∂ρ∂z=∂ρ∂z_prefactor)
end

ind_losses = [individual_loss(ps, truth, param, x₀, ps_baseclosure, sts, NNs) for (truth, x₀, param) in zip(truths, x₀s, params)]
ind_loss = (; T=sum([loss.T for loss in individual_losses]), 
              S=sum([loss.S for loss in individual_losses]), 
              ρ=sum([loss.ρ for loss in individual_losses]), 
              ∂T∂z=sum([loss.∂T∂z for loss in individual_losses]), 
              ∂S∂z=sum([loss.∂S∂z for loss in individual_losses]), 
              ∂ρ∂z=sum([loss.∂ρ∂z for loss in individual_losses]))

loss_prefactor = compute_loss_prefactor(ind_loss)

function loss_multipleics(ps, truths, params, x₀s, ps_baseclosure, sts, NNs, losses_prefactor=(; T=1, S=1, ρ=1, ∂T∂z=1, ∂S∂z=1, ∂ρ∂z=1))
    losses = [loss(ps, truth, param, x₀, ps_baseclosure, sts, NNs, losses_prefactor) for (truth, x₀, param) in zip(truths, x₀s, params)]
    return mean(losses)
end

function train_NDE_multipleics(ps, params, ps_baseclosure, sts, NNs, truths, x₀s; n_epochs=2, rule=Optimisers.Adam(), loss_prefactor=(; T=1, S=1, ρ=1, ∂T∂z=1, ∂S∂z=1, ∂ρ∂z=1))
    opt_state = Optimisers.setup(rule, ps)
    dps = deepcopy(ps) .= 0
    wall_clock = [time_ns()]
    maxiter = n_epochs
    iter = 1
    losses = zeros(n_epochs)
    for epoch in 1:n_epochs
        _, l = autodiff(Enzyme.ReverseWithPrimal, 
                        loss_multipleics, 
                        Active, 
                        Duplicated(ps, dps), 
                        DuplicatedNoNeed(truths, deepcopy(truths)), 
                        DuplicatedNoNeed(params, deepcopy(params)), 
                        DuplicatedNoNeed(x₀s, deepcopy(x₀s)), 
                        DuplicatedNoNeed(ps_baseclosure, deepcopy(ps_baseclosure)), 
                        Const(sts), 
                        Const(NNs),
                        Const(loss_prefactor))
        opt_state, ps = Optimisers.update!(opt_state, ps, dps)
        @printf("%s, Δt %s, iter %d/%d, loss total %6.10e, max NN weight %6.5e\n",
                    Dates.now(), prettytime(1e-9 * (time_ns() - wall_clock[1])), iter, maxiter, l, 
                    maximum(abs, ps))
        losses[epoch] = l
        wall_clock = [time_ns()]
        iter += 1
        dps .= 0
    end
    return ps
end

ps = train_NDE_multipleics(ps, params, ps_baseclosure, sts, NNs, truths, x₀s; n_epochs=20, rule=Optimisers.Adam(1e-3), loss_prefactor=loss_prefactor)

