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
using Colors
using ArgParse
using SeawaterPolynomials
import SeawaterPolynomials.TEOS10: s, ΔS, Sₐᵤ
s(Sᴬ) = Sᴬ + ΔS >= 0 ? √((Sᴬ + ΔS) / Sₐᵤ) : NaN

rng = Random.default_rng(123)

# FILE_DIR = "./final_training_output/finalfullrun_slightlylocalNN_dTSrho/NDE_enzyme_2layer_64_relu_1.0Sscaling/training_results_epoch20000_end285.jld2"
# ps = jldopen(FILE_DIR, "r")["u"]

# hidden_layer_size = 64
# N_hidden_layer = 2
# activation = relu
# NN_layers = vcat(Dense(12, hidden_layer_size, activation), [Dense(hidden_layer_size, hidden_layer_size, activation) for _ in 1:N_hidden_layer-1]..., Dense(hidden_layer_size, 1))

# wT_NN = Chain(NN_layers...)
# wS_NN = Chain(NN_layers...)

# ps_wT, st_wT = Lux.setup(rng, wT_NN)
# ps_wS, st_wS = Lux.setup(rng, wS_NN)

# ps_wT = ps_wT |> ComponentArray .|> Float64
# ps_wS = ps_wS |> ComponentArray .|> Float64

# ps_wT .= ps.wT
# ps_wS .= ps.wS

# NNs = (wT=wT_NN, wS=wS_NN)
# sts = (wT=st_wT, wS=st_wS)

TRAINING_LES_FILE_DIRS = [
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0005_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0001_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0_QS_-5.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0_QS_-2.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_0.0_QT_0.0005_QS_0.0_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_0.0_QT_0.0001_QS_0.0_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_0.0_QT_0.0_QS_-5.0e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_0.0_QT_0.0_QS_-2.0e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
]

INTERPOLATING_FILE_DIRS = [
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0004_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0003_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0_QS_-4.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0_QS_-3.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_0.0_QT_0.0004_QS_0.0_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_0.0_QT_0.0003_QS_0.0_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_0.0_QT_0.0_QS_-4.0e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_0.0_QT_0.0_QS_-3.0e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_0.0_QT_0.0003_QS_-3.0e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0003_QS_-3.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
]

training_field_datasets = [FieldDataset(FILE_DIR, backend=OnDisk()) for FILE_DIR in TRAINING_LES_FILE_DIRS]
interpolating_field_datasets = [FieldDataset(FILE_DIR, backend=OnDisk()) for FILE_DIR in INTERPOLATING_FILE_DIRS]

timeframes_training = [25:10:length(data["ubar"].times) for data in training_field_datasets]
full_timeframes_training = [25:length(data["ubar"].times) for data in training_field_datasets]

full_timeframes_interpolating = [25:length(data["ubar"].times) for data in interpolating_field_datasets]

training_data = LESDatasets(training_field_datasets, ZeroMeanUnitVarianceScaling, timeframes_training)
coarse_size = 32
training_data_plot = LESDatasets(training_field_datasets, ZeroMeanUnitVarianceScaling, full_timeframes_training)
interpolating_data_plot = LESDatasets(interpolating_field_datasets, ZeroMeanUnitVarianceScaling, full_timeframes_interpolating)
scaling = training_data.scaling

x₀s_training = [(; T=data.profile.T.scaled[:, 1], S=data.profile.S.scaled[:, 1]) for data in training_data.data]
x₀s_interpolating = [(; T=scaling.T.(data.profile.T.unscaled[:, 1]), S=scaling.S.(data.profile.S.unscaled[:, 1])) for data in interpolating_data_plot.data]

training_params = [(                   f = data.coriolis.unscaled,
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
                        scaling = scaling,
                        ) for (data, plot_data) in zip(training_data.data, training_data_plot.data)]

interpolating_params = [( f = data.coriolis.unscaled,
                        f_scaled = scaling.f(data.coriolis.unscaled),
                            τ = data.times[end] - data.times[1],
                        N_timesteps = length(data.times),
                    scaled_time = (data.times .- data.times[1]) ./ (data.times[end] - data.times[1]),
        scaled_original_time = (data.times .- data.times[1]) ./ (data.times[end] - data.times[1]),
                            zC = data.metadata["zC"],
                            H = data.metadata["original_grid"].Lz,
                            g = data.metadata["gravitational_acceleration"],
                    coarse_size = coarse_size, 
                            Dᶜ = Dᶜ(coarse_size, data.metadata["zC"][2] - data.metadata["zC"][1]),
                            Dᶠ = Dᶠ(coarse_size, data.metadata["zF"][3] - data.metadata["zF"][2]),
                        Dᶜ_hat = Dᶜ(coarse_size, data.metadata["zC"][2] - data.metadata["zC"][1]) .* data.metadata["original_grid"].Lz,
                        Dᶠ_hat = Dᶠ(coarse_size, data.metadata["zF"][3] - data.metadata["zF"][2]) .* data.metadata["original_grid"].Lz,
                            wT = (scaled = (top=scaling.wT(data.flux.wT.surface.unscaled), bottom=scaling.wT(data.flux.wT.bottom.unscaled)),
                                unscaled = (top=data.flux.wT.surface.unscaled, bottom=data.flux.wT.bottom.unscaled)),
                            wS = (scaled = (top=scaling.wS(data.flux.wS.surface.unscaled), bottom=scaling.wS(data.flux.wS.bottom.unscaled)),
                                unscaled = (top=data.flux.wS.surface.unscaled, bottom=data.flux.wS.bottom.unscaled)),
                        scaling = scaling,
                        ) for data in interpolating_data_plot.data]

# function predict_residual_flux(∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat, p, params, sts, NNs)
#     common_variables = vcat(params.wT.scaled.top, params.wS.scaled.top, params.f_scaled)

#     wT = zeros(params.coarse_size+1)
#     wS = zeros(params.coarse_size+1)

#     for i in 2:params.coarse_size
#         wT[i] = first(NNs.wT(vcat(∂T∂z_hat[i-1:i+1], ∂S∂z_hat[i-1:i+1], ∂ρ∂z_hat[i-1:i+1], common_variables), p.wT, sts.wT))[1]
#         wS[i] = first(NNs.wS(vcat(∂T∂z_hat[i-1:i+1], ∂S∂z_hat[i-1:i+1], ∂ρ∂z_hat[i-1:i+1], common_variables), p.wS, sts.wS))[1]
#     end

#     wT[1] = wT[2]
#     wS[1] = wS[2]

#     return wT, wS
# end

# function predict_boundary_flux(params)
#     wT = vcat(fill(params.wT.scaled.bottom, params.coarse_size), params.wT.scaled.top)
#     wS = vcat(fill(params.wS.scaled.bottom, params.coarse_size), params.wS.scaled.top)

#     return wT, wS
# end

# function predict_boundary_flux!(wT, wS, params)
#     wT[1:end-1] .= params.wT.scaled.bottom
#     wS[1:end-1] .= params.wS.scaled.bottom

#     wT[end] = params.wT.scaled.top
#     wS[end] = params.wS.scaled.top

#     return nothing
# end

# function solve_NDE(ps, params, x₀, sts, NNs, timestep, Nt, timestep_multiple=2)
#     eos = TEOS10EquationOfState()
#     coarse_size = params.coarse_size
#     Δt = timestep / timestep_multiple
#     Nt_solve = (Nt - 1) * timestep_multiple + 1
#     Dᶜ_hat = params.Dᶜ_hat
#     Dᶠ_hat = params.Dᶠ_hat
#     Dᶠ = params.Dᶠ

#     scaling = params.scaling
#     τ, H = params.τ, params.H

#     T_hat = deepcopy(x₀.T)
#     S_hat = deepcopy(x₀.S)
#     ρ_hat = zeros(coarse_size)

#     ∂T∂z_hat = zeros(coarse_size+1)
#     ∂S∂z_hat = zeros(coarse_size+1)
#     ∂ρ∂z_hat = zeros(coarse_size+1)

#     T = zeros(coarse_size)
#     S = zeros(coarse_size)
#     ρ = zeros(coarse_size)
    
#     T_RHS = zeros(coarse_size)
#     S_RHS = zeros(coarse_size)

#     wT_residual = zeros(coarse_size+1)
#     wS_residual = zeros(coarse_size+1)

#     wT_boundary = zeros(coarse_size+1)
#     wS_boundary = zeros(coarse_size+1)

#     sol_T = zeros(coarse_size, Nt_solve)
#     sol_S = zeros(coarse_size, Nt_solve)
#     sol_ρ = zeros(coarse_size, Nt_solve)

#     sol_T[:, 1] .= T_hat
#     sol_S[:, 1] .= S_hat

#     for i in 2:Nt_solve
#         T .= inv(scaling.T).(T_hat)
#         S .= inv(scaling.S).(S_hat)

#         ρ .= TEOS10.ρ.(T, S, 0, Ref(eos))
#         ρ_hat .= scaling.ρ.(ρ)
#         sol_ρ[:, i-1] .= ρ_hat

#         ∂T∂z_hat .= scaling.∂T∂z.(Dᶠ * T)
#         ∂S∂z_hat .= scaling.∂S∂z.(Dᶠ * S)
#         ∂ρ∂z_hat .= scaling.∂ρ∂z.(Dᶠ * ρ)

#         ∂T∂z_hat[1] = ∂T∂z_hat[2]
#         ∂S∂z_hat[1] = ∂S∂z_hat[2]
#         ∂ρ∂z_hat[1] = ∂ρ∂z_hat[2]

#         ∂T∂z_hat[end] = ∂T∂z_hat[end-1]
#         ∂S∂z_hat[end] = ∂S∂z_hat[end-1]
#         ∂ρ∂z_hat[end] = ∂ρ∂z_hat[end-1]

#         wT_residual, wS_residual = predict_residual_flux(∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat, ps, params, sts, NNs)
#         predict_boundary_flux!(wT_boundary, wS_boundary, params)

#         T_RHS .= - τ / H * scaling.wT.σ / scaling.T.σ .* (Dᶜ_hat * (wT_boundary .+ wT_residual))
#         S_RHS .= - τ / H * scaling.wS.σ / scaling.S.σ .* (Dᶜ_hat * (wS_boundary .+ wS_residual))

#         T_hat .= T_hat .+ Δt .* T_RHS
#         S_hat .= S_hat .+ Δt .* S_RHS

#         sol_T[:, i] .= T_hat
#         sol_S[:, i] .= S_hat
#     end

#     sol_ρ[:, end] .= scaling.ρ.(TEOS10.ρ.(inv(scaling.T).(T_hat), inv(scaling.S).(S_hat), 0, Ref(eos)))

#     return (; T=sol_T[:, 1:timestep_multiple:end], S=sol_S[:, 1:timestep_multiple:end], ρ=sol_ρ[:, 1:timestep_multiple:end])
# end

# function predict_residual_flux_dimensional(∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat, p, params, sts, NNs)
#     wT_hat, wS_hat = predict_residual_flux(∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat, p, params, sts, NNs)
    
#     wT_hat = wT_hat .- wT_hat[1]
#     wS_hat = wT_hat .- wS_hat[1]

#     wT = inv(params.scaling.wT).(wT_hat)
#     wS = inv(params.scaling.wS).(wS_hat)

#     wT = wT .- wT[1]
#     wS = wS .- wS[1]

#     return wT, wS
# end

# function predict_boundary_flux_dimensional(params)
#     _wT_boundary, _wS_boundary = predict_boundary_flux(params)

#     wT = inv(params.scaling.wT).(_wT_boundary)
#     wS = inv(params.scaling.wS).(_wS_boundary)

#     return wT, wS
# end

# function diagnose_fields(ps, params, x₀, sts, NNs, train_data_plot, timestep, Nt, timestep_multiple=2)
#     sols = solve_NDE(ps, params, x₀, sts, NNs, timestep, Nt, timestep_multiple)

#     ps_noNN = deepcopy(ps) .= 0
#     sols_noNN = solve_NDE(ps_noNN, params, x₀, sts, NNs, timestep, Nt, timestep_multiple)

#     coarse_size = params.coarse_size
#     Dᶠ = params.Dᶠ
#     scaling = params.scaling

#     Ts = inv(scaling.T).(sols.T)
#     Ss = inv(scaling.S).(sols.S)
#     ρs = inv(scaling.ρ).(sols.ρ)

#     Ts_noNN = inv(scaling.T).(sols_noNN.T)
#     Ss_noNN = inv(scaling.S).(sols_noNN.S)
#     ρs_noNN = inv(scaling.ρ).(sols_noNN.ρ)
    
#     ∂T∂z_hats = hcat([params.scaling.∂T∂z.(params.Dᶠ * T) for T in eachcol(Ts)]...)
#     ∂S∂z_hats = hcat([params.scaling.∂S∂z.(params.Dᶠ * S) for S in eachcol(Ss)]...)
#     ∂ρ∂z_hats = hcat([params.scaling.∂ρ∂z.(params.Dᶠ * ρ) for ρ in eachcol(ρs)]...)

#     ∂T∂z_hats[1, :] .= ∂T∂z_hats[2, :]
#     ∂S∂z_hats[1, :] .= ∂S∂z_hats[2, :]
#     ∂ρ∂z_hats[1, :] .= ∂ρ∂z_hats[2, :]

#     ∂T∂z_hats[end, :] .= ∂T∂z_hats[end-1, :]
#     ∂S∂z_hats[end, :] .= ∂S∂z_hats[end-1, :]
#     ∂ρ∂z_hats[end, :] .= ∂ρ∂z_hats[end-1, :]

#     eos = TEOS10EquationOfState()
#     Ris_truth = hcat([calculate_Ri(zeros(coarse_size), zeros(coarse_size), ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf)) for ρ in eachcol(train_data_plot.profile.ρ.unscaled)]...)
#     Ris = hcat([calculate_Ri(zeros(coarse_size), zeros(coarse_size), ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf)) for ρ in eachcol(ρs)]...)
#     Ris_noNN = hcat([calculate_Ri(zeros(coarse_size), zeros(coarse_size), ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf)) for ρ in eachcol(ρs_noNN)]...)
    
#     νs, κs = zeros(coarse_size+1, size(Ts, 2)), zeros(coarse_size+1, size(Ts, 2))
#     νs_noNN, κs_noNN = zeros(coarse_size+1, size(Ts, 2)), zeros(coarse_size+1, size(Ts, 2))

#     wT_residuals = zeros(coarse_size+1, size(Ts, 2))
#     wS_residuals = zeros(coarse_size+1, size(Ts, 2))

#     wT_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))
#     wS_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))

#     wT_diffusive_boundarys_noNN = zeros(coarse_size+1, size(Ts, 2))
#     wS_diffusive_boundarys_noNN = zeros(coarse_size+1, size(Ts, 2))

#     for i in 1:size(wT_residuals, 2)
#         wT_residuals[:, i], wS_residuals[:, i] = predict_residual_flux_dimensional(∂T∂z_hats[:, i], ∂S∂z_hats[:, i], ∂ρ∂z_hats[:, i], ps, params, sts, NNs)
#         wT_diffusive_boundarys[:, i], wS_diffusive_boundarys[:, i] = predict_boundary_flux_dimensional(params)        

#         wT_diffusive_boundarys_noNN[:, i], wS_diffusive_boundarys_noNN[:, i] = predict_boundary_flux_dimensional(params)
#     end

#     wT_totals = wT_residuals .+ wT_diffusive_boundarys
#     wS_totals = wS_residuals .+ wS_diffusive_boundarys

#     fluxes = (; wT = (; diffusive_boundary=wT_diffusive_boundarys, residual=wT_residuals, total=wT_totals), 
#                 wS = (; diffusive_boundary=wS_diffusive_boundarys, residual=wS_residuals, total=wS_totals))

#     fluxes_noNN = (; wT = (; total=wT_diffusive_boundarys_noNN), 
#                      wS = (; total=wS_diffusive_boundarys_noNN))

#     diffusivities = (; ν=νs, κ=κs, Ri=Ris, Ri_truth=Ris_truth)

#     diffusivities_noNN = (; ν=νs_noNN, κ=κs_noNN, Ri=Ris_noNN)

#     sols_dimensional = (; T=Ts, S=Ss, ρ=ρs)
#     sols_dimensional_noNN = (; T=Ts_noNN, S=Ss_noNN, ρ=ρs_noNN)
#     return (; sols_dimensional, sols_dimensional_noNN, fluxes, fluxes_noNN, diffusivities, diffusivities_noNN)
# end

# training_results = [diagnose_fields(ps, params, x₀, sts, NNs, plot_data, params.scaled_original_time[2] - params.scaled_original_time[1], length(full_timeframes_training[1]), 2) for (params, x₀, plot_data) in zip(training_params, x₀s_training, training_data_plot.data)]
# interpolating_results = [diagnose_fields(ps, params, x₀, sts, NNs, plot_data, params.scaled_original_time[2] - params.scaled_original_time[1], length(full_timeframes_interpolating[1]), 2) for (params, x₀, plot_data) in zip(interpolating_params, x₀s_interpolating, interpolating_data_plot.data)]

# jldsave("./plots_data/slightlylocalNN_dTdSdrho_training_results.jld2", u=training_results)
# jldsave("./plots_data/slightlylocalNN_dTdSdrho_interpolating_results.jld2", u=interpolating_results)

training_results = jldopen("./plots_data/slightlylocalNN_dTdSdrho_training_results.jld2", "r")["u"]
interpolating_results = jldopen("./plots_data/slightlylocalNN_dTdSdrho_interpolating_results.jld2", "r")["u"]
#%%
function compute_density_contribution(data)
    eos = TEOS10EquationOfState()
    ρ = data.profile.ρ.unscaled[:, 1]
    T = data.profile.T.unscaled[:, 1]
    S = data.profile.S.unscaled[:, 1]

    Δρ = maximum(ρ) - minimum(ρ)
    ΔT = maximum(T) - minimum(T)
    ΔS = maximum(S) - minimum(S)

    α = mean(SeawaterPolynomials.thermal_expansion.(T, S, 0, Ref(eos)))
    β = mean(SeawaterPolynomials.haline_contraction.(T, S, 0, Ref(eos)))
    ρ₀ = eos.reference_density

    T_contribution = α * ΔT * ρ₀
    S_contribution = β * ΔS * ρ₀

    return (; T=T_contribution, S=S_contribution, ρ=Δρ)
end

function compute_loss_prefactor_density_contribution(individual_loss, contribution, S_scaling=1.0)
    T_loss, S_loss, ρ_loss, ∂T∂z_loss, ∂S∂z_loss, ∂ρ∂z_loss = mean.(values(individual_loss))
    
    total_contribution = contribution.T + contribution.S
    T_prefactor = total_contribution / contribution.T
    S_prefactor = total_contribution / contribution.S

    TS_loss = T_prefactor * T_loss + S_prefactor * S_loss

    ρ_prefactor = TS_loss / ρ_loss * 0.1 / 0.9
    ∂T∂z_prefactor = T_prefactor
    ∂S∂z_prefactor = S_prefactor

    ∂TS∂z_loss = ∂T∂z_loss + ∂S∂z_loss
    ∂ρ∂z_prefactor = ∂TS∂z_loss / ∂ρ∂z_loss * 0.1 / 0.9

    profile_loss = T_prefactor * T_loss + S_prefactor * S_loss + ρ_prefactor * ρ_loss
    gradient_loss = ∂T∂z_prefactor * ∂T∂z_loss + ∂S∂z_prefactor * ∂S∂z_loss + ∂ρ∂z_prefactor * ∂ρ∂z_loss

    gradient_prefactor = profile_loss / gradient_loss

    ∂ρ∂z_prefactor *= gradient_prefactor
    ∂T∂z_prefactor *= gradient_prefactor
    ∂S∂z_prefactor *= gradient_prefactor

    S_prefactor *= S_scaling
    ∂S∂z_prefactor *= S_scaling

    return (T=T_prefactor, S=S_prefactor, ρ=ρ_prefactor, ∂T∂z=∂T∂z_prefactor, ∂S∂z=∂S∂z_prefactor, ∂ρ∂z=∂ρ∂z_prefactor)
end
#%%
density_contribution_training = compute_density_contribution.(training_data_plot.data)
density_contribution_interpolating = compute_density_contribution.(interpolating_data_plot.data)

function individual_loss(sol, truth, Dᶠ, scaling)
    sol_T, sol_S, sol_ρ = sol.T, sol.S, sol.ρ

    T_loss = vec(mean((scaling.T.(sol_T) .- scaling.T.(truth.profile.T.unscaled)).^2, dims=1))
    S_loss = vec(mean((scaling.S.(sol_S) .- scaling.S.(truth.profile.S.unscaled)).^2, dims=1))
    ρ_loss = vec(mean((scaling.ρ.(sol_ρ) .- scaling.ρ.(truth.profile.ρ.unscaled)).^2, dims=1))

    T = inv(scaling.T).(sol_T)
    S = inv(scaling.S).(sol_S)
    ρ = inv(scaling.ρ).(sol_ρ)

    ∂T∂z = scaling.∂T∂z.(Dᶠ * T)
    ∂S∂z = scaling.∂S∂z.(Dᶠ * S)
    ∂ρ∂z = scaling.∂ρ∂z.(Dᶠ * ρ)

    ∂T∂z_loss = vec(mean((∂T∂z[1:end-2, :] .- scaling.∂T∂z.(truth.profile.∂T∂z.unscaled)[1:end-2, :]).^2, dims=1))
    ∂S∂z_loss = vec(mean((∂S∂z[1:end-2, :] .- scaling.∂S∂z.(truth.profile.∂S∂z.unscaled)[1:end-2, :]).^2, dims=1))
    ∂ρ∂z_loss = vec(mean((∂ρ∂z[1:end-2, :] .- scaling.∂ρ∂z.(truth.profile.∂ρ∂z.unscaled)[1:end-2, :]).^2, dims=1))

    return (; T=T_loss, S=S_loss, ρ=ρ_loss, ∂T∂z=∂T∂z_loss, ∂S∂z=∂S∂z_loss, ∂ρ∂z=∂ρ∂z_loss)
end

individual_training_losses = [individual_loss(res.sols_dimensional, truth, params.Dᶠ, scaling) for (res, truth, params) in zip(training_results, training_data_plot.data, training_params)]
individual_interpolating_losses = [individual_loss(res.sols_dimensional, truth, params.Dᶠ, scaling) for (res, truth, params) in zip(interpolating_results, interpolating_data_plot.data, interpolating_params)]

weighted_training_loss_prefactors = compute_loss_prefactor_density_contribution.(individual_training_losses, density_contribution_training)
weighted_interpolating_loss_prefactors = compute_loss_prefactor_density_contribution.(individual_interpolating_losses, density_contribution_interpolating)

weighted_training_losses = [(; T=weighted_training_loss_prefactors[i].T * individual_training_losses[i].T,
                            S=weighted_training_loss_prefactors[i].S * individual_training_losses[i].S,
                            ρ=weighted_training_loss_prefactors[i].ρ * individual_training_losses[i].ρ,
                            ∂T∂z=weighted_training_loss_prefactors[i].∂T∂z * individual_training_losses[i].∂T∂z,
                            ∂S∂z=weighted_training_loss_prefactors[i].∂S∂z * individual_training_losses[i].∂S∂z,
                            ∂ρ∂z=weighted_training_loss_prefactors[i].∂ρ∂z * individual_training_losses[i].∂ρ∂z) for i in 1:length(training_results)]

weighted_interpolating_losses = [(; T=weighted_interpolating_loss_prefactors[i].T * individual_interpolating_losses[i].T,
                            S=weighted_interpolating_loss_prefactors[i].S * individual_interpolating_losses[i].S,
                            ρ=weighted_interpolating_loss_prefactors[i].ρ * individual_interpolating_losses[i].ρ,
                            ∂T∂z=weighted_interpolating_loss_prefactors[i].∂T∂z * individual_interpolating_losses[i].∂T∂z,
                            ∂S∂z=weighted_interpolating_loss_prefactors[i].∂S∂z * individual_interpolating_losses[i].∂S∂z,
                            ∂ρ∂z=weighted_interpolating_loss_prefactors[i].∂ρ∂z * individual_interpolating_losses[i].∂ρ∂z) for i in 1:length(interpolating_results)]

#%%
colors = distinguishable_colors(5, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

with_theme(theme_latexfonts()) do
    fig = Figure(size=(700, 700), fontsize=30)

    axσ = CairoMakie.Axis(fig[1,1], ylabel="z (m)", xlabel=L"$\overline{\sigma}$ (kg m$^{-3}$)", xticks=LinearTicks(3))

    res = interpolating_results[1]
    truth = interpolating_data_plot.data[1]
    loss = weighted_interpolating_losses[1]

    zC = training_data.data[1].metadata["zC"]
    zF = training_data.data[1].metadata["zF"]

    timeframes = [1, 50, 150, 265]
    times = round.((truth.times[timeframes] .- truth.times[timeframes[1]]) ./ 24 ./ 60^2, digits=1)

    full_times = truth.times ./ 24 ./ 60^2
    
    for (i, n) in enumerate(timeframes)
        lines!(axσ, truth.profile.ρ.unscaled[:, n], zC, color=colors[i], alpha=0.4, linewidth=15, label="$(times[i]) days")
        lines!(axσ, res.sols_dimensional.ρ[:, n], zC, color=colors[i], linewidth=5, linestyle=nothing)
    end
    axislegend(axσ, loc=":lb", labelfont=:bold)

    lin_LES = LineElement(color=(:black, 0.4), linewidth=15)
    lin_NN = LineElement(color=:black, linewidth=5)

    hidedecorations!(axσ, ticks=false, ticklabels=false, label=false)

    Legend(fig[2, 1], [lin_LES, lin_NN], ["Large Eddy Simulation", "NN Only"], orientation=:horizontal, tellwidth=false, patchsize=(40, 20), labelfont=:bold, labelsize=28)

    display(fig)
    # save("./poster_figures/density_slightlylocalNN_interpolation_1.png", fig, px_per_unit=16)
end
#%%