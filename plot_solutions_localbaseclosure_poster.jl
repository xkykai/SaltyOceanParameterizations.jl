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

# BASECLOSURE_FILE_DIR = "./training_output/localbaseclosure_convectivetanh_shearlinear_TSrho_EKI/training_results.jld2"
# ps_baseclosure = jldopen(BASECLOSURE_FILE_DIR, "r")["u"]

TRAINING_LES_FILE_DIRS = [
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.0005_QT_0.0_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.0002_QT_0.0_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0005_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0001_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0_QS_-5.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0_QS_-2.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.00015_QT_0.00045_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.0004_QT_0.00015_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.00015_QT_0.0_QS_-4.5e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.0004_QT_0.0_QS_-2.5e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",


    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.0005_QT_0.0_QS_0.0_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.0002_QT_0.0_QS_0.0_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_0.0_QT_0.0005_QS_0.0_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_0.0_QT_0.0001_QS_0.0_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_0.0_QT_0.0_QS_-5.0e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_0.0_QT_0.0_QS_-2.0e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.00015_QT_0.00045_QS_0.0_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.0004_QT_0.00015_QS_0.0_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.00015_QT_0.0_QS_-4.5e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.0004_QT_0.0_QS_-2.5e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
]

INTERPOLATING_FILE_DIRS = [
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.0004_QT_0.0_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.0003_QT_0.0_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0004_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0003_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0_QS_-4.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0_QS_-3.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.0002_QT_0.0_QS_-4.0e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.00035_QT_0.0_QS_-3.0e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.0004_QT_0.0_QS_0.0_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.0003_QT_0.0_QS_0.0_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_0.0_QT_0.0004_QS_0.0_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_0.0_QT_0.0003_QS_0.0_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.0002_QT_4.0e-5_QS_0.0_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.00035_QT_2.0e-5_QS_0.0_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.0002_QT_0.0_QS_-4.0e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.00035_QT_0.0_QS_-3.0e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_0.0_QT_0.0_QS_-4.0e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_0.0_QT_0.0_QS_-3.0e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_0.0_QT_0.0003_QS_-3.0e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0003_QS_-3.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.0002_QT_0.0004_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.00035_QT_0.0002_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
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

x₀s_training = [(; u=data.profile.u.scaled[:, 1], v=data.profile.v.scaled[:, 1], T=data.profile.T.scaled[:, 1], S=data.profile.S.scaled[:, 1]) for data in training_data.data]
x₀s_interpolating = [(; u=scaling.u.(data.profile.u.unscaled[:, 1]), v=scaling.v.(data.profile.v.unscaled[:, 1]), T=scaling.T.(data.profile.T.unscaled[:, 1]), S=scaling.S.(data.profile.S.unscaled[:, 1])) for data in interpolating_data_plot.data]

# training_params = [(                   f = data.coriolis.unscaled,
#                      f_scaled = data.coriolis.scaled,
#                             τ = data.times[end] - data.times[1],
#                         N_timesteps = length(data.times),
#                     scaled_time = (data.times .- data.times[1]) ./ (data.times[end] - data.times[1]),
#         scaled_original_time = (plot_data.times .- plot_data.times[1]) ./ (plot_data.times[end] - plot_data.times[1]),
#                             zC = data.metadata["zC"],
#                             H = data.metadata["original_grid"].Lz,
#                             g = data.metadata["gravitational_acceleration"],
#                     coarse_size = coarse_size, 
#                             Dᶜ = Dᶜ(coarse_size, data.metadata["zC"][2] - data.metadata["zC"][1]),
#                             Dᶠ = Dᶠ(coarse_size, data.metadata["zF"][3] - data.metadata["zF"][2]),
#                         Dᶜ_hat = Dᶜ(coarse_size, data.metadata["zC"][2] - data.metadata["zC"][1]) .* data.metadata["original_grid"].Lz,
#                         Dᶠ_hat = Dᶠ(coarse_size, data.metadata["zF"][3] - data.metadata["zF"][2]) .* data.metadata["original_grid"].Lz,
#                         uw = (scaled = (top=data.flux.uw.surface.scaled, bottom=data.flux.uw.bottom.scaled),
#                             unscaled = (top=data.flux.uw.surface.unscaled, bottom=data.flux.uw.bottom.unscaled)),
#                         vw = (scaled = (top=data.flux.vw.surface.scaled, bottom=data.flux.vw.bottom.scaled),
#                             unscaled = (top=data.flux.vw.surface.unscaled, bottom=data.flux.vw.bottom.unscaled)),
#                             wT = (scaled = (top=data.flux.wT.surface.scaled, bottom=data.flux.wT.bottom.scaled),
#                                 unscaled = (top=data.flux.wT.surface.unscaled, bottom=data.flux.wT.bottom.unscaled)),
#                             wS = (scaled = (top=data.flux.wS.surface.scaled, bottom=data.flux.wS.bottom.scaled),
#                                 unscaled = (top=data.flux.wS.surface.unscaled, bottom=data.flux.wS.bottom.unscaled)),
#                         scaling = scaling,
#                         ) for (data, plot_data) in zip(training_data.data, training_data_plot.data)]

# interpolating_params = [( f = data.coriolis.unscaled,
#                         f_scaled = scaling.f(data.coriolis.unscaled),
#                             τ = data.times[end] - data.times[1],
#                         N_timesteps = length(data.times),
#                     scaled_time = (data.times .- data.times[1]) ./ (data.times[end] - data.times[1]),
#         scaled_original_time = (data.times .- data.times[1]) ./ (data.times[end] - data.times[1]),
#                             zC = data.metadata["zC"],
#                             H = data.metadata["original_grid"].Lz,
#                             g = data.metadata["gravitational_acceleration"],
#                     coarse_size = coarse_size, 
#                             Dᶜ = Dᶜ(coarse_size, data.metadata["zC"][2] - data.metadata["zC"][1]),
#                             Dᶠ = Dᶠ(coarse_size, data.metadata["zF"][3] - data.metadata["zF"][2]),
#                         Dᶜ_hat = Dᶜ(coarse_size, data.metadata["zC"][2] - data.metadata["zC"][1]) .* data.metadata["original_grid"].Lz,
#                         Dᶠ_hat = Dᶠ(coarse_size, data.metadata["zF"][3] - data.metadata["zF"][2]) .* data.metadata["original_grid"].Lz,
#                             uw = (scaled = (top=scaling.uw(data.flux.uw.surface.unscaled), bottom=scaling.uw(data.flux.uw.bottom.unscaled)),
#                                 unscaled = (top=data.flux.uw.surface.unscaled, bottom=data.flux.uw.bottom.unscaled)),
#                             vw = (scaled = (top=scaling.vw(data.flux.vw.surface.unscaled), bottom=scaling.vw(data.flux.vw.bottom.unscaled)),
#                                 unscaled = (top=data.flux.vw.surface.unscaled, bottom=data.flux.vw.bottom.unscaled)),
#                             wT = (scaled = (top=scaling.wT(data.flux.wT.surface.unscaled), bottom=scaling.wT(data.flux.wT.bottom.unscaled)),
#                                 unscaled = (top=data.flux.wT.surface.unscaled, bottom=data.flux.wT.bottom.unscaled)),
#                             wS = (scaled = (top=scaling.wS(data.flux.wS.surface.unscaled), bottom=scaling.wS(data.flux.wS.bottom.unscaled)),
#                                 unscaled = (top=data.flux.wS.surface.unscaled, bottom=data.flux.wS.bottom.unscaled)),
#                         scaling = scaling,
#                         ) for data in interpolating_data_plot.data]

# function predict_boundary_flux(params)
#     uw = vcat(fill(params.uw.scaled.bottom, params.coarse_size), params.uw.scaled.top)
#     vw = vcat(fill(params.vw.scaled.bottom, params.coarse_size), params.vw.scaled.top)
#     wT = vcat(fill(params.wT.scaled.bottom, params.coarse_size), params.wT.scaled.top)
#     wS = vcat(fill(params.wS.scaled.bottom, params.coarse_size), params.wS.scaled.top)

#     return uw, vw, wT, wS
# end

# function predict_boundary_flux!(uw, vw, wT, wS, params)
#     uw[1:end-1] .= params.uw.scaled.bottom
#     vw[1:end-1] .= params.vw.scaled.bottom
#     wT[1:end-1] .= params.wT.scaled.bottom
#     wS[1:end-1] .= params.wS.scaled.bottom

#     uw[end] = params.uw.scaled.top
#     vw[end] = params.vw.scaled.top
#     wT[end] = params.wT.scaled.top
#     wS[end] = params.wS.scaled.top

#     return nothing
# end

# function predict_diffusivities(Ris, ps_baseclosure)
#     νs = local_Ri_ν_convectivetanh_shearlinear.(Ris, ps_baseclosure.ν_conv, ps_baseclosure.ν_shear, ps_baseclosure.m, ps_baseclosure.ΔRi)
#     κs = local_Ri_κ_convectivetanh_shearlinear.(νs, ps_baseclosure.Pr)
#     return νs, κs
# end

# function predict_diffusivities!(νs, κs, Ris, ps_baseclosure)
#     νs .= local_Ri_ν_convectivetanh_shearlinear.(Ris, ps_baseclosure.ν_conv, ps_baseclosure.ν_shear, ps_baseclosure.m, ps_baseclosure.ΔRi)
#     κs .= local_Ri_κ_convectivetanh_shearlinear.(νs, ps_baseclosure.Pr)
#     return nothing
# end

# function solve_NDE(ps, params, x₀, timestep, Nt, timestep_multiple=2)
#     eos = TEOS10EquationOfState()
#     coarse_size = params.coarse_size
#     Δt = timestep / timestep_multiple
#     Nt_solve = (Nt - 1) * timestep_multiple + 1
#     Dᶜ_hat = params.Dᶜ_hat
#     Dᶠ_hat = params.Dᶠ_hat
#     Dᶠ = params.Dᶠ

#     scaling = params.scaling
#     τ, H = params.τ, params.H
#     f = params.f

#     u_hat = deepcopy(x₀.u)
#     v_hat = deepcopy(x₀.v)
#     T_hat = deepcopy(x₀.T)
#     S_hat = deepcopy(x₀.S)
#     ρ_hat = zeros(coarse_size)

#     u = zeros(coarse_size)
#     v = zeros(coarse_size)
#     T = zeros(coarse_size)
#     S = zeros(coarse_size)
#     ρ = zeros(coarse_size)
    
#     u_RHS = zeros(coarse_size)
#     v_RHS = zeros(coarse_size)
#     T_RHS = zeros(coarse_size)
#     S_RHS = zeros(coarse_size)

#     sol_u = zeros(coarse_size, Nt_solve)
#     sol_v = zeros(coarse_size, Nt_solve)
#     sol_T = zeros(coarse_size, Nt_solve)
#     sol_S = zeros(coarse_size, Nt_solve)
#     sol_ρ = zeros(coarse_size, Nt_solve)

#     uw_boundary = zeros(coarse_size+1)
#     vw_boundary = zeros(coarse_size+1)
#     wT_boundary = zeros(coarse_size+1)
#     wS_boundary = zeros(coarse_size+1)

#     νs = zeros(coarse_size+1)
#     κs = zeros(coarse_size+1)

#     Ris = zeros(coarse_size+1)

#     sol_u[:, 1] .= u_hat
#     sol_v[:, 1] .= v_hat
#     sol_T[:, 1] .= T_hat
#     sol_S[:, 1] .= S_hat

#     ν_LHS = Tridiagonal(zeros(32, 32))
#     κ_LHS = Tridiagonal(zeros(32, 32))

#     for i in 2:Nt_solve
#         u .= inv(scaling.u).(u_hat)
#         v .= inv(scaling.v).(v_hat)
#         T .= inv(scaling.T).(T_hat)
#         S .= inv(scaling.S).(S_hat)

#         ρ .= TEOS10.ρ.(T, S, 0, Ref(eos))
#         ρ_hat .= scaling.ρ.(ρ)
#         sol_ρ[:, i-1] .= ρ_hat

#         Ris .= calculate_Ri(u, v, ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf))
#         predict_diffusivities!(νs, κs, Ris, ps)

#         Dν = Dᶜ_hat * (-νs .* Dᶠ_hat)
#         Dκ = Dᶜ_hat * (-κs .* Dᶠ_hat)

#         predict_boundary_flux!(uw_boundary, vw_boundary, wT_boundary, wS_boundary, params)

#         ν_LHS .= Tridiagonal(-τ / H^2 .* Dν)
#         κ_LHS .= Tridiagonal(-τ / H^2 .* Dκ)

#         u_RHS .= - τ / H * scaling.uw.σ / scaling.u.σ .* (Dᶜ_hat * (uw_boundary)) .+ f * τ ./ scaling.u.σ .* v
#         v_RHS .= - τ / H * scaling.vw.σ / scaling.v.σ .* (Dᶜ_hat * (vw_boundary)) .- f * τ ./ scaling.v.σ .* u
#         T_RHS .= - τ / H * scaling.wT.σ / scaling.T.σ .* (Dᶜ_hat * (wT_boundary))
#         S_RHS .= - τ / H * scaling.wS.σ / scaling.S.σ .* (Dᶜ_hat * (wS_boundary))

#         u_hat .= (I - Δt .* ν_LHS) \ (u_hat .+ Δt .* u_RHS)
#         v_hat .= (I - Δt .* ν_LHS) \ (v_hat .+ Δt .* v_RHS)
#         T_hat .= (I - Δt .* κ_LHS) \ (T_hat .+ Δt .* T_RHS)
#         S_hat .= (I - Δt .* κ_LHS) \ (S_hat .+ Δt .* S_RHS)

#         sol_u[:, i] .= u_hat
#         sol_v[:, i] .= v_hat
#         sol_T[:, i] .= T_hat
#         sol_S[:, i] .= S_hat
#     end

#     sol_ρ[:, end] .= scaling.ρ.(TEOS10.ρ.(inv(scaling.T).(T_hat), inv(scaling.S).(S_hat), 0, Ref(eos)))

#     return (; u=sol_u[:, 1:timestep_multiple:end], v=sol_v[:, 1:timestep_multiple:end], T=sol_T[:, 1:timestep_multiple:end], S=sol_S[:, 1:timestep_multiple:end], ρ=sol_ρ[:, 1:timestep_multiple:end])
# end

# function predict_diffusive_flux(Ris, u_hat, v_hat, T_hat, S_hat, ps_baseclosure, params)
#     νs, κs = predict_diffusivities(Ris, ps_baseclosure)

#     ∂u∂z_hat = params.Dᶠ_hat * u_hat
#     ∂v∂z_hat = params.Dᶠ_hat * v_hat
#     ∂T∂z_hat = params.Dᶠ_hat * T_hat
#     ∂S∂z_hat = params.Dᶠ_hat * S_hat

#     uw_diffusive = -νs .* ∂u∂z_hat
#     vw_diffusive = -νs .* ∂v∂z_hat
#     wT_diffusive = -κs .* ∂T∂z_hat
#     wS_diffusive = -κs .* ∂S∂z_hat
#     return uw_diffusive, vw_diffusive, wT_diffusive, wS_diffusive
# end

# function predict_diffusive_boundary_flux_dimensional(Ris, u_hat, v_hat, T_hat, S_hat, ps_baseclosure, params)
#     _uw_diffusive, _vw_diffusive, _wT_diffusive, _wS_diffusive = predict_diffusive_flux(Ris, u_hat, v_hat, T_hat, S_hat, ps_baseclosure, params)
#     _uw_boundary, _vw_boundary, _wT_boundary, _wS_boundary = predict_boundary_flux(params)

#     uw_diffusive = params.scaling.u.σ / params.H .* _uw_diffusive
#     vw_diffusive = params.scaling.v.σ / params.H .* _vw_diffusive
#     wT_diffusive = params.scaling.T.σ / params.H .* _wT_diffusive
#     wS_diffusive = params.scaling.S.σ / params.H .* _wS_diffusive

#     uw_boundary = inv(params.scaling.uw).(_uw_boundary)
#     vw_boundary = inv(params.scaling.vw).(_vw_boundary)
#     wT_boundary = inv(params.scaling.wT).(_wT_boundary)
#     wS_boundary = inv(params.scaling.wS).(_wS_boundary)

#     uw = uw_diffusive .+ uw_boundary
#     vw = vw_diffusive .+ vw_boundary
#     wT = wT_diffusive .+ wT_boundary
#     wS = wS_diffusive .+ wS_boundary

#     return uw, vw, wT, wS
# end

# function diagnose_fields(ps, params, x₀, train_data_plot, timestep, Nt, timestep_multiple=2)
#     sols = solve_NDE(ps, params, x₀, timestep, Nt, timestep_multiple)

#     coarse_size = params.coarse_size
#     Dᶠ = params.Dᶠ
#     scaling = params.scaling

#     us = inv(scaling.u).(sols.u)
#     vs = inv(scaling.v).(sols.v)
#     Ts = inv(scaling.T).(sols.T)
#     Ss = inv(scaling.S).(sols.S)
#     ρs = inv(scaling.ρ).(sols.ρ)

#     eos = TEOS10EquationOfState()
#     Ris_truth = hcat([calculate_Ri(u, v, ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf)) for (u, v, ρ) in zip(eachcol(train_data_plot.profile.u.unscaled), eachcol(train_data_plot.profile.v.unscaled), eachcol(train_data_plot.profile.ρ.unscaled))]...)
#     Ris = hcat([calculate_Ri(u, v, ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf)) for (u, v, ρ) in zip(eachcol(us), eachcol(vs), eachcol(ρs))]...)
    
#     νs, κs = predict_diffusivities(Ris, ps)

#     uw_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))
#     vw_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))
#     wT_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))
#     wS_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))

#     for i in 1:size(wT_diffusive_boundarys, 2)
#         uw_diffusive_boundarys[:, i], vw_diffusive_boundarys[:, i], wT_diffusive_boundarys[:, i], wS_diffusive_boundarys[:, i] = predict_diffusive_boundary_flux_dimensional(Ris[:, i], sols.u[:, i], sols.v[:, i], sols.T[:, i], sols.S[:, i], ps, params)        
#     end

#     uw_totals = uw_diffusive_boundarys
#     vw_totals = vw_diffusive_boundarys
#     wT_totals = wT_diffusive_boundarys
#     wS_totals = wS_diffusive_boundarys

#     fluxes = (; uw = (; total=uw_totals),
#                 vw = (; total=vw_totals),
#                 wT = (; total=wT_totals), 
#                 wS = (; total=wS_totals))

#     diffusivities = (; ν=νs, κ=κs, Ri=Ris, Ri_truth=Ris_truth)

#     sols_dimensional = (; u=us, v=vs, T=Ts, S=Ss, ρ=ρs)
#     return (; sols_dimensional, fluxes, diffusivities)
# end

# training_results = [diagnose_fields(ps_baseclosure, params, x₀, plot_data, params.scaled_original_time[2] - params.scaled_original_time[1], length(full_timeframes_training[1]), 2) for (params, x₀, plot_data) in zip(training_params, x₀s_training, training_data_plot.data)]
# interpolating_results = [diagnose_fields(ps_baseclosure, params, x₀, plot_data, params.scaled_original_time[2] - params.scaled_original_time[1], length(full_timeframes_interpolating[1]), 2) for (params, x₀, plot_data) in zip(interpolating_params, x₀s_interpolating, interpolating_data_plot.data)]

# jldsave("./plots_data/localbaseclosure_convectivetanh_shearlinear_TSrho_EKI_training_results.jld2", u=training_results)
# jldsave("./plots_data/localbaseclosure_convectivetanh_shearlinear_TSrho_EKI_interpolating_results.jld2", u=interpolating_results)

training_results = jldopen("./plots_data/localbaseclosure_convectivetanh_shearlinear_TSrho_EKI_training_results.jld2")["u"]
interpolating_results = jldopen("./plots_data/localbaseclosure_convectivetanh_shearlinear_TSrho_EKI_interpolating_results.jld2")["u"]
#%%
colors = distinguishable_colors(4, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

with_theme(theme_latexfonts()) do
    fig = Figure(size=(1400, 650), fontsize=30)

    l1 = fig[1, 1:2] = GridLayout()
    l2 = fig[2, 1:2] = GridLayout()

    axu = CairoMakie.Axis(l1[1,1], ylabel="z (m)", xlabel=L"$\overline{u}$ (m s$^{-1}$)")
    axv = CairoMakie.Axis(l1[1,2], ylabel="z (m)", xlabel=L"$\overline{v}$ (m s$^{-1}$)", xticks=LinearTicks(3))
    axT = CairoMakie.Axis(l1[1,3], ylabel="z (m)", xlabel=L"$\overline{T}$ (°C)")
    axS = CairoMakie.Axis(l1[1,4], ylabel="z (m)", xlabel=L"$\overline{S}$ (g kg$^{-1}$)", xticks=LinearTicks(3))

    train_res = interpolating_results[1]
    truth_train = interpolating_data_plot.data[1]

    zC = training_data.data[1].metadata["zC"]
    zF = training_data.data[1].metadata["zF"]

    timeframes = [1, 50, 170, 265]
    times = round.((truth_train.times[timeframes] .- truth_train.times[timeframes[1]]) ./ 24 ./ 60^2, digits=2)

    for (i, n) in enumerate(timeframes)
        lines!(axu, truth_train.profile.u.unscaled[17:end, n], zC[17:end], color=colors[i], linewidth=15, alpha=0.4, label="$(times[i]) days")
        lines!(axT, truth_train.profile.T.unscaled[17:end, n], zC[17:end], color=colors[i], linewidth=15, alpha=0.4, label="$(times[i]) days")
        lines!(axS, truth_train.profile.S.unscaled[17:end, n], zC[17:end], color=colors[i], linewidth=15, alpha=0.4, label="$(times[i]) days")
        
        if i == 1
            lines!(axv, truth_train.profile.v.unscaled[17:end, n], zC[17:end], color=(colors[i], 0.4), linewidth=15, label="Large Eddy Simulation")
            lines!(axv, train_res.sols_dimensional.v[17:end, n], zC[17:end], color=colors[i], linestyle=:dot, linewidth=5, label="Physical Closure")
        else
            lines!(axv, truth_train.profile.v.unscaled[17:end, n], zC[17:end], color=(colors[i], 0.4), linewidth=15)
            lines!(axv, train_res.sols_dimensional.v[17:end, n], zC[17:end], color=colors[i], linestyle=:dot, linewidth=5)
        end

        lines!(axu, train_res.sols_dimensional.u[17:end, n], zC[17:end], color=colors[i], linestyle=:dot, linewidth=5)
        lines!(axT, train_res.sols_dimensional.T[17:end, n], zC[17:end], color=colors[i], linestyle=:dot, linewidth=5)
        lines!(axS, train_res.sols_dimensional.S[17:end, n], zC[17:end], color=colors[i], linestyle=:dot, linewidth=5)
    end

    linkyaxes!(axu, axv, axT, axS)
    hidedecorations!(axu, ticks=false, ticklabels=false, label=false)
    hidedecorations!(axv, ticks=false, ticklabels=false, label=false)
    hidedecorations!(axT, ticks=false, ticklabels=false, label=false)
    hidedecorations!(axS, ticks=false, ticklabels=false, label=false)

    hideydecorations!(axv, ticks=false)
    hideydecorations!(axT, ticks=false)
    hideydecorations!(axS, ticks=false)

    hidexdecorations!(axu, ticks=false, ticklabels=false, label=false)
    hidexdecorations!(axv, ticks=false, ticklabels=false, label=false)
    hidexdecorations!(axT, ticks=false, ticklabels=false, label=false)
    hidexdecorations!(axS, ticks=false, ticklabels=false, label=false)

    # Legend(l2[1, 1], axu, orientation=:horizontal, tellwidth=false)
    axislegend(axu, position=:rb, labelfont=:bold)
    lin_LES = LineElement(color=(:black, 0.4), linewidth=15)
    lin_closure = LineElement(color=:black, linewidth=5, linestyle=:dot)

    # Legend(l2[1, 1], axv, orientation=:horizontal, tellwidth=false, labelfont=:bold, patchsize=(70, 20))
    Legend(l2[1, 1], [lin_LES, lin_closure], ["Large Eddy Simulation", "Physical Closure"], orientation=:horizontal, tellwidth=false, labelfont=:bold, patchsize=(70, 20))

    display(fig)
    # save("./poster_figures/localbaseclosure_convectivetanh_shearlinear_interpolating_1.png", fig, px_per_unit=16)
end
#%%