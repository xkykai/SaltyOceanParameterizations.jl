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
using ColorSchemes

function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

BASECLOSURE_FILE_DIR = "./training_output/localbaseclosure_convectivetanh_shearlinear_TSrho_EKI/training_results.jld2"
ps_baseclosure = jldopen(BASECLOSURE_FILE_DIR, "r")["u"]

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
                        uw = (scaled = (top=data.flux.uw.surface.scaled, bottom=data.flux.uw.bottom.scaled),
                            unscaled = (top=data.flux.uw.surface.unscaled, bottom=data.flux.uw.bottom.unscaled)),
                        vw = (scaled = (top=data.flux.vw.surface.scaled, bottom=data.flux.vw.bottom.scaled),
                            unscaled = (top=data.flux.vw.surface.unscaled, bottom=data.flux.vw.bottom.unscaled)),
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
                            uw = (scaled = (top=scaling.uw(data.flux.uw.surface.unscaled), bottom=scaling.uw(data.flux.uw.bottom.unscaled)),
                                unscaled = (top=data.flux.uw.surface.unscaled, bottom=data.flux.uw.bottom.unscaled)),
                            vw = (scaled = (top=scaling.vw(data.flux.vw.surface.unscaled), bottom=scaling.vw(data.flux.vw.bottom.unscaled)),
                                unscaled = (top=data.flux.vw.surface.unscaled, bottom=data.flux.vw.bottom.unscaled)),
                            wT = (scaled = (top=scaling.wT(data.flux.wT.surface.unscaled), bottom=scaling.wT(data.flux.wT.bottom.unscaled)),
                                unscaled = (top=data.flux.wT.surface.unscaled, bottom=data.flux.wT.bottom.unscaled)),
                            wS = (scaled = (top=scaling.wS(data.flux.wS.surface.unscaled), bottom=scaling.wS(data.flux.wS.bottom.unscaled)),
                                unscaled = (top=data.flux.wS.surface.unscaled, bottom=data.flux.wS.bottom.unscaled)),
                        scaling = scaling,
                        ) for data in interpolating_data_plot.data]

function predict_boundary_flux(params)
    uw = vcat(fill(params.uw.scaled.bottom, params.coarse_size), params.uw.scaled.top)
    vw = vcat(fill(params.vw.scaled.bottom, params.coarse_size), params.vw.scaled.top)
    wT = vcat(fill(params.wT.scaled.bottom, params.coarse_size), params.wT.scaled.top)
    wS = vcat(fill(params.wS.scaled.bottom, params.coarse_size), params.wS.scaled.top)

    return uw, vw, wT, wS
end

function predict_boundary_flux!(uw, vw, wT, wS, params)
    uw[1:end-1] .= params.uw.scaled.bottom
    vw[1:end-1] .= params.vw.scaled.bottom
    wT[1:end-1] .= params.wT.scaled.bottom
    wS[1:end-1] .= params.wS.scaled.bottom

    uw[end] = params.uw.scaled.top
    vw[end] = params.vw.scaled.top
    wT[end] = params.wT.scaled.top
    wS[end] = params.wS.scaled.top

    return nothing
end

function predict_diffusivities(Ris, ps_baseclosure)
    νs = local_Ri_ν_convectivetanh_shearlinear.(Ris, ps_baseclosure.ν_conv, ps_baseclosure.ν_shear, ps_baseclosure.m, ps_baseclosure.ΔRi)
    κs = local_Ri_κ_convectivetanh_shearlinear.(νs, ps_baseclosure.Pr)
    return νs, κs
end

function predict_diffusivities!(νs, κs, Ris, ps_baseclosure)
    νs .= local_Ri_ν_convectivetanh_shearlinear.(Ris, ps_baseclosure.ν_conv, ps_baseclosure.ν_shear, ps_baseclosure.m, ps_baseclosure.ΔRi)
    κs .= local_Ri_κ_convectivetanh_shearlinear.(νs, ps_baseclosure.Pr)
    return nothing
end

function solve_NDE(ps, params, x₀, timestep, Nt, timestep_multiple=2)
    eos = TEOS10EquationOfState()
    coarse_size = params.coarse_size
    Δt = timestep / timestep_multiple
    Nt_solve = (Nt - 1) * timestep_multiple + 1
    Dᶜ_hat = params.Dᶜ_hat
    Dᶠ_hat = params.Dᶠ_hat
    Dᶠ = params.Dᶠ

    scaling = params.scaling
    τ, H = params.τ, params.H
    f = params.f

    u_hat = deepcopy(x₀.u)
    v_hat = deepcopy(x₀.v)
    T_hat = deepcopy(x₀.T)
    S_hat = deepcopy(x₀.S)
    ρ_hat = zeros(coarse_size)

    u = zeros(coarse_size)
    v = zeros(coarse_size)
    T = zeros(coarse_size)
    S = zeros(coarse_size)
    ρ = zeros(coarse_size)
    
    u_RHS = zeros(coarse_size)
    v_RHS = zeros(coarse_size)
    T_RHS = zeros(coarse_size)
    S_RHS = zeros(coarse_size)

    sol_u = zeros(coarse_size, Nt_solve)
    sol_v = zeros(coarse_size, Nt_solve)
    sol_T = zeros(coarse_size, Nt_solve)
    sol_S = zeros(coarse_size, Nt_solve)
    sol_ρ = zeros(coarse_size, Nt_solve)

    uw_boundary = zeros(coarse_size+1)
    vw_boundary = zeros(coarse_size+1)
    wT_boundary = zeros(coarse_size+1)
    wS_boundary = zeros(coarse_size+1)

    νs = zeros(coarse_size+1)
    κs = zeros(coarse_size+1)

    Ris = zeros(coarse_size+1)

    sol_u[:, 1] .= u_hat
    sol_v[:, 1] .= v_hat
    sol_T[:, 1] .= T_hat
    sol_S[:, 1] .= S_hat

    ν_LHS = Tridiagonal(zeros(32, 32))
    κ_LHS = Tridiagonal(zeros(32, 32))

    for i in 2:Nt_solve
        u .= inv(scaling.u).(u_hat)
        v .= inv(scaling.v).(v_hat)
        T .= inv(scaling.T).(T_hat)
        S .= inv(scaling.S).(S_hat)

        ρ .= TEOS10.ρ.(T, S, 0, Ref(eos))
        ρ_hat .= scaling.ρ.(ρ)
        sol_ρ[:, i-1] .= ρ_hat

        Ris .= calculate_Ri(u, v, ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf))
        predict_diffusivities!(νs, κs, Ris, ps)

        Dν = Dᶜ_hat * (-νs .* Dᶠ_hat)
        Dκ = Dᶜ_hat * (-κs .* Dᶠ_hat)

        predict_boundary_flux!(uw_boundary, vw_boundary, wT_boundary, wS_boundary, params)

        ν_LHS .= Tridiagonal(-τ / H^2 .* Dν)
        κ_LHS .= Tridiagonal(-τ / H^2 .* Dκ)

        u_RHS .= - τ / H * scaling.uw.σ / scaling.u.σ .* (Dᶜ_hat * (uw_boundary)) .+ f * τ ./ scaling.u.σ .* v
        v_RHS .= - τ / H * scaling.vw.σ / scaling.v.σ .* (Dᶜ_hat * (vw_boundary)) .- f * τ ./ scaling.v.σ .* u
        T_RHS .= - τ / H * scaling.wT.σ / scaling.T.σ .* (Dᶜ_hat * (wT_boundary))
        S_RHS .= - τ / H * scaling.wS.σ / scaling.S.σ .* (Dᶜ_hat * (wS_boundary))

        u_hat .= (I - Δt .* ν_LHS) \ (u_hat .+ Δt .* u_RHS)
        v_hat .= (I - Δt .* ν_LHS) \ (v_hat .+ Δt .* v_RHS)
        T_hat .= (I - Δt .* κ_LHS) \ (T_hat .+ Δt .* T_RHS)
        S_hat .= (I - Δt .* κ_LHS) \ (S_hat .+ Δt .* S_RHS)

        sol_u[:, i] .= u_hat
        sol_v[:, i] .= v_hat
        sol_T[:, i] .= T_hat
        sol_S[:, i] .= S_hat
    end

    sol_ρ[:, end] .= scaling.ρ.(TEOS10.ρ.(inv(scaling.T).(T_hat), inv(scaling.S).(S_hat), 0, Ref(eos)))

    return (; u=sol_u[:, 1:timestep_multiple:end], v=sol_v[:, 1:timestep_multiple:end], T=sol_T[:, 1:timestep_multiple:end], S=sol_S[:, 1:timestep_multiple:end], ρ=sol_ρ[:, 1:timestep_multiple:end])
end

function predict_diffusive_flux(Ris, u_hat, v_hat, T_hat, S_hat, ps_baseclosure, params)
    νs, κs = predict_diffusivities(Ris, ps_baseclosure)

    ∂u∂z_hat = params.Dᶠ_hat * u_hat
    ∂v∂z_hat = params.Dᶠ_hat * v_hat
    ∂T∂z_hat = params.Dᶠ_hat * T_hat
    ∂S∂z_hat = params.Dᶠ_hat * S_hat

    uw_diffusive = -νs .* ∂u∂z_hat
    vw_diffusive = -νs .* ∂v∂z_hat
    wT_diffusive = -κs .* ∂T∂z_hat
    wS_diffusive = -κs .* ∂S∂z_hat
    return uw_diffusive, vw_diffusive, wT_diffusive, wS_diffusive
end

function predict_diffusive_boundary_flux_dimensional(Ris, u_hat, v_hat, T_hat, S_hat, ps_baseclosure, params)
    _uw_diffusive, _vw_diffusive, _wT_diffusive, _wS_diffusive = predict_diffusive_flux(Ris, u_hat, v_hat, T_hat, S_hat, ps_baseclosure, params)
    _uw_boundary, _vw_boundary, _wT_boundary, _wS_boundary = predict_boundary_flux(params)

    uw_diffusive = params.scaling.u.σ / params.H .* _uw_diffusive
    vw_diffusive = params.scaling.v.σ / params.H .* _vw_diffusive
    wT_diffusive = params.scaling.T.σ / params.H .* _wT_diffusive
    wS_diffusive = params.scaling.S.σ / params.H .* _wS_diffusive

    uw_boundary = inv(params.scaling.uw).(_uw_boundary)
    vw_boundary = inv(params.scaling.vw).(_vw_boundary)
    wT_boundary = inv(params.scaling.wT).(_wT_boundary)
    wS_boundary = inv(params.scaling.wS).(_wS_boundary)

    uw = uw_diffusive .+ uw_boundary
    vw = vw_diffusive .+ vw_boundary
    wT = wT_diffusive .+ wT_boundary
    wS = wS_diffusive .+ wS_boundary

    return uw, vw, wT, wS
end

function diagnose_fields(ps, params, x₀, train_data_plot, timestep, Nt, timestep_multiple=2)
    sols = solve_NDE(ps, params, x₀, timestep, Nt, timestep_multiple)

    coarse_size = params.coarse_size
    Dᶠ = params.Dᶠ
    scaling = params.scaling

    us = inv(scaling.u).(sols.u)
    vs = inv(scaling.v).(sols.v)
    Ts = inv(scaling.T).(sols.T)
    Ss = inv(scaling.S).(sols.S)
    ρs = inv(scaling.ρ).(sols.ρ)

    eos = TEOS10EquationOfState()
    Ris_truth = hcat([calculate_Ri(u, v, ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf)) for (u, v, ρ) in zip(eachcol(train_data_plot.profile.u.unscaled), eachcol(train_data_plot.profile.v.unscaled), eachcol(train_data_plot.profile.ρ.unscaled))]...)
    Ris = hcat([calculate_Ri(u, v, ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf)) for (u, v, ρ) in zip(eachcol(us), eachcol(vs), eachcol(ρs))]...)
    
    νs, κs = predict_diffusivities(Ris, ps)

    uw_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))
    vw_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))
    wT_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))
    wS_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))

    for i in 1:size(wT_diffusive_boundarys, 2)
        uw_diffusive_boundarys[:, i], vw_diffusive_boundarys[:, i], wT_diffusive_boundarys[:, i], wS_diffusive_boundarys[:, i] = predict_diffusive_boundary_flux_dimensional(Ris[:, i], sols.u[:, i], sols.v[:, i], sols.T[:, i], sols.S[:, i], ps, params)        
    end

    uw_totals = uw_diffusive_boundarys
    vw_totals = vw_diffusive_boundarys
    wT_totals = wT_diffusive_boundarys
    wS_totals = wS_diffusive_boundarys

    fluxes = (; uw = (; total=uw_totals),
                vw = (; total=vw_totals),
                wT = (; total=wT_totals), 
                wS = (; total=wS_totals))

    diffusivities = (; ν=νs, κ=κs, Ri=Ris, Ri_truth=Ris_truth)

    sols_dimensional = (; u=us, v=vs, T=Ts, S=Ss, ρ=ρs)
    return (; sols_dimensional, fluxes, diffusivities)
end

training_results = [diagnose_fields(ps_baseclosure, params, x₀, plot_data, params.scaled_original_time[2] - params.scaled_original_time[1], length(full_timeframes_training[1]), 2) for (params, x₀, plot_data) in zip(training_params, x₀s_training, training_data_plot.data)]
interpolating_results = [diagnose_fields(ps_baseclosure, params, x₀, plot_data, params.scaled_original_time[2] - params.scaled_original_time[1], length(full_timeframes_interpolating[1]), 2) for (params, x₀, plot_data) in zip(interpolating_params, x₀s_interpolating, interpolating_data_plot.data)]

jldsave("./plots_data/localbaseclosure_convectivetanh_shearlinear_TSrho_EKI_training_results.jld2", u=training_results)
jldsave("./plots_data/localbaseclosure_convectivetanh_shearlinear_TSrho_EKI_interpolating_results.jld2", u=interpolating_results)

#%%
# colors = distinguishable_colors(2, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
colors = colorschemes[:Set1_9]

for (i, (train_res, truth_train)) in enumerate(zip(training_results, training_data_plot.data))
    @info "Plotting $i"

    if i in [3, 4, 5, 6, 13, 14, 15, 16]
        FC = true
    else
        FC = false
    end

    if i in [1, 2, 11, 12]
        PW = true
    else
        PW = false
    end

    with_theme(theme_latexfonts()) do
        fig = Figure(size=(1920, 980), fontsize=20)

        l1 = fig[1, 1:2] = GridLayout()

        axu = CairoMakie.Axis(l1[1,1], ylabel="z (m)", xlabel=L"$\overline{u}$ (m s$^{-1}$)")
        axv = CairoMakie.Axis(l1[1,2], ylabel="z (m)", xlabel=L"$\overline{v}$ (m s$^{-1}$)")
        axT = CairoMakie.Axis(l1[1,3], ylabel="z (m)", xlabel=L"$\overline{T}$ (°C)")
        axS = CairoMakie.Axis(l1[1,4], ylabel="z (m)", xlabel=L"$\overline{S}$ (g kg$^{-1}$)")
        axσ = CairoMakie.Axis(l1[1,5], ylabel="z (m)", xlabel=L"$\overline{\sigma}$ (kg m$^{-3}$)", xticks=LinearTicks(2))
        axRi = CairoMakie.Axis(l1[1,6], ylabel="z (m)", xlabel=L"Ri")
        axuw = CairoMakie.Axis(l1[2,1], ylabel="z (m)", xlabel=L"$\overline{u\prime w\prime}$ (m$^{2}$ s$^{-2}$)", xticks=LinearTicks(2))
        axvw = CairoMakie.Axis(l1[2,2], ylabel="z (m)", xlabel=L"$\overline{v\prime w\prime}$ (m$^{2}$ s$^{-2}$)", xticks=LinearTicks(2))
        axwT = CairoMakie.Axis(l1[2,3], ylabel="z (m)", xlabel=L"$\overline{w\prime T\prime}$ (°C m s$^{-1}$)", xticks=LinearTicks(2))
        axwS = CairoMakie.Axis(l1[2,4], ylabel="z (m)", xlabel=L"$\overline{w\prime S\prime}$ (g kg$^{-1}$ m s$^{-1}$)", xticks=LinearTicks(2))
        axν = CairoMakie.Axis(l1[2,5], ylabel="z (m)", xlabel=L"Closure Diffusivities (m$^{2}$ s$^{-1}$)", xscale=log10)

        # train_res = training_results[1]
        # truth_train = training_data_plot.data[1]

        zC = training_data.data[1].metadata["zC"]
        zF = training_data.data[1].metadata["zF"]

        times = round.((truth_train.times .- truth_train.times[1]) ./ 24 ./ 60^2, digits=2)
        Qᵁ = truth_train.metadata["momentum_flux"]
        Qᵀ = truth_train.metadata["temperature_flux"]
        Qˢ = truth_train.metadata["salinity_flux"]
        f = truth_train.metadata["coriolis_parameter"]
        # times = truth_train.times
        Nt = length(times)

        ulim = (find_min(train_res.sols_dimensional.u, truth_train.profile.u.unscaled), 
                find_max(train_res.sols_dimensional.u, truth_train.profile.u.unscaled))
        vlim = (find_min(train_res.sols_dimensional.v, truth_train.profile.v.unscaled),
                find_max(train_res.sols_dimensional.v, truth_train.profile.v.unscaled))
        Tlim = (find_min(train_res.sols_dimensional.T, truth_train.profile.T.unscaled),
                find_max(train_res.sols_dimensional.T, truth_train.profile.T.unscaled))
        Slim = (find_min(train_res.sols_dimensional.S, truth_train.profile.S.unscaled),
                find_max(train_res.sols_dimensional.S, truth_train.profile.S.unscaled))
        σlim = (find_min(train_res.sols_dimensional.ρ, truth_train.profile.ρ.unscaled),
                find_max(train_res.sols_dimensional.ρ, truth_train.profile.ρ.unscaled))

        if !PW
            uwlim = (find_min(truth_train.flux.uw.column.unscaled[:, 50:end]), 
                     find_max(truth_train.flux.uw.column.unscaled[:, 50:end]))
            vwlim = (find_min(truth_train.flux.vw.column.unscaled[:, 50:end]),
                     find_max(truth_train.flux.vw.column.unscaled[:, 50:end]))
            wTlim = (find_min(truth_train.flux.wT.column.unscaled[:, 50:end]),
                     find_max(truth_train.flux.wT.column.unscaled[:, 50:end]))
            wSlim = (find_min(truth_train.flux.wS.column.unscaled[:, 50:end]),
                     find_max(truth_train.flux.wS.column.unscaled[:, 50:end]))
        else
            uwlim = (find_min(train_res.fluxes.uw.total[:, 1:end], truth_train.flux.uw.column.unscaled[:, 1:end]), 
                     find_max(train_res.fluxes.uw.total[:, 1:end], truth_train.flux.uw.column.unscaled[:, 1:end]))
            vwlim = (find_min(train_res.fluxes.vw.total[:, 1:end], truth_train.flux.vw.column.unscaled[:, 1:end]),
                     find_max(train_res.fluxes.vw.total[:, 1:end], truth_train.flux.vw.column.unscaled[:, 1:end]))
                     
            wTlim = (find_min(train_res.fluxes.wT.total[:, 1:end], truth_train.flux.wT.column.unscaled[:, 1:end]),
                     find_max(train_res.fluxes.wT.total[:, 1:end], truth_train.flux.wT.column.unscaled[:, 1:end]))
            wSlim = (find_min(train_res.fluxes.wS.total[:, 1:end], truth_train.flux.wS.column.unscaled[:, 1:end]),
                     find_max(train_res.fluxes.wS.total[:, 1:end], truth_train.flux.wS.column.unscaled[:, 1:end]))
        end

        νlim = (find_min(train_res.diffusivities.ν, train_res.diffusivities.κ),
        find_max(train_res.diffusivities.ν, train_res.diffusivities.κ))
        
        n = Observable(1)
        time_str = @lift "Local Base Closure Training: Qᵁ = $(Qᵁ) m² s⁻², Qᵀ = $(Qᵀ) m s⁻¹ °C, Qˢ = $(Qˢ) m s⁻¹ g kg⁻¹, f = $(f) s⁻¹, Time = $(round(times[$n], digits=1)) days"

        Label(fig[0, :], time_str, font=:bold)

        u_truthₙ = @lift truth_train.profile.u.unscaled[:, $n]
        v_truthₙ = @lift truth_train.profile.v.unscaled[:, $n]
        T_truthₙ = @lift truth_train.profile.T.unscaled[:, $n]
        S_truthₙ = @lift truth_train.profile.S.unscaled[:, $n]
        ρ_truthₙ = @lift truth_train.profile.ρ.unscaled[:, $n]

        u_predₙ = @lift train_res.sols_dimensional.u[:, $n]
        v_predₙ = @lift train_res.sols_dimensional.v[:, $n]
        T_predₙ = @lift train_res.sols_dimensional.T[:, $n]
        S_predₙ = @lift train_res.sols_dimensional.S[:, $n]
        ρ_predₙ = @lift train_res.sols_dimensional.ρ[:, $n]

        ν_predₙ = @lift train_res.diffusivities.ν[2:end-1, $n]
        κ_predₙ = @lift train_res.diffusivities.κ[2:end-1, $n]

        uw_truthₙ = @lift truth_train.flux.uw.column.unscaled[1:end-1, $n]
        vw_truthₙ = @lift truth_train.flux.vw.column.unscaled[1:end-1, $n]
        wT_truthₙ = @lift truth_train.flux.wT.column.unscaled[1:end-1, $n]
        wS_truthₙ = @lift truth_train.flux.wS.column.unscaled[1:end-1, $n]

        uw_predₙ = @lift train_res.fluxes.uw.total[1:end-1, $n]
        vw_predₙ = @lift train_res.fluxes.vw.total[1:end-1, $n]
        wT_predₙ = @lift train_res.fluxes.wT.total[1:end-1, $n]
        wS_predₙ = @lift train_res.fluxes.wS.total[1:end-1, $n]

        Ri_truthₙ = @lift clamp.(train_res.diffusivities.Ri_truth[2:end-1, $n], -10, 10)
        Ri_predₙ = @lift clamp.(train_res.diffusivities.Ri[2:end-1, $n], -10, 10)

        lines!(axu, u_truthₙ, zC, color=(colors[1], 0.5), linewidth=5, label="LES Solution")
        lines!(axv, v_truthₙ, zC, color=(colors[1], 0.5), linewidth=5, label="LES Solution")
        lines!(axT, T_truthₙ, zC, color=(colors[1], 0.5), linewidth=5, label="LES Solution")
        lines!(axS, S_truthₙ, zC, color=(colors[1], 0.5), linewidth=5, label="LES Solution")
        lines!(axσ, ρ_truthₙ, zC, color=(colors[1], 0.5), linewidth=5, label="LES Solution")

        lines!(axu, u_predₙ, zC, color=colors[2], linewidth=3, label="Local Base Closure")
        lines!(axv, v_predₙ, zC, color=colors[2], linewidth=3, label="Local Base Closure")
        lines!(axT, T_predₙ, zC, color=colors[2], linewidth=3, label="Local Base Closure")
        lines!(axS, S_predₙ, zC, color=colors[2], linewidth=3, label="Local Base Closure")
        lines!(axσ, ρ_predₙ, zC, color=colors[2], linewidth=3, label="Local Base Closure")

        lines!(axuw, uw_truthₙ, zF[1:end-1], color=(colors[1], 0.5), linewidth=5)
        lines!(axvw, vw_truthₙ, zF[1:end-1], color=(colors[1], 0.5), linewidth=5)
        lines!(axwT, wT_truthₙ, zF[1:end-1], color=(colors[1], 0.5), linewidth=5)
        lines!(axwS, wS_truthₙ, zF[1:end-1], color=(colors[1], 0.5), linewidth=5)

        lines!(axuw, uw_predₙ, zF[1:end-1], color=colors[2], linewidth=3)
        lines!(axvw, vw_predₙ, zF[1:end-1], color=colors[2], linewidth=3)
        lines!(axwT, wT_predₙ, zF[1:end-1], color=colors[2], linewidth=3)
        lines!(axwS, wS_predₙ, zF[1:end-1], color=colors[2], linewidth=3)

        lines!(axRi, Ri_truthₙ, zF[2:end-1], color=(colors[1], 0.5), linewidth=3)
        lines!(axRi, Ri_predₙ, zF[2:end-1], color=colors[2], linewidth=3)

        if !FC
            lines!(axν, ν_predₙ, zF[2:end-1], color=colors[3], linewidth=3, label=L"$\nu$")
        end

        lines!(axν, κ_predₙ, zF[2:end-1], color=colors[4], linewidth=3, label=L"$\kappa$")

        linkyaxes!(axu, axv, axT, axS, axσ, axν, axuw, axvw, axwT, axwS)

        hidedecorations!(axu, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axv, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axT, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axS, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axσ, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axν, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axuw, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axvw, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axwT, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axwS, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axRi, ticks=false, ticklabels=false, label=false, grid=false)

        hideydecorations!(axv, ticks=false)
        hideydecorations!(axT, ticks=false)
        hideydecorations!(axS, ticks=false)
        hideydecorations!(axσ, ticks=false)
        hideydecorations!(axν, ticks=false)
        hideydecorations!(axvw, ticks=false)
        hideydecorations!(axwT, ticks=false)
        hideydecorations!(axwS, ticks=false)
        hideydecorations!(axRi, ticks=false)

        hidexdecorations!(axu, ticks=false, ticklabels=false, label=false)
        hidexdecorations!(axv, ticks=false, ticklabels=false, label=false)
        hidexdecorations!(axT, ticks=false, ticklabels=false, label=false)
        hidexdecorations!(axS, ticks=false, ticklabels=false, label=false)
        hidexdecorations!(axσ, ticks=false, ticklabels=false, label=false)
        hidexdecorations!(axν, ticks=false, ticklabels=false, label=false)

        if FC
            xlims!(axu, (-0.1, 0.1))
            xlims!(axv, (-0.1, 0.1))
            xlims!(axuw, (-1e-4, 1e-4))
            xlims!(axvw, (-1e-4, 1e-4))
        else
            xlims!(axu, ulim)
            xlims!(axv, vlim)
            xlims!(axuw, uwlim)
            xlims!(axvw, vwlim)
        end

        xlims!(axT, Tlim)
        xlims!(axS, Slim)
        xlims!(axσ, σlim)
        xlims!(axwT, wTlim)
        xlims!(axwS, wSlim)
        xlims!(axν, νlim)
        xlims!(axRi, (-10, 10))

        Legend(l1[2, 6], axu, tellwidth=false, unique=true)
        axislegend(axν, position=:rb)

        # display(fig)
        CairoMakie.record(fig, "./slides_figures/localbaseclosure/training_$(i).mp4", 1:Nt, framerate=20) do nn
            n[] = nn
        end
    end
end
#%%
for (i, (train_res, truth_train)) in enumerate(zip(interpolating_results, interpolating_data_plot.data))
    @info "Plotting $i"

    if i in [3, 4, 5, 6, 11, 12, 17, 18, 19, 20]
        FC = true
    else
        FC = false
    end

    if i in [1, 2, 9, 10]
        PW = true
    else
        PW = false
    end

    with_theme(theme_latexfonts()) do
        fig = Figure(size=(1920, 980), fontsize=20)

        l1 = fig[1, 1:2] = GridLayout()

        axu = CairoMakie.Axis(l1[1,1], ylabel="z (m)", xlabel=L"$\overline{u}$ (m s$^{-1}$)")
        axv = CairoMakie.Axis(l1[1,2], ylabel="z (m)", xlabel=L"$\overline{v}$ (m s$^{-1}$)")
        axT = CairoMakie.Axis(l1[1,3], ylabel="z (m)", xlabel=L"$\overline{T}$ (°C)")
        axS = CairoMakie.Axis(l1[1,4], ylabel="z (m)", xlabel=L"$\overline{S}$ (g kg$^{-1}$)")
        axσ = CairoMakie.Axis(l1[1,5], ylabel="z (m)", xlabel=L"$\overline{\sigma}$ (kg m$^{-3}$)", xticks=LinearTicks(2))
        axRi = CairoMakie.Axis(l1[1,6], ylabel="z (m)", xlabel=L"Ri")
        axuw = CairoMakie.Axis(l1[2,1], ylabel="z (m)", xlabel=L"$\overline{u\prime w\prime}$ (m$^{2}$ s$^{-2}$)", xticks=LinearTicks(2))
        axvw = CairoMakie.Axis(l1[2,2], ylabel="z (m)", xlabel=L"$\overline{v\prime w\prime}$ (m$^{2}$ s$^{-2}$)", xticks=LinearTicks(2))
        axwT = CairoMakie.Axis(l1[2,3], ylabel="z (m)", xlabel=L"$\overline{w\prime T\prime}$ (°C m s$^{-1}$)", xticks=LinearTicks(2))
        axwS = CairoMakie.Axis(l1[2,4], ylabel="z (m)", xlabel=L"$\overline{w\prime S\prime}$ (g kg$^{-1}$ m s$^{-1}$)", xticks=LinearTicks(2))
        axν = CairoMakie.Axis(l1[2,5], ylabel="z (m)", xlabel=L"Closure Diffusivities (m$^{2}$ s$^{-1}$)", xscale=log10)

        # train_res = training_results[1]
        # truth_train = training_data_plot.data[1]

        zC = training_data.data[1].metadata["zC"]
        zF = training_data.data[1].metadata["zF"]

        times = round.((truth_train.times .- truth_train.times[1]) ./ 24 ./ 60^2, digits=2)
        Qᵁ = truth_train.metadata["momentum_flux"]
        Qᵀ = truth_train.metadata["temperature_flux"]
        Qˢ = truth_train.metadata["salinity_flux"]
        f = truth_train.metadata["coriolis_parameter"]
        # times = truth_train.times
        Nt = length(times)

        ulim = (find_min(train_res.sols_dimensional.u, truth_train.profile.u.unscaled), 
                find_max(train_res.sols_dimensional.u, truth_train.profile.u.unscaled))
        vlim = (find_min(train_res.sols_dimensional.v, truth_train.profile.v.unscaled),
                find_max(train_res.sols_dimensional.v, truth_train.profile.v.unscaled))
        Tlim = (find_min(train_res.sols_dimensional.T, truth_train.profile.T.unscaled),
                find_max(train_res.sols_dimensional.T, truth_train.profile.T.unscaled))
        Slim = (find_min(train_res.sols_dimensional.S, truth_train.profile.S.unscaled),
                find_max(train_res.sols_dimensional.S, truth_train.profile.S.unscaled))
        σlim = (find_min(train_res.sols_dimensional.ρ, truth_train.profile.ρ.unscaled),
                find_max(train_res.sols_dimensional.ρ, truth_train.profile.ρ.unscaled))

        if !PW
            uwlim = (find_min(truth_train.flux.uw.column.unscaled[:, 50:end]), 
                     find_max(truth_train.flux.uw.column.unscaled[:, 50:end]))
            vwlim = (find_min(truth_train.flux.vw.column.unscaled[:, 50:end]),
                     find_max(truth_train.flux.vw.column.unscaled[:, 50:end]))
            wTlim = (find_min(truth_train.flux.wT.column.unscaled[:, 50:end]),
                     find_max(truth_train.flux.wT.column.unscaled[:, 50:end]))
            wSlim = (find_min(truth_train.flux.wS.column.unscaled[:, 50:end]),
                     find_max(truth_train.flux.wS.column.unscaled[:, 50:end]))
        else
            uwlim = (find_min(train_res.fluxes.uw.total[:, 1:end], truth_train.flux.uw.column.unscaled[:, 1:end]), 
                     find_max(train_res.fluxes.uw.total[:, 1:end], truth_train.flux.uw.column.unscaled[:, 1:end]))
            vwlim = (find_min(train_res.fluxes.vw.total[:, 1:end], truth_train.flux.vw.column.unscaled[:, 1:end]),
                     find_max(train_res.fluxes.vw.total[:, 1:end], truth_train.flux.vw.column.unscaled[:, 1:end]))
                     
            wTlim = (find_min(train_res.fluxes.wT.total[:, 1:end], truth_train.flux.wT.column.unscaled[:, 1:end]),
                     find_max(train_res.fluxes.wT.total[:, 1:end], truth_train.flux.wT.column.unscaled[:, 1:end]))
            wSlim = (find_min(train_res.fluxes.wS.total[:, 1:end], truth_train.flux.wS.column.unscaled[:, 1:end]),
                     find_max(train_res.fluxes.wS.total[:, 1:end], truth_train.flux.wS.column.unscaled[:, 1:end]))
        end

        νlim = (find_min(train_res.diffusivities.ν, train_res.diffusivities.κ),
                find_max(train_res.diffusivities.ν, train_res.diffusivities.κ))
        
        
        n = Observable(1)
        time_str = @lift "Local Base Closure Validation: Qᵁ = $(Qᵁ) m² s⁻², Qᵀ = $(Qᵀ) m s⁻¹ °C, Qˢ = $(Qˢ) m s⁻¹ g kg⁻¹, f = $(f) s⁻¹, Time = $(round(times[$n], digits=1)) days"

        Label(fig[0, :], time_str, font=:bold)

        u_truthₙ = @lift truth_train.profile.u.unscaled[:, $n]
        v_truthₙ = @lift truth_train.profile.v.unscaled[:, $n]
        T_truthₙ = @lift truth_train.profile.T.unscaled[:, $n]
        S_truthₙ = @lift truth_train.profile.S.unscaled[:, $n]
        ρ_truthₙ = @lift truth_train.profile.ρ.unscaled[:, $n]

        u_predₙ = @lift train_res.sols_dimensional.u[:, $n]
        v_predₙ = @lift train_res.sols_dimensional.v[:, $n]
        T_predₙ = @lift train_res.sols_dimensional.T[:, $n]
        S_predₙ = @lift train_res.sols_dimensional.S[:, $n]
        ρ_predₙ = @lift train_res.sols_dimensional.ρ[:, $n]

        ν_predₙ = @lift train_res.diffusivities.ν[2:end-1, $n]
        κ_predₙ = @lift train_res.diffusivities.κ[2:end-1, $n]

        uw_truthₙ = @lift truth_train.flux.uw.column.unscaled[1:end-1, $n]
        vw_truthₙ = @lift truth_train.flux.vw.column.unscaled[1:end-1, $n]
        wT_truthₙ = @lift truth_train.flux.wT.column.unscaled[1:end-1, $n]
        wS_truthₙ = @lift truth_train.flux.wS.column.unscaled[1:end-1, $n]

        uw_predₙ = @lift train_res.fluxes.uw.total[1:end-1, $n]
        vw_predₙ = @lift train_res.fluxes.vw.total[1:end-1, $n]
        wT_predₙ = @lift train_res.fluxes.wT.total[1:end-1, $n]
        wS_predₙ = @lift train_res.fluxes.wS.total[1:end-1, $n]

        Ri_truthₙ = @lift clamp.(train_res.diffusivities.Ri_truth[2:end-1, $n], -10, 10)
        Ri_predₙ = @lift clamp.(train_res.diffusivities.Ri[2:end-1, $n], -10, 10)

        lines!(axu, u_truthₙ, zC, color=(colors[1], 0.5), linewidth=5, label="LES Solution")
        lines!(axv, v_truthₙ, zC, color=(colors[1], 0.5), linewidth=5, label="LES Solution")
        lines!(axT, T_truthₙ, zC, color=(colors[1], 0.5), linewidth=5, label="LES Solution")
        lines!(axS, S_truthₙ, zC, color=(colors[1], 0.5), linewidth=5, label="LES Solution")
        lines!(axσ, ρ_truthₙ, zC, color=(colors[1], 0.5), linewidth=5, label="LES Solution")

        lines!(axu, u_predₙ, zC, color=colors[2], linewidth=3, label="Local Base Closure")
        lines!(axv, v_predₙ, zC, color=colors[2], linewidth=3, label="Local Base Closure")
        lines!(axT, T_predₙ, zC, color=colors[2], linewidth=3, label="Local Base Closure")
        lines!(axS, S_predₙ, zC, color=colors[2], linewidth=3, label="Local Base Closure")
        lines!(axσ, ρ_predₙ, zC, color=colors[2], linewidth=3, label="Local Base Closure")

        lines!(axuw, uw_truthₙ, zF[1:end-1], color=(colors[1], 0.5), linewidth=5)
        lines!(axvw, vw_truthₙ, zF[1:end-1], color=(colors[1], 0.5), linewidth=5)
        lines!(axwT, wT_truthₙ, zF[1:end-1], color=(colors[1], 0.5), linewidth=5)
        lines!(axwS, wS_truthₙ, zF[1:end-1], color=(colors[1], 0.5), linewidth=5)

        lines!(axuw, uw_predₙ, zF[1:end-1], color=colors[2], linewidth=3)
        lines!(axvw, vw_predₙ, zF[1:end-1], color=colors[2], linewidth=3)
        lines!(axwT, wT_predₙ, zF[1:end-1], color=colors[2], linewidth=3)
        lines!(axwS, wS_predₙ, zF[1:end-1], color=colors[2], linewidth=3)

        lines!(axRi, Ri_truthₙ, zF[2:end-1], color=(colors[1], 0.5), linewidth=3)
        lines!(axRi, Ri_predₙ, zF[2:end-1], color=colors[2], linewidth=3)

        if !FC
            lines!(axν, ν_predₙ, zF[2:end-1], color=colors[3], linewidth=3, label=L"$\nu$")
        end

        lines!(axν, κ_predₙ, zF[2:end-1], color=colors[4], linewidth=3, label=L"$\kappa$")

        linkyaxes!(axu, axv, axT, axS, axσ, axν, axuw, axvw, axwT, axwS, axRi)

        hidedecorations!(axu, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axv, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axT, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axS, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axσ, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axν, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axuw, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axvw, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axwT, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axwS, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axRi, ticks=false, ticklabels=false, label=false, grid=false)

        hideydecorations!(axv, ticks=false)
        hideydecorations!(axT, ticks=false)
        hideydecorations!(axS, ticks=false)
        hideydecorations!(axσ, ticks=false)
        hideydecorations!(axν, ticks=false)
        hideydecorations!(axvw, ticks=false)
        hideydecorations!(axwT, ticks=false)
        hideydecorations!(axwS, ticks=false)
        hideydecorations!(axRi, ticks=false)

        hidexdecorations!(axu, ticks=false, ticklabels=false, label=false)
        hidexdecorations!(axv, ticks=false, ticklabels=false, label=false)
        hidexdecorations!(axT, ticks=false, ticklabels=false, label=false)
        hidexdecorations!(axS, ticks=false, ticklabels=false, label=false)
        hidexdecorations!(axσ, ticks=false, ticklabels=false, label=false)
        hidexdecorations!(axν, ticks=false, ticklabels=false, label=false)

        if FC
            xlims!(axu, (-0.1, 0.1))
            xlims!(axv, (-0.1, 0.1))
            xlims!(axuw, (-1e-4, 1e-4))
            xlims!(axvw, (-1e-4, 1e-4))
        else
            xlims!(axu, ulim)
            xlims!(axv, vlim)
            xlims!(axuw, uwlim)
            xlims!(axvw, vwlim)
        end

        xlims!(axT, Tlim)
        xlims!(axS, Slim)
        xlims!(axσ, σlim)
        xlims!(axwT, wTlim)
        xlims!(axwS, wSlim)
        xlims!(axν, νlim)
        xlims!(axRi, (-10, 10))

        Legend(l1[2, 6], axu, tellwidth=false, unique=true)
        axislegend(axν, position=:rb)

        # display(fig)
        CairoMakie.record(fig, "./slides_figures/localbaseclosure/validation_$(i).mp4", 1:Nt, framerate=20) do nn
            n[] = nn
        end
    end
end
#%%
colors = colorschemes[:Set1_9]

FCs_training = [3, 4, 5, 6, 13, 14, 15, 16]

for i in FCs_training
    @info "Plotting $i"

    truth_train = training_data_plot.data[i]
    train_res = training_results[i]

    with_theme(theme_latexfonts()) do
        fig = Figure(size=(1200, 550), fontsize=20)

        l1 = fig[1, 1:2] = GridLayout()

        axT = CairoMakie.Axis(l1[1,1], ylabel="z (m)", xlabel=L"$\overline{T}$ (°C)")
        axS = CairoMakie.Axis(l1[1,2], ylabel="z (m)", xlabel=L"$\overline{S}$ (g kg$^{-1}$)")
        axσ = CairoMakie.Axis(l1[1,3], ylabel="z (m)", xlabel=L"$\overline{\sigma}$ (kg m$^{-3}$)", xticks=LinearTicks(2))

        zC = training_data.data[1].metadata["zC"]
        zF = training_data.data[1].metadata["zF"]

        times = round.((truth_train.times .- truth_train.times[1]) ./ 24 ./ 60^2, digits=2)
        Qᵁ = truth_train.metadata["momentum_flux"]
        Qᵀ = truth_train.metadata["temperature_flux"]
        Qˢ = truth_train.metadata["salinity_flux"]
        f = truth_train.metadata["coriolis_parameter"]
        # times = truth_train.times
        Nt = length(times)

        Tlim = (find_min(train_res.sols_dimensional.T, truth_train.profile.T.unscaled),
                find_max(train_res.sols_dimensional.T, truth_train.profile.T.unscaled))
        Slim = (find_min(train_res.sols_dimensional.S, truth_train.profile.S.unscaled),
                find_max(train_res.sols_dimensional.S, truth_train.profile.S.unscaled))
        σlim = (find_min(train_res.sols_dimensional.ρ, truth_train.profile.ρ.unscaled),
                find_max(train_res.sols_dimensional.ρ, truth_train.profile.ρ.unscaled))

        n = Observable(1)
        # time_str = @lift "Local Base Closure Training: Qᵁ = $(Qᵁ) m² s⁻², Qᵀ = $(Qᵀ) m s⁻¹ °C, Qˢ = $(Qˢ) m s⁻¹ g kg⁻¹, f = $(f) s⁻¹ \nTime = $(round(times[$n], digits=1)) days"
        time_str = @lift "Time = $(round(times[$n], digits=1)) days"

        Label(fig[0, :], time_str, font=:bold)

        T_truthₙ = @lift truth_train.profile.T.unscaled[:, $n]
        S_truthₙ = @lift truth_train.profile.S.unscaled[:, $n]
        ρ_truthₙ = @lift truth_train.profile.ρ.unscaled[:, $n]

        T_predₙ = @lift train_res.sols_dimensional.T[:, $n]
        S_predₙ = @lift train_res.sols_dimensional.S[:, $n]
        ρ_predₙ = @lift train_res.sols_dimensional.ρ[:, $n]

        lines!(axT, T_truthₙ, zC, color=(colors[1], 0.5), linewidth=5, label="LES Solution")
        lines!(axS, S_truthₙ, zC, color=(colors[1], 0.5), linewidth=5, label="LES Solution")
        lines!(axσ, ρ_truthₙ, zC, color=(colors[1], 0.5), linewidth=5, label="LES Solution")

        lines!(axT, T_predₙ, zC, color=colors[2], linewidth=3, label="Local Base Closure")
        lines!(axS, S_predₙ, zC, color=colors[2], linewidth=3, label="Local Base Closure")
        lines!(axσ, ρ_predₙ, zC, color=colors[2], linewidth=3, label="Local Base Closure")

        linkyaxes!(axT, axS, axσ)

        hidedecorations!(axT, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axS, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axσ, ticks=false, ticklabels=false, label=false)

        hideydecorations!(axS, ticks=false)
        hideydecorations!(axσ, ticks=false)

        hidexdecorations!(axT, ticks=false, ticklabels=false, label=false)
        hidexdecorations!(axS, ticks=false, ticklabels=false, label=false)
        hidexdecorations!(axσ, ticks=false, ticklabels=false, label=false)

        xlims!(axT, Tlim)
        xlims!(axS, Slim)
        xlims!(axσ, σlim)

        Legend(l1[2, :], axT, tellwidth=false, orientation=:horizontal)

        # display(fig)

        CairoMakie.record(fig, "./slides_figures/localbaseclosure_simple/training_$(i).mp4", 1:Nt, framerate=20) do nn
            n[] = nn
        end
    end
end
#%%
notFCs_training = setdiff(1:20, FCs_training)

for i in notFCs_training
    @info "Plotting $i"

    truth_train = training_data_plot.data[i]
    train_res = training_results[i]

    with_theme(theme_latexfonts()) do
        fig = Figure(size=(1920, 550), fontsize=20)

        l1 = fig[1, 1:2] = GridLayout()

        axu = CairoMakie.Axis(l1[1,1], ylabel="z (m)", xlabel=L"$\overline{u}$ (m s$^{-1}$)")
        axv = CairoMakie.Axis(l1[1,2], ylabel="z (m)", xlabel=L"$\overline{v}$ (m s$^{-1}$)")
        axT = CairoMakie.Axis(l1[1,3], ylabel="z (m)", xlabel=L"$\overline{T}$ (°C)")
        axS = CairoMakie.Axis(l1[1,4], ylabel="z (m)", xlabel=L"$\overline{S}$ (g kg$^{-1}$)")
        axσ = CairoMakie.Axis(l1[1,5], ylabel="z (m)", xlabel=L"$\overline{\sigma}$ (kg m$^{-3}$)", xticks=LinearTicks(2))

        zC = training_data.data[1].metadata["zC"]
        zF = training_data.data[1].metadata["zF"]

        times = round.((truth_train.times .- truth_train.times[1]) ./ 24 ./ 60^2, digits=2)
        Qᵁ = truth_train.metadata["momentum_flux"]
        Qᵀ = truth_train.metadata["temperature_flux"]
        Qˢ = truth_train.metadata["salinity_flux"]
        f = truth_train.metadata["coriolis_parameter"]
        # times = truth_train.times
        Nt = length(times)

        ulim = (find_min(train_res.sols_dimensional.u, truth_train.profile.u.unscaled), 
                find_max(train_res.sols_dimensional.u, truth_train.profile.u.unscaled))
        vlim = (find_min(train_res.sols_dimensional.v, truth_train.profile.v.unscaled),
                find_max(train_res.sols_dimensional.v, truth_train.profile.v.unscaled))
        Tlim = (find_min(train_res.sols_dimensional.T, truth_train.profile.T.unscaled),
                find_max(train_res.sols_dimensional.T, truth_train.profile.T.unscaled))
        Slim = (find_min(train_res.sols_dimensional.S, truth_train.profile.S.unscaled),
                find_max(train_res.sols_dimensional.S, truth_train.profile.S.unscaled))
        σlim = (find_min(train_res.sols_dimensional.ρ, truth_train.profile.ρ.unscaled),
                find_max(train_res.sols_dimensional.ρ, truth_train.profile.ρ.unscaled))

        n = Observable(1)
        # time_str = @lift "Local Base Closure Training: Qᵁ = $(Qᵁ) m² s⁻², Qᵀ = $(Qᵀ) m s⁻¹ °C, Qˢ = $(Qˢ) m s⁻¹ g kg⁻¹, f = $(f) s⁻¹, Time = $(round(times[$n], digits=1)) days"
        time_str = @lift "Time = $(round(times[$n], digits=1)) days"

        Label(fig[0, :], time_str, font=:bold)

        u_truthₙ = @lift truth_train.profile.u.unscaled[:, $n]
        v_truthₙ = @lift truth_train.profile.v.unscaled[:, $n]
        T_truthₙ = @lift truth_train.profile.T.unscaled[:, $n]
        S_truthₙ = @lift truth_train.profile.S.unscaled[:, $n]
        ρ_truthₙ = @lift truth_train.profile.ρ.unscaled[:, $n]

        u_predₙ = @lift train_res.sols_dimensional.u[:, $n]
        v_predₙ = @lift train_res.sols_dimensional.v[:, $n]
        T_predₙ = @lift train_res.sols_dimensional.T[:, $n]
        S_predₙ = @lift train_res.sols_dimensional.S[:, $n]
        ρ_predₙ = @lift train_res.sols_dimensional.ρ[:, $n]

        ν_predₙ = @lift train_res.diffusivities.ν[2:end-1, $n]
        κ_predₙ = @lift train_res.diffusivities.κ[2:end-1, $n]

        lines!(axu, u_truthₙ, zC, color=(colors[1], 0.5), linewidth=5, label="LES Solution")
        lines!(axv, v_truthₙ, zC, color=(colors[1], 0.5), linewidth=5, label="LES Solution")
        lines!(axT, T_truthₙ, zC, color=(colors[1], 0.5), linewidth=5, label="LES Solution")
        lines!(axS, S_truthₙ, zC, color=(colors[1], 0.5), linewidth=5, label="LES Solution")
        lines!(axσ, ρ_truthₙ, zC, color=(colors[1], 0.5), linewidth=5, label="LES Solution")

        lines!(axu, u_predₙ, zC, color=colors[2], linewidth=3, label="Local Base Closure")
        lines!(axv, v_predₙ, zC, color=colors[2], linewidth=3, label="Local Base Closure")
        lines!(axT, T_predₙ, zC, color=colors[2], linewidth=3, label="Local Base Closure")
        lines!(axS, S_predₙ, zC, color=colors[2], linewidth=3, label="Local Base Closure")
        lines!(axσ, ρ_predₙ, zC, color=colors[2], linewidth=3, label="Local Base Closure")

        linkyaxes!(axu, axv, axT, axS, axσ)

        hidedecorations!(axu, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axv, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axT, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axS, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axσ, ticks=false, ticklabels=false, label=false)

        hideydecorations!(axv, ticks=false)
        hideydecorations!(axT, ticks=false)
        hideydecorations!(axS, ticks=false)
        hideydecorations!(axσ, ticks=false)

        hidexdecorations!(axu, ticks=false, ticklabels=false, label=false)
        hidexdecorations!(axv, ticks=false, ticklabels=false, label=false)
        hidexdecorations!(axT, ticks=false, ticklabels=false, label=false)
        hidexdecorations!(axS, ticks=false, ticklabels=false, label=false)
        hidexdecorations!(axσ, ticks=false, ticklabels=false, label=false)

        xlims!(axu, ulim)
        xlims!(axv, vlim)
        xlims!(axT, Tlim)
        xlims!(axS, Slim)
        xlims!(axσ, σlim)

        Legend(l1[2, :], axu, tellwidth=false, orientation=:horizontal)

        # display(fig)
        CairoMakie.record(fig, "./slides_figures/localbaseclosure_simple/training_$(i).mp4", 1:Nt, framerate=20) do nn
            n[] = nn
        end
    end
end
#%%
FCs_interpolating = [3, 4, 5, 6, 11, 12, 17, 18, 19, 20]

for i in FCs_interpolating
    @info "Plotting $i"

    truth_train = interpolating_data_plot.data[i]
    train_res = interpolating_results[i]

    with_theme(theme_latexfonts()) do
        fig = Figure(size=(1200, 550), fontsize=20)

        l1 = fig[1, 1:2] = GridLayout()

        axT = CairoMakie.Axis(l1[1,1], ylabel="z (m)", xlabel=L"$\overline{T}$ (°C)")
        axS = CairoMakie.Axis(l1[1,2], ylabel="z (m)", xlabel=L"$\overline{S}$ (g kg$^{-1}$)")
        axσ = CairoMakie.Axis(l1[1,3], ylabel="z (m)", xlabel=L"$\overline{\sigma}$ (kg m$^{-3}$)", xticks=LinearTicks(2))

        zC = training_data.data[1].metadata["zC"]
        zF = training_data.data[1].metadata["zF"]

        times = round.((truth_train.times .- truth_train.times[1]) ./ 24 ./ 60^2, digits=2)
        Qᵁ = truth_train.metadata["momentum_flux"]
        Qᵀ = truth_train.metadata["temperature_flux"]
        Qˢ = truth_train.metadata["salinity_flux"]
        f = truth_train.metadata["coriolis_parameter"]
        # times = truth_train.times
        Nt = length(times)

        Tlim = (find_min(train_res.sols_dimensional.T, truth_train.profile.T.unscaled),
                find_max(train_res.sols_dimensional.T, truth_train.profile.T.unscaled))
        Slim = (find_min(train_res.sols_dimensional.S, truth_train.profile.S.unscaled),
                find_max(train_res.sols_dimensional.S, truth_train.profile.S.unscaled))
        σlim = (find_min(train_res.sols_dimensional.ρ, truth_train.profile.ρ.unscaled),
                find_max(train_res.sols_dimensional.ρ, truth_train.profile.ρ.unscaled))

        n = Observable(1)
        # time_str = @lift "Local Base Closure Training: Qᵁ = $(Qᵁ) m² s⁻², Qᵀ = $(Qᵀ) m s⁻¹ °C, Qˢ = $(Qˢ) m s⁻¹ g kg⁻¹, f = $(f) s⁻¹ \nTime = $(round(times[$n], digits=1)) days"
        time_str = @lift "Time = $(round(times[$n], digits=1)) days"

        Label(fig[0, :], time_str, font=:bold)

        T_truthₙ = @lift truth_train.profile.T.unscaled[:, $n]
        S_truthₙ = @lift truth_train.profile.S.unscaled[:, $n]
        ρ_truthₙ = @lift truth_train.profile.ρ.unscaled[:, $n]

        T_predₙ = @lift train_res.sols_dimensional.T[:, $n]
        S_predₙ = @lift train_res.sols_dimensional.S[:, $n]
        ρ_predₙ = @lift train_res.sols_dimensional.ρ[:, $n]

        lines!(axT, T_truthₙ, zC, color=(colors[1], 0.5), linewidth=5, label="LES Solution")
        lines!(axS, S_truthₙ, zC, color=(colors[1], 0.5), linewidth=5, label="LES Solution")
        lines!(axσ, ρ_truthₙ, zC, color=(colors[1], 0.5), linewidth=5, label="LES Solution")

        lines!(axT, T_predₙ, zC, color=colors[2], linewidth=3, label="Local Base Closure")
        lines!(axS, S_predₙ, zC, color=colors[2], linewidth=3, label="Local Base Closure")
        lines!(axσ, ρ_predₙ, zC, color=colors[2], linewidth=3, label="Local Base Closure")

        linkyaxes!(axT, axS, axσ)

        hidedecorations!(axT, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axS, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axσ, ticks=false, ticklabels=false, label=false)

        hideydecorations!(axS, ticks=false)
        hideydecorations!(axσ, ticks=false)

        hidexdecorations!(axT, ticks=false, ticklabels=false, label=false)
        hidexdecorations!(axS, ticks=false, ticklabels=false, label=false)
        hidexdecorations!(axσ, ticks=false, ticklabels=false, label=false)

        xlims!(axT, Tlim)
        xlims!(axS, Slim)
        xlims!(axσ, σlim)

        Legend(l1[2, :], axT, tellwidth=false, orientation=:horizontal)

        # display(fig)

        CairoMakie.record(fig, "./slides_figures/localbaseclosure_simple/validation_$(i).mp4", 1:Nt, framerate=20) do nn
            n[] = nn
        end
    end
end
#%%
notFCs_interpolating = setdiff(1:22, FCs_interpolating)

for i in notFCs_interpolating
    @info "Plotting $i"

    truth_train = interpolating_data_plot.data[i]
    train_res = interpolating_results[i]

    with_theme(theme_latexfonts()) do
        fig = Figure(size=(1920, 550), fontsize=20)

        l1 = fig[1, 1:2] = GridLayout()

        axu = CairoMakie.Axis(l1[1,1], ylabel="z (m)", xlabel=L"$\overline{u}$ (m s$^{-1}$)")
        axv = CairoMakie.Axis(l1[1,2], ylabel="z (m)", xlabel=L"$\overline{v}$ (m s$^{-1}$)")
        axT = CairoMakie.Axis(l1[1,3], ylabel="z (m)", xlabel=L"$\overline{T}$ (°C)")
        axS = CairoMakie.Axis(l1[1,4], ylabel="z (m)", xlabel=L"$\overline{S}$ (g kg$^{-1}$)")
        axσ = CairoMakie.Axis(l1[1,5], ylabel="z (m)", xlabel=L"$\overline{\sigma}$ (kg m$^{-3}$)", xticks=LinearTicks(2))

        zC = training_data.data[1].metadata["zC"]
        zF = training_data.data[1].metadata["zF"]

        times = round.((truth_train.times .- truth_train.times[1]) ./ 24 ./ 60^2, digits=2)
        Qᵁ = truth_train.metadata["momentum_flux"]
        Qᵀ = truth_train.metadata["temperature_flux"]
        Qˢ = truth_train.metadata["salinity_flux"]
        f = truth_train.metadata["coriolis_parameter"]
        # times = truth_train.times
        Nt = length(times)

        ulim = (find_min(train_res.sols_dimensional.u, truth_train.profile.u.unscaled), 
                find_max(train_res.sols_dimensional.u, truth_train.profile.u.unscaled))
        vlim = (find_min(train_res.sols_dimensional.v, truth_train.profile.v.unscaled),
                find_max(train_res.sols_dimensional.v, truth_train.profile.v.unscaled))
        Tlim = (find_min(train_res.sols_dimensional.T, truth_train.profile.T.unscaled),
                find_max(train_res.sols_dimensional.T, truth_train.profile.T.unscaled))
        Slim = (find_min(train_res.sols_dimensional.S, truth_train.profile.S.unscaled),
                find_max(train_res.sols_dimensional.S, truth_train.profile.S.unscaled))
        σlim = (find_min(train_res.sols_dimensional.ρ, truth_train.profile.ρ.unscaled),
                find_max(train_res.sols_dimensional.ρ, truth_train.profile.ρ.unscaled))

        n = Observable(1)
        # time_str = @lift "Local Base Closure Training: Qᵁ = $(Qᵁ) m² s⁻², Qᵀ = $(Qᵀ) m s⁻¹ °C, Qˢ = $(Qˢ) m s⁻¹ g kg⁻¹, f = $(f) s⁻¹, Time = $(round(times[$n], digits=1)) days"
        time_str = @lift "Time = $(round(times[$n], digits=1)) days"

        Label(fig[0, :], time_str, font=:bold)

        u_truthₙ = @lift truth_train.profile.u.unscaled[:, $n]
        v_truthₙ = @lift truth_train.profile.v.unscaled[:, $n]
        T_truthₙ = @lift truth_train.profile.T.unscaled[:, $n]
        S_truthₙ = @lift truth_train.profile.S.unscaled[:, $n]
        ρ_truthₙ = @lift truth_train.profile.ρ.unscaled[:, $n]

        u_predₙ = @lift train_res.sols_dimensional.u[:, $n]
        v_predₙ = @lift train_res.sols_dimensional.v[:, $n]
        T_predₙ = @lift train_res.sols_dimensional.T[:, $n]
        S_predₙ = @lift train_res.sols_dimensional.S[:, $n]
        ρ_predₙ = @lift train_res.sols_dimensional.ρ[:, $n]

        ν_predₙ = @lift train_res.diffusivities.ν[2:end-1, $n]
        κ_predₙ = @lift train_res.diffusivities.κ[2:end-1, $n]

        lines!(axu, u_truthₙ, zC, color=(colors[1], 0.5), linewidth=5, label="LES Solution")
        lines!(axv, v_truthₙ, zC, color=(colors[1], 0.5), linewidth=5, label="LES Solution")
        lines!(axT, T_truthₙ, zC, color=(colors[1], 0.5), linewidth=5, label="LES Solution")
        lines!(axS, S_truthₙ, zC, color=(colors[1], 0.5), linewidth=5, label="LES Solution")
        lines!(axσ, ρ_truthₙ, zC, color=(colors[1], 0.5), linewidth=5, label="LES Solution")

        lines!(axu, u_predₙ, zC, color=colors[2], linewidth=3, label="Local Base Closure")
        lines!(axv, v_predₙ, zC, color=colors[2], linewidth=3, label="Local Base Closure")
        lines!(axT, T_predₙ, zC, color=colors[2], linewidth=3, label="Local Base Closure")
        lines!(axS, S_predₙ, zC, color=colors[2], linewidth=3, label="Local Base Closure")
        lines!(axσ, ρ_predₙ, zC, color=colors[2], linewidth=3, label="Local Base Closure")

        linkyaxes!(axu, axv, axT, axS, axσ)

        hidedecorations!(axu, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axv, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axT, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axS, ticks=false, ticklabels=false, label=false)
        hidedecorations!(axσ, ticks=false, ticklabels=false, label=false)

        hideydecorations!(axv, ticks=false)
        hideydecorations!(axT, ticks=false)
        hideydecorations!(axS, ticks=false)
        hideydecorations!(axσ, ticks=false)

        hidexdecorations!(axu, ticks=false, ticklabels=false, label=false)
        hidexdecorations!(axv, ticks=false, ticklabels=false, label=false)
        hidexdecorations!(axT, ticks=false, ticklabels=false, label=false)
        hidexdecorations!(axS, ticks=false, ticklabels=false, label=false)
        hidexdecorations!(axσ, ticks=false, ticklabels=false, label=false)

        xlims!(axu, ulim)
        xlims!(axv, vlim)
        xlims!(axT, Tlim)
        xlims!(axS, Slim)
        xlims!(axσ, σlim)

        Legend(l1[2, :], axu, tellwidth=false, orientation=:horizontal)

        # display(fig)
        CairoMakie.record(fig, "./slides_figures/localbaseclosure_simple/interpolating_$(i).mp4", 1:Nt, framerate=20) do nn
            n[] = nn
        end
    end
end

#%%