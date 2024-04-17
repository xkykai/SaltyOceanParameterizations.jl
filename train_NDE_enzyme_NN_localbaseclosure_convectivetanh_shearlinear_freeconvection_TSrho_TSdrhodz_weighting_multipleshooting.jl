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

function parse_commandline()
    s = ArgParseSettings()
  
    @add_arg_table! s begin
    #   "--loss_type"
    #     help = "Loss function used"
    #     arg_type = String
    #     default = "mse"
      "--hidden_layer_size"
        help = "Size of hidden layer"
        arg_type = Int64
        default = 512
      "--hidden_layer"
        help = "Number of hidden layers"
        arg_type = Int64
        default = 2
      "--activation"
        help = "Activation function"
        arg_type = String
        default = "swish"
      "--S_scaling"
        help = "Scaling factor for S"
        arg_type = Float64
        default = 1.0
    end
    return parse_args(s)
end

args = parse_commandline()

function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

const hidden_layer_size = args["hidden_layer_size"]
const N_hidden_layer = args["hidden_layer"]

if args["activation"] == "relu"
    const activation = relu
elseif args["activation"] == "leakyrelu"
    const activation = leakyrelu
elseif args["activation"] == "swish"
    const activation = swish
else
    error("Activation function not recognized")
end

const S_scaling = args["S_scaling"]

FILE_DIR = "./training_output/multipleshooting/NDE_enzyme_$(args["hidden_layer"])layer_$(args["hidden_layer_size"])_$(args["activation"])_$(S_scaling)Sscaling"
mkpath(FILE_DIR)
@info FILE_DIR

LES_FILE_DIRS = [
    # "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.0005_QT_0.0_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    # "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.0002_QT_0.0_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0005_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0001_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0_QS_-5.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0_QS_-2.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    # "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.00015_QT_0.00045_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    # "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.0004_QT_0.00015_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    # "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.00015_QT_0.0_QS_-4.5e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    # "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.0004_QT_0.0_QS_-2.5e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",


    # "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.0005_QT_0.0_QS_0.0_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    # "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.0002_QT_0.0_QS_0.0_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_0.0_QT_0.0005_QS_0.0_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_0.0_QT_0.0001_QS_0.0_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_0.0_QT_0.0_QS_-5.0e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_0.0_QT_0.0_QS_-2.0e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    # "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.00015_QT_0.00045_QS_0.0_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    # "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.0004_QT_0.00015_QS_0.0_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    # "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.00015_QT_0.0_QS_-4.5e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    # "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.0004_QT_0.0_QS_-2.5e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
]

BASECLOSURE_FILE_DIR = "./training_output/localbaseclosure_convectivetanh_shearlinear_TSrho_EKI/training_results.jld2"

field_datasets = [FieldDataset(FILE_DIR, backend=OnDisk()) for FILE_DIR in LES_FILE_DIRS]

ps_baseclosure = jldopen(BASECLOSURE_FILE_DIR, "r")["u"]

timeframes = [25:10:length(data["ubar"].times) for data in field_datasets]
time_indices = [1:length(timeframe) for timeframe in timeframes]


split_length = 1
split_timeframes = vcat([[time=index[i:i+split_length] for i in 1:split_length:length(index)-split_length] for index in time_indices]...)
simulation_indices_split = vcat([vcat([j for _ in 1:split_length:length(index)-split_length]...) for (j, index) in enumerate(time_indices)]...)

full_timeframes = [25:length(data["ubar"].times) for data in field_datasets]
train_data = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, timeframes)
coarse_size = 32

x₀s = [(; T=data.profile.T.scaled[:, 1], S=data.profile.S.scaled[:, 1]) for data in train_data.data]
x₀s_split = [(; T=train_data.data[sim].profile.T.scaled[:, time[1]], 
                S=train_data.data[sim].profile.S.scaled[:, time[1]]) for (sim, time) in zip(simulation_indices_split, split_timeframes)]

truths_split = [(; T=train_data.data[sim].profile.T.scaled[:, time], 
             S=train_data.data[sim].profile.S.scaled[:, time], 
             ρ=train_data.data[sim].profile.ρ.scaled[:, time], 
             ∂T∂z=train_data.data[sim].profile.∂T∂z.scaled[:, time], 
             ∂S∂z=train_data.data[sim].profile.∂S∂z.scaled[:, time], 
             ∂ρ∂z=train_data.data[sim].profile.∂ρ∂z.scaled[:, time]) for (sim, time) in zip(simulation_indices_split, split_timeframes)]

truths = [(; T=data.profile.T.scaled, S=data.profile.S.scaled, ρ=data.profile.ρ.scaled, ∂T∂z=data.profile.∂T∂z.scaled, ∂S∂z=data.profile.∂S∂z.scaled, ∂ρ∂z=data.profile.∂ρ∂z.scaled) for data in train_data.data]

train_data_plot = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, full_timeframes)

params = [(                   f = data.coriolis.unscaled,
                     f_scaled = data.coriolis.scaled,
                            τ = data.times[end] - data.times[1],
                        N_timesteps = length(data.times),
                    scaled_time = (data.times .- data.times[1]) ./ (data.times[end] - data.times[1]),
                    scaled_Δt = (data.times[2] - data.times[1]) / (data.times[end] - data.times[1]),
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

params_split = [params[sim] for sim in simulation_indices_split]

rng = Random.default_rng(123)

#%%
NN_layers = vcat(Dense(100, hidden_layer_size, activation), [Dense(hidden_layer_size, hidden_layer_size, activation) for _ in 1:N_hidden_layer-1]..., Dense(hidden_layer_size, 31))

wT_NN = Chain(NN_layers...)
wS_NN = Chain(NN_layers...)

ps_wT, st_wT = Lux.setup(rng, wT_NN)
ps_wS, st_wS = Lux.setup(rng, wS_NN)

ps_wT = ps_wT |> ComponentArray .|> Float64
ps_wS = ps_wS |> ComponentArray .|> Float64

ps_wT .= glorot_uniform(rng, Float64, length(ps_wT))
ps_wS .= glorot_uniform(rng, Float64, length(ps_wS))

ps_wT .*= 1e-5
ps_wS .*= 1e-5

ps = ComponentArray(; wT=ps_wT, wS=ps_wS)
NNs = (wT=wT_NN, wS=wS_NN)
sts = (wT=st_wT, wS=st_wS)

function predict_residual_flux(T_hat, S_hat, ∂ρ∂z_hat, p, params, sts, NNs)
    x′ = vcat(T_hat, S_hat, ∂ρ∂z_hat, params.wT.scaled.top, params.wS.scaled.top, params.f_scaled)
    
    wT = vcat(0, NNs.wT(x′, p.wT, sts.wT)[1], 0)
    wS = vcat(0, NNs.wS(x′, p.wS, sts.wS)[1], 0)

    return wT, wS
end

function predict_residual_flux!(wT, wS, x′, p, sts, NNs)
    wT .= vcat(0, NNs.wT(x′, p.wT, sts.wT)[1], 0)
    wS .= vcat(0, NNs.wS(x′, p.wS, sts.wS)[1], 0)

    return nothing
end

function predict_boundary_flux(params)
    wT = vcat(fill(params.wT.scaled.bottom, params.coarse_size), params.wT.scaled.top)
    wS = vcat(fill(params.wS.scaled.bottom, params.coarse_size), params.wS.scaled.top)

    return wT, wS
end

function predict_boundary_flux!(wT, wS, params)
    wT[1:end-1] .= params.wT.scaled.bottom
    wS[1:end-1] .= params.wS.scaled.bottom

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

function solve_NDE(ps, params, x₀, ps_baseclosure, sts, NNs, timestep, Nt, timestep_multiple=10)
    eos = TEOS10EquationOfState()
    coarse_size = params.coarse_size
    Δt = timestep / timestep_multiple
    Nt_solve = (Nt - 1) * timestep_multiple + 1
    Dᶜ_hat = params.Dᶜ_hat
    Dᶠ_hat = params.Dᶠ_hat
    Dᶠ = params.Dᶠ

    scaling = params.scaling
    τ, H = params.τ, params.H

    T_hat = deepcopy(x₀.T)
    S_hat = deepcopy(x₀.S)
    ρ_hat = zeros(coarse_size)
    ∂ρ∂z_hat = zeros(coarse_size+1)

    T = zeros(coarse_size)
    S = zeros(coarse_size)
    ρ = zeros(coarse_size)
    
    T_RHS = zeros(coarse_size)
    S_RHS = zeros(coarse_size)

    wT_residual = zeros(coarse_size+1)
    wS_residual = zeros(coarse_size+1)

    wT_boundary = zeros(coarse_size+1)
    wS_boundary = zeros(coarse_size+1)

    νs = zeros(coarse_size+1)
    κs = zeros(coarse_size+1)

    Ris = zeros(coarse_size+1)
    
    sol_T = zeros(coarse_size, Nt_solve)
    sol_S = zeros(coarse_size, Nt_solve)
    sol_ρ = zeros(coarse_size, Nt_solve)

    sol_T[:, 1] .= T_hat
    sol_S[:, 1] .= S_hat

    LHS = zeros(coarse_size, coarse_size)

    x′ = zeros(3*coarse_size+1 + 3)

    for i in 2:Nt_solve
        T .= inv(scaling.T).(T_hat)
        S .= inv(scaling.S).(S_hat)

        ρ .= TEOS10.ρ.(T, S, 0, Ref(eos))
        ρ_hat .= scaling.ρ.(ρ)
        sol_ρ[:, i-1] .= ρ_hat

        ∂ρ∂z_hat .= scaling.∂ρ∂z.(Dᶠ * ρ)

        x′[1:coarse_size] .= T_hat
        x′[coarse_size+1:2*coarse_size] .= S_hat
        x′[2*coarse_size+1:3*coarse_size+1] .= ∂ρ∂z_hat
        x′[3*coarse_size+2] = params.wT.scaled.top
        x′[3*coarse_size+3] = params.wS.scaled.top
        x′[3*coarse_size+4] = params.f_scaled

        Ris .= calculate_Ri(zeros(coarse_size), zeros(coarse_size), ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf))
        predict_diffusivities!(νs, κs, Ris, ps_baseclosure)

        LHS .= Dᶜ_hat * (-κs .* Dᶠ_hat)
        LHS .*= -τ / H^2

        predict_residual_flux!(wT_residual, wS_residual, x′, ps, sts, NNs)
        predict_boundary_flux!(wT_boundary, wS_boundary, params)

        T_RHS .= - τ / H * scaling.wT.σ / scaling.T.σ .* (Dᶜ_hat * (wT_boundary .+ wT_residual))
        S_RHS .= - τ / H * scaling.wS.σ / scaling.S.σ .* (Dᶜ_hat * (wS_boundary .+ wS_residual))

        T_hat .= (I - Δt .* LHS) \ (T_hat .+ Δt .* T_RHS)
        S_hat .= (I - Δt .* LHS) \ (S_hat .+ Δt .* S_RHS)

        sol_T[:, i] .= T_hat
        sol_S[:, i] .= S_hat
    end

    sol_ρ[:, end] .= scaling.ρ.(TEOS10.ρ.(inv(scaling.T).(T_hat), inv(scaling.S).(S_hat), 0, Ref(eos)))

    return (; T=sol_T[:, 1:timestep_multiple:end], S=sol_S[:, 1:timestep_multiple:end], ρ=sol_ρ[:, 1:timestep_multiple:end])
end

# sol_T, sol_S, sol_ρ = solve_NDE(ps, params[1], x₀s[1], ps_baseclosure, sts, NNs, params[1].scaled_time[2] - params[1].scaled_time[1], length(25:10:45))

sol_T, sol_S, sol_ρ = solve_NDE(ps, params[simulation_indices_split[20]], x₀s_split[20], ps_baseclosure, sts, NNs, params[simulation_indices_split[20]].scaled_Δt, split_length+1)

#%%
truth = truths_split[20]
fig = Figure(size=(900, 600))
axT = CairoMakie.Axis(fig[1, 1], xlabel="T", ylabel="z")
axS = CairoMakie.Axis(fig[1, 2], xlabel="S", ylabel="z")
axρ = CairoMakie.Axis(fig[1, 3], xlabel="ρ", ylabel="z")

lines!(axT, sol_T[:, 1], params[1].zC, label="initial")
lines!(axT, sol_T[:, end], params[1].zC, label="final")
lines!(axT, truth.T[:, end], train_data.data[1].metadata["zC"], label="truth")

lines!(axS, sol_S[:, 1], params[1].zC, label="initial")
lines!(axS, sol_S[:, end], params[1].zC, label="final")
lines!(axS, truth.S[:, end], train_data.data[1].metadata["zC"], label="truth")

lines!(axρ, sol_ρ[:, 1], params[1].zC, label="initial")
lines!(axρ, sol_ρ[:, end], params[1].zC, label="final")
lines!(axρ, truth.ρ[:, end], train_data.data[1].metadata["zC"], label="truth")

axislegend(axT, orientation=:vertical, position=:rb)
display(fig)
#%%
function individual_loss(ps, truth, params, x₀, ps_baseclosure, sts, NNs, timestep, Nt, tstart=1, timestep_multiple=10)
    Dᶠ = params.Dᶠ
    scaling = params.scaling
    sol_T, sol_S, sol_ρ = solve_NDE(ps, params, x₀, ps_baseclosure, sts, NNs, timestep, Nt, timestep_multiple)

    T_loss = mean((sol_T .- truth.T[:, tstart:tstart+Nt-1]).^2)
    S_loss = mean((sol_S .- truth.S[:, tstart:tstart+Nt-1]).^2)
    ρ_loss = mean((sol_ρ .- truth.ρ[:, tstart:tstart+Nt-1]).^2)

    T = inv(scaling.T).(sol_T)
    S = inv(scaling.S).(sol_S)
    ρ = inv(scaling.ρ).(sol_ρ)

    ∂T∂z = scaling.∂T∂z.(Dᶠ * T)
    ∂S∂z = scaling.∂S∂z.(Dᶠ * S)
    ∂ρ∂z = scaling.∂ρ∂z.(Dᶠ * ρ)

    ∂T∂z_loss = mean((∂T∂z[1:end-2,:] .- truth.∂T∂z[1:end-2, tstart:tstart+Nt-1]).^2)
    ∂S∂z_loss = mean((∂S∂z[1:end-2,:] .- truth.∂S∂z[1:end-2, tstart:tstart+Nt-1]).^2)
    ∂ρ∂z_loss = mean((∂ρ∂z[1:end-2,:] .- truth.∂ρ∂z[1:end-2, tstart:tstart+Nt-1]).^2)

    return (; T=T_loss, S=S_loss, ρ=ρ_loss, ∂T∂z=∂T∂z_loss, ∂S∂z=∂S∂z_loss, ∂ρ∂z=∂ρ∂z_loss)
end

function loss(ps, truth, params, x₀, ps_baseclosure, sts, NNs, timestep, Nt, tstart=1, timestep_multiple=10, losses_prefactor=(; T=1, S=1, ρ=1, ∂T∂z=1, ∂S∂z=1, ∂ρ∂z=1))
    losses = individual_loss(ps, truth, params, x₀, ps_baseclosure, sts, NNs, timestep, Nt, tstart, timestep_multiple)
    return sum(values(losses) .* values(losses_prefactor))
end

# loss(ps, truths[1], params[1], x₀s[1], ps_baseclosure, sts, NNs, params[1].scaled_time[2] - params[1].scaled_time[1], length(25:10:45))
loss(ps, truths_split[20], params[simulation_indices_split[20]], x₀s_split[20], ps_baseclosure, sts, NNs, params[simulation_indices_split[20]].scaled_Δt, length(split_timeframes[20]))

# dps = deepcopy(ps) .= 0
# autodiff(Enzyme.ReverseWithPrimal, 
#          loss, 
#          Active, 
#          DuplicatedNoNeed(ps, dps), 
#          Const(truths[1]), 
#          Const(params[1]), 
#          DuplicatedNoNeed(x₀s[1], deepcopy(x₀s[1])), 
#          Const(ps_baseclosure), 
#          Const(sts), 
#          Const(NNs),
#          Const(params[1].scaled_time[2] - params[1].scaled_time[1]),
#          Const(length(25:10:45)))

# dps = deepcopy(ps) .= 0
# autodiff(Enzyme.ReverseWithPrimal, 
#          loss, 
#          Active, 
#          DuplicatedNoNeed(ps, dps), 
#          Const(truths_split[20]), 
#          Const(params[simulation_indices_split[20]]), 
#          DuplicatedNoNeed(x₀s_split[20], deepcopy(x₀s_split[20])), 
#          Const(ps_baseclosure), 
#          Const(sts), 
#          Const(NNs),
#          Const(params[simulation_indices_split[20]].scaled_Δt),
#          Const(length(split_timeframes[20])))

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
    T_loss, S_loss, ρ_loss, ∂T∂z_loss, ∂S∂z_loss, ∂ρ∂z_loss = values(individual_loss)
    
    total_contribution = contribution.T + contribution.S
    T_prefactor = total_contribution / contribution.T
    S_prefactor = total_contribution / contribution.S

    TS_loss = T_prefactor * T_loss + S_prefactor * S_loss

    ρ_prefactor = TS_loss / ρ_loss * 0.1 / 0.4
    ∂T∂z_prefactor = T_prefactor
    ∂S∂z_prefactor = S_prefactor

    ∂TS∂z_loss = ∂T∂z_loss + ∂S∂z_loss
    ∂ρ∂z_prefactor = ∂TS∂z_loss / ∂ρ∂z_loss * 0.1 / 0.4

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

# function compute_loss_prefactor(individual_loss)
#     T_loss, S_loss, ρ_loss, ∂T∂z_loss, ∂S∂z_loss, ∂ρ∂z_loss = values(individual_loss)

#     T_prefactor = 1
#     S_prefactor = T_loss / S_loss
#     ρ_prefactor = T_loss / ρ_loss

#     ∂T∂z_prefactor = 1
#     ∂S∂z_prefactor = ∂T∂z_loss / ∂S∂z_loss
#     ∂ρ∂z_prefactor = ∂T∂z_loss / ∂ρ∂z_loss

#     profile_loss = T_prefactor * T_loss + S_prefactor * S_loss + ρ_prefactor * ρ_loss
#     gradient_loss = ∂T∂z_prefactor * ∂T∂z_loss + ∂S∂z_prefactor * ∂S∂z_loss + ∂ρ∂z_prefactor * ∂ρ∂z_loss

#     gradient_prefactor = profile_loss / gradient_loss

#     ∂ρ∂z_prefactor *= gradient_prefactor
#     ∂T∂z_prefactor *= gradient_prefactor
#     ∂S∂z_prefactor *= gradient_prefactor

#     S_prefactor *= S_scaling
#     ∂S∂z_prefactor *= S_scaling

#     return (T=T_prefactor, S=S_prefactor, ρ=ρ_prefactor, ∂T∂z=∂T∂z_prefactor, ∂S∂z=∂S∂z_prefactor, ∂ρ∂z=∂ρ∂z_prefactor)
# end

# ind_losses = [individual_loss(ps, truth, params[sim], x₀, ps_baseclosure, sts, NNs, params[sim].scaled_Δt, length(time)) for (sim, time, truth, x₀) in zip(simulation_indices_split, split_timeframes, truths_split, x₀s_split)]

# ind_loss = (; T=sum([loss.T for loss in ind_losses]), 
#               S=sum([loss.S for loss in ind_losses]), 
#               ρ=sum([loss.ρ for loss in ind_losses]), 
#               ∂T∂z=sum([loss.∂T∂z for loss in ind_losses]), 
#               ∂S∂z=sum([loss.∂S∂z for loss in ind_losses]), 
#               ∂ρ∂z=sum([loss.∂ρ∂z for loss in ind_losses]))

# loss_prefactor = compute_loss_prefactor(ind_loss)

# density_contributions = [compute_density_contribution(data) for data in train_data.data]
# density_contributions = [compute_density_contribution(train_data.data[sim]) for sim in simulation_indices_split]

# loss_prefactors = compute_loss_prefactor_density_contribution.(ind_losses, density_contributions, S_scaling)

# loss_prefactors = [compute_loss_prefactor_density_contribution(ind_loss, density_contributions[sim], S_scaling) for (sim, ind_loss) in zip(simulation_indices_split, ind_losses)]

function loss_multipleics(ps, truths, params, x₀s, ps_baseclosure, sts, NNs, losses_prefactor, timestep, Nt, tstart=1, timestep_multiple=10)
    losses = [loss(ps, truth, param, x₀, ps_baseclosure, sts, NNs, timestep, Nt, tstart, timestep_multiple, loss_prefactor) for (truth, x₀, param, loss_prefactor) in zip(truths, x₀s, params, losses_prefactor)]
    return mean(losses)
end

# loss_multipleics(ps, [truths[1]], [params[1]], [x₀s[1]], ps_baseclosure, sts, NNs, [loss_prefactors[1]], params[1].scaled_time[2] - params[1].scaled_time[1], length(25:10:45))

loss_multipleics(ps, [truths_split[20]], [params[simulation_indices_split[20]]], [x₀s_split[20]], ps_baseclosure, sts, NNs, [loss_prefactors[20]], params[simulation_indices_split[20]].scaled_time[1], length(split_timeframes[20]))

# dps = deepcopy(ps) .= 0
# autodiff(Enzyme.ReverseWithPrimal, 
#          loss_multipleics, 
#          Active, 
#          DuplicatedNoNeed(ps, dps), 
#          DuplicatedNoNeed([truths[1]], deepcopy([truths[1]])), 
#          DuplicatedNoNeed([params[1]], deepcopy([params[1]])), 
#          DuplicatedNoNeed([x₀s[1]], deepcopy([x₀s[1]])), 
#          DuplicatedNoNeed(ps_baseclosure, deepcopy(ps_baseclosure)), 
#          Const(sts), 
#          Const(NNs), 
#          DuplicatedNoNeed([loss_prefactors[1]], deepcopy([loss_prefactors[1]])),
#          Const(params[1].scaled_time[2] - params[1].scaled_time[1]),
#          Const(length(25:10:45)))

# dps = deepcopy(ps) .= 0
# autodiff(Enzyme.ReverseWithPrimal, 
#          loss_multipleics, 
#          Active, 
#          DuplicatedNoNeed(ps, dps), 
#          DuplicatedNoNeed([truths_split[20]], deepcopy([truths_split[20]])), 
#          DuplicatedNoNeed([params[simulation_indices_split[20]]], deepcopy([params[simulation_indices_split[20]]])), 
#          DuplicatedNoNeed([x₀s_split[20]], deepcopy([x₀s_split[20]])), 
#          DuplicatedNoNeed(ps_baseclosure, deepcopy(ps_baseclosure)), 
#          Const(sts), 
#          Const(NNs), 
#          DuplicatedNoNeed([loss_prefactors[20]], deepcopy([loss_prefactors[20]])),
#          Const(params[simulation_indices_split[20]].scaled_Δt),
#          Const(length(split_timeframes[20])))

# autodiff(Enzyme.ReverseWithPrimal, 
#          loss_multipleics, 
#          Active, 
#          DuplicatedNoNeed(ps, dps), 
#          DuplicatedNoNeed(truths_split, deepcopy(truths_split)),
#          DuplicatedNoNeed(params_split, deepcopy(params_split)),
#          DuplicatedNoNeed(x₀s_split, deepcopy(x₀s_split)),
#          DuplicatedNoNeed(ps_baseclosure, deepcopy(ps_baseclosure)),
#          Const(sts),
#          Const(NNs),
#          DuplicatedNoNeed(loss_prefactors, deepcopy(loss_prefactors)),
#          Const(params[simulation_indices_split[20]].scaled_Δt),
#          Const(length(split_timeframes[20])))

function predict_residual_flux_dimensional(T_hat, S_hat, ∂ρ∂z_hat, p, params, sts, NNs)
    wT_hat, wS_hat = predict_residual_flux(T_hat, S_hat, ∂ρ∂z_hat, p, params, sts, NNs)
    
    wT = inv(params.scaling.wT).(wT_hat)
    wS = inv(params.scaling.wS).(wS_hat)

    wT = wT .- wT[1]
    wS = wS .- wS[1]

    return wT, wS
end

function predict_diffusive_flux(Ris, T_hat, S_hat, ps_baseclosure, params)
    _, κs = predict_diffusivities(Ris, ps_baseclosure)

    ∂T∂z_hat = params.Dᶠ_hat * T_hat
    ∂S∂z_hat = params.Dᶠ_hat * S_hat

    wT_diffusive = -κs .* ∂T∂z_hat
    wS_diffusive = -κs .* ∂S∂z_hat
    return wT_diffusive, wS_diffusive
end

function predict_diffusive_boundary_flux_dimensional(Ris, T_hat, S_hat, ps_baseclosure, params)
    _wT_diffusive, _wS_diffusive = predict_diffusive_flux(Ris, T_hat, S_hat, ps_baseclosure, params)
    _wT_boundary, _wS_boundary = predict_boundary_flux(params)

    wT_diffusive = params.scaling.T.σ / params.H .* _wT_diffusive
    wS_diffusive = params.scaling.S.σ / params.H .* _wS_diffusive

    wT_boundary = inv(params.scaling.wT).(_wT_boundary)
    wS_boundary = inv(params.scaling.wS).(_wS_boundary)

    wT = wT_diffusive .+ wT_boundary
    wS = wS_diffusive .+ wS_boundary

    return wT, wS
end

function solve_NDE_postprocessing(ps, params, x₀, ps_baseclosure, sts, NNs, timestep, Nt, timestep_multiple=2)
    eos = TEOS10EquationOfState()
    coarse_size = params.coarse_size
    Δt = timestep / timestep_multiple
    Nt_solve = (Nt - 1) * timestep_multiple + 1
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

function diagnose_fields(ps, params, x₀, ps_baseclosure, sts, NNs, train_data_plot, timestep, Nt, timestep_multiple=2)
    sols = solve_NDE_postprocessing(ps, params, x₀, ps_baseclosure, sts, NNs, timestep, Nt, timestep_multiple)

    ps_noNN = deepcopy(ps) .= 0
    sols_noNN = solve_NDE_postprocessing(ps_noNN, params, x₀, ps_baseclosure, sts, NNs, timestep, Nt, timestep_multiple)

    coarse_size = params.coarse_size
    Dᶠ = params.Dᶠ
    scaling = params.scaling

    Ts = inv(scaling.T).(sols.T)
    Ss = inv(scaling.S).(sols.S)
    ρs = inv(scaling.ρ).(sols.ρ)

    Ts_noNN = inv(scaling.T).(sols_noNN.T)
    Ss_noNN = inv(scaling.S).(sols_noNN.S)
    ρs_noNN = inv(scaling.ρ).(sols_noNN.ρ)
    
    ∂ρ∂z_hats = hcat([params.scaling.∂ρ∂z.(params.Dᶠ * ρ) for ρ in eachcol(ρs)]...)

    eos = TEOS10EquationOfState()
    Ris_truth = hcat([calculate_Ri(zeros(coarse_size), zeros(coarse_size), ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf)) for ρ in eachcol(train_data_plot.profile.ρ.unscaled)]...)
    Ris = hcat([calculate_Ri(zeros(coarse_size), zeros(coarse_size), ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf)) for ρ in eachcol(ρs)]...)
    Ris_noNN = hcat([calculate_Ri(zeros(coarse_size), zeros(coarse_size), ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf)) for ρ in eachcol(ρs_noNN)]...)
    
    νs, κs = predict_diffusivities(Ris, ps_baseclosure)
    νs_noNN, κs_noNN = predict_diffusivities(Ris_noNN, ps_baseclosure)

    wT_residuals = zeros(coarse_size+1, size(Ts, 2))
    wS_residuals = zeros(coarse_size+1, size(Ts, 2))

    wT_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))
    wS_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))

    wT_diffusive_boundarys_noNN = zeros(coarse_size+1, size(Ts, 2))
    wS_diffusive_boundarys_noNN = zeros(coarse_size+1, size(Ts, 2))

    for i in 1:size(wT_residuals, 2)
        wT_residuals[:, i], wS_residuals[:, i] = predict_residual_flux_dimensional(sols.T[:, i], sols.S[:, i], ∂ρ∂z_hats[:, i], ps, params, sts, NNs)
        wT_diffusive_boundarys[:, i], wS_diffusive_boundarys[:, i] = predict_diffusive_boundary_flux_dimensional(Ris[:, i], sols.T[:, i], sols.S[:, i], ps_baseclosure, params)        

        wT_diffusive_boundarys_noNN[:, i], wS_diffusive_boundarys_noNN[:, i] = predict_diffusive_boundary_flux_dimensional(Ris_truth[:, i], sols_noNN.T[:, i], sols_noNN.S[:, i], ps_baseclosure, params)
    end

    wT_totals = wT_residuals .+ wT_diffusive_boundarys
    wS_totals = wS_residuals .+ wS_diffusive_boundarys

    fluxes = (; wT = (; diffusive_boundary=wT_diffusive_boundarys, residual=wT_residuals, total=wT_totals), 
                wS = (; diffusive_boundary=wS_diffusive_boundarys, residual=wS_residuals, total=wS_totals))

    fluxes_noNN = (; wT = (; total=wT_diffusive_boundarys_noNN), 
                     wS = (; total=wS_diffusive_boundarys_noNN))

    diffusivities = (; ν=νs, κ=κs, Ri=Ris, Ri_truth=Ris_truth)

    diffusivities_noNN = (; ν=νs_noNN, κ=κs_noNN, Ri=Ris_noNN)

    sols_dimensional = (; T=Ts, S=Ss, ρ=ρs)
    sols_dimensional_noNN = (; T=Ts_noNN, S=Ss_noNN, ρ=ρs_noNN)
    return (; sols_dimensional, sols_dimensional_noNN, fluxes, fluxes_noNN, diffusivities, diffusivities_noNN)
end

function animate_data(train_data, sols, fluxes, diffusivities, sols_noNN, fluxes_noNN, diffusivities_noNN, index, FILE_DIR, Nframes; suffix=1)
    fig = Figure(size=(1920, 1080))
    axT = CairoMakie.Axis(fig[1, 1], title="T", xlabel="T (°C)", ylabel="z (m)")
    axS = CairoMakie.Axis(fig[1, 2], title="S", xlabel="S (g kg⁻¹)", ylabel="z (m)")
    axρ = CairoMakie.Axis(fig[1, 3], title="ρ", xlabel="ρ (kg m⁻³)", ylabel="z (m)")
    axwT = CairoMakie.Axis(fig[2, 1], title="wT", xlabel="wT (m s⁻¹ °C)", ylabel="z (m)")
    axwS = CairoMakie.Axis(fig[2, 2], title="wS", xlabel="wS (m s⁻¹ g kg⁻¹)", ylabel="z (m)")
    axRi = CairoMakie.Axis(fig[1, 4], title="Ri", xlabel="Ri", ylabel="z (m)")
    axdiffusivity = CairoMakie.Axis(fig[2, 3], title="Diffusivity", xlabel="Diffusivity (m² s⁻¹)", ylabel="z (m)")

    n = Observable(1)
    zC = train_data.metadata["zC"]
    zF = train_data.metadata["zF"]

    Ri_max = 3

    T_NDE = sols.T
    S_NDE = sols.S
    ρ_NDE = sols.ρ

    wT_residual = fluxes.wT.residual
    wS_residual = fluxes.wS.residual

    wT_diffusive_boundary = fluxes.wT.diffusive_boundary
    wS_diffusive_boundary = fluxes.wS.diffusive_boundary

    wT_total = fluxes.wT.total
    wS_total = fluxes.wS.total

    T_noNN = sols_noNN.T
    S_noNN = sols_noNN.S
    ρ_noNN = sols_noNN.ρ

    wT_noNN = fluxes_noNN.wT.total
    wS_noNN = fluxes_noNN.wS.total

    Tlim = (find_min(T_NDE, train_data.profile.T.unscaled, T_noNN), find_max(T_NDE, train_data.profile.T.unscaled, T_noNN))
    Slim = (find_min(S_NDE, train_data.profile.S.unscaled, S_noNN), find_max(S_NDE, train_data.profile.S.unscaled, S_noNN))
    ρlim = (find_min(ρ_NDE, train_data.profile.ρ.unscaled, ρ_noNN), find_max(ρ_NDE, train_data.profile.ρ.unscaled, ρ_noNN))

    wTlim = (find_min(wT_residual, train_data.flux.wT.column.unscaled),
             find_max(wT_residual, train_data.flux.wT.column.unscaled))
    wSlim = (find_min(wS_residual, train_data.flux.wS.column.unscaled),
             find_max(wS_residual, train_data.flux.wS.column.unscaled))

    wTlim = (find_min(wT_residual, train_data.flux.wT.column.unscaled),
             find_max(wT_residual, train_data.flux.wT.column.unscaled))
    wSlim = (find_min(wS_residual, train_data.flux.wS.column.unscaled),
             find_max(wS_residual, train_data.flux.wS.column.unscaled))

    Rilim = (find_min(diffusivities.Ri, diffusivities.Ri_truth, diffusivities_noNN.Ri,), Ri_max)

    diffusivitylim = (find_min(diffusivities.ν, diffusivities.κ, diffusivities_noNN.ν, diffusivities_noNN.κ), 
                      find_max(diffusivities.ν, diffusivities.κ, diffusivities_noNN.ν, diffusivities_noNN.κ),)

    T_truthₙ = @lift train_data.profile.T.unscaled[:, $n]
    S_truthₙ = @lift train_data.profile.S.unscaled[:, $n]
    ρ_truthₙ = @lift train_data.profile.ρ.unscaled[:, $n]

    wT_truthₙ = @lift train_data.flux.wT.column.unscaled[:, $n]
    wS_truthₙ = @lift train_data.flux.wS.column.unscaled[:, $n]

    T_NDEₙ = @lift T_NDE[:, $n]
    S_NDEₙ = @lift S_NDE[:, $n]
    ρ_NDEₙ = @lift ρ_NDE[:, $n]

    T_noNNₙ = @lift T_noNN[:, $n]
    S_noNNₙ = @lift S_noNN[:, $n]
    ρ_noNNₙ = @lift ρ_noNN[:, $n]

    wT_residualₙ = @lift wT_residual[:, $n]
    wS_residualₙ = @lift wS_residual[:, $n]

    wT_diffusive_boundaryₙ = @lift wT_diffusive_boundary[:, $n]
    wS_diffusive_boundaryₙ = @lift wS_diffusive_boundary[:, $n]

    wT_totalₙ = @lift wT_total[:, $n]
    wS_totalₙ = @lift wS_total[:, $n]

    wT_noNNₙ = @lift wT_noNN[:, $n]
    wS_noNNₙ = @lift wS_noNN[:, $n]

    Ri_truthₙ = @lift clamp.(diffusivities.Ri_truth[:, $n], Ref(-Inf..Ri_max))
    Riₙ = @lift clamp.(diffusivities.Ri[:, $n], Ref(-Inf..Ri_max))
    Ri_noNNₙ = @lift clamp.(diffusivities_noNN.Ri[:, $n], Ref(-Inf..Ri_max))

    νₙ = @lift diffusivities.ν[:, $n]
    κₙ = @lift diffusivities.κ[:, $n]

    ν_noNNₙ = @lift diffusivities_noNN.ν[:, $n]
    κ_noNNₙ = @lift diffusivities_noNN.κ[:, $n]

    Qᵀ = train_data.metadata["temperature_flux"]
    Qˢ = train_data.metadata["salinity_flux"]
    f = train_data.metadata["coriolis_parameter"]
    times = train_data.times
    Nt = length(times)

    time_str = @lift "Qᵀ = $(Qᵀ) m s⁻¹ °C, Qˢ = $(Qˢ) m s⁻¹ g kg⁻¹, f = $(f) s⁻¹, Time = $(round(times[$n]/24/60^2, digits=3)) days"

    lines!(axT, T_truthₙ, zC, label="Truth", linewidth=4, alpha=0.5)
    lines!(axT, T_noNNₙ, zC, label="Base closure only")
    lines!(axT, T_NDEₙ, zC, label="NDE", color=:black)

    lines!(axS, S_truthₙ, zC, label="Truth", linewidth=4, alpha=0.5)
    lines!(axS, S_noNNₙ, zC, label="Base closure only")
    lines!(axS, S_NDEₙ, zC, label="NDE", color=:black)

    lines!(axρ, ρ_truthₙ, zC, label="Truth", linewidth=4, alpha=0.5)
    lines!(axρ, ρ_noNNₙ, zC, label="Base closure only")
    lines!(axρ, ρ_NDEₙ, zC, label="NDE", color=:black)

    lines!(axwT, wT_truthₙ, zF, label="Truth", linewidth=4, alpha=0.5)
    lines!(axwT, wT_noNNₙ, zF, label="Base closure only")
    lines!(axwT, wT_diffusive_boundaryₙ, zF, label="Base closure")
    lines!(axwT, wT_totalₙ, zF, label="NDE")
    lines!(axwT, wT_residualₙ, zF, label="Residual", color=:black)

    lines!(axwS, wS_truthₙ, zF, label="Truth", linewidth=4, alpha=0.5)
    lines!(axwS, wS_noNNₙ, zF, label="Base closure only")
    lines!(axwS, wS_diffusive_boundaryₙ, zF, label="Base closure")
    lines!(axwS, wS_totalₙ, zF, label="NDE")
    lines!(axwS, wS_residualₙ, zF, label="Residual", color=:black)

    lines!(axRi, Ri_truthₙ, zF, label="Truth", linewidth=4, alpha=0.5)
    lines!(axRi, Ri_noNNₙ, zF, label="Base closure only")
    lines!(axRi, Riₙ, zF, label="NDE", color=:black)

    lines!(axdiffusivity, ν_noNNₙ, zF, label="ν, Base closure only")
    lines!(axdiffusivity, κ_noNNₙ, zF, label="κ, Base closure only")
    lines!(axdiffusivity, νₙ, zF, label="ν, NDE")
    lines!(axdiffusivity, κₙ, zF, label="κ, NDE")

    axislegend(axT, position=:lb)
    axislegend(axwT, position=:rb)
    axislegend(axdiffusivity, position=:rb)
    
    Label(fig[0, :], time_str, font=:bold, tellwidth=false)

    xlims!(axT, Tlim)
    xlims!(axS, Slim)
    xlims!(axρ, ρlim)
    xlims!(axwT, wTlim)
    xlims!(axwS, wSlim)
    xlims!(axRi, Rilim)
    xlims!(axdiffusivity, diffusivitylim)

    CairoMakie.record(fig, "$(FILE_DIR)/training_$(index)_$(suffix).mp4", 1:Nframes, framerate=10) do nn
        n[] = nn
    end
end

function plot_loss(losses, FILE_DIR; suffix=1)
    colors = distinguishable_colors(10, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    
    fig = Figure(size=(1000, 600))
    axtotalloss = CairoMakie.Axis(fig[1, 1], title="Total Loss", xlabel="Iterations", ylabel="Loss", yscale=log10)
    # axindividualloss = CairoMakie.Axis(fig[1, 2], title="Individual Loss", xlabel="Iterations", ylabel="Loss", yscale=log10)

    lines!(axtotalloss, losses.total, label="Total Loss", color=colors[1])

    # lines!(axindividualloss, losses.T, label="T", color=colors[3])
    # lines!(axindividualloss, losses.S, label="S", color=colors[4])
    # lines!(axindividualloss, losses.ρ, label="ρ", color=colors[5])
    # lines!(axindividualloss, losses.∂T∂z, label="∂T∂z", color=colors[8])
    # lines!(axindividualloss, losses.∂S∂z, label="∂S∂z", color=colors[9])
    # lines!(axindividualloss, losses.∂ρ∂z, label="∂ρ∂z", color=colors[10])

    axislegend(axtotalloss, position=:lb)
    # axislegend(axindividualloss, position=:rt)
    save("$(FILE_DIR)/losses_$(suffix).png", fig, px_per_unit=8)
end

# function train_NDE_stochastic(ps, params, ps_baseclosure, sts, NNs, truths, x₀s, losses_prefactors, rng, train_data_plot; epoch=1, maxiter=2, rule=Optimisers.Adam())
#     opt_state = Optimisers.setup(rule, ps)
#     opt_statemin = deepcopy(opt_state)
#     l_min = Inf
#     ps_min = deepcopy(ps)
#     dps = deepcopy(ps) .= 0
#     wall_clock = [time_ns()]
#     losses = zeros(maxiter)
#     mean_loss = mean(losses)
#     stochastic_batch = collect(1:length(truths))
#     ind_loss = zeros(length(truths))
#     for iter in 1:maxiter
#         shuffle!(rng, stochastic_batch)
#         for sim_index in stochastic_batch
#             truth = truths[sim_index]
#             x₀ = x₀s[sim_index]
#             param = params[sim_index]
#             loss_prefactor = losses_prefactors[sim_index]
#             _, l = autodiff(Enzyme.ReverseWithPrimal, 
#                             loss, 
#                             Active, 
#                             Duplicated(ps, dps), 
#                             Const(truth), 
#                             Const(param), 
#                             DuplicatedNoNeed(x₀, deepcopy(x₀)), 
#                             Const(ps_baseclosure), 
#                             Const(sts), 
#                             Const(NNs),
#                             Const(loss_prefactor))
#             ind_loss[sim_index] = l

#             opt_state, ps = Optimisers.update!(opt_state, ps, dps)

#             losses[iter] = l
#             dps .= 0
#         end
#         mean_loss = mean(ind_loss)

#         @printf("%s, Δt %s, iter %d/%d, loss average %6.10e, max NN weight %6.5e\n",
#                 Dates.now(), prettytime(1e-9 * (time_ns() - wall_clock[1])), iter, maxiter, mean_loss, 
#                 maximum(abs, ps))

#         if mean_loss < l_min
#             l_min = mean_loss
#             opt_statemin = deepcopy(opt_state)
#             ps_min .= ps
#         end

#         if iter % 200 == 0
#             @info "Saving intermediate results"
#             jldsave("$(FILE_DIR)/intermediate_training_results_round$(epoch)_epoch$(iter).jld2"; u=ps_min, state=opt_statemin, loss=l_min)
#             sols = [diagnose_fields(ps_min, param, x₀, ps_baseclosure, sts, NNs, data) for (data, x₀, param) in zip(train_data_plot.data, x₀s, params)]
#             for (index, sol) in enumerate(sols)
#                 animate_data(train_data_plot.data[index], sol.sols_dimensional, sol.fluxes, sol.diffusivities, sol.sols_dimensional_noNN, sol.fluxes_noNN, sol.diffusivities_noNN, index, FILE_DIR; epoch="intermediateround$(epoch)_$(iter)")
#             end
#         end
#         wall_clock = [time_ns()]
#     end
#     return ps_min, (; total=losses), opt_statemin
# end

# optimizers = [Optimisers.Adam(3e-4), Optimisers.Adam(1e-4), Optimisers.Adam(3e-5)]
# maxiters = [1000, 1000, 1000]
# end_epochs = cumsum(maxiters)
# # optimizers = [Optimisers.Adam(3e-4)]
# # maxiters = [3]
# # end_epochs = cumsum(maxiters)

# for (i, (epoch, optimizer, maxiter)) in enumerate(zip(end_epochs, optimizers, maxiters))
#     global ps = ps
#     ps, losses, opt_state = train_NDE_stochastic(ps, params, ps_baseclosure, sts, NNs, truths, x₀s, loss_prefactors, rng, train_data_plot; epoch=i, maxiter=maxiter, rule=optimizer)
    
#     jldsave("$(FILE_DIR)/training_results_epoch$(epoch).jld2"; u=ps, losses=losses, state=opt_state)
#     sols = [diagnose_fields(ps, param, x₀, ps_baseclosure, sts, NNs, data) for (data, x₀, param) in zip(train_data_plot.data, x₀s, params)]
#     for (index, sol) in enumerate(sols)
#         animate_data(train_data_plot.data[index], sol.sols_dimensional, sol.fluxes, sol.diffusivities, sol.sols_dimensional_noNN, sol.fluxes_noNN, sol.diffusivities_noNN, index, FILE_DIR; epoch=epoch)
#     end
#     plot_loss(losses, FILE_DIR; epoch=epoch)
# end

function train_NDE_multipleics(ps, params, ps_baseclosure, sts, NNs, truths, x₀s, train_data_plot, timeframes, S_scaling, density_contributions; sim_index=[1], epoch=1, maxiter=2, rule=Optimisers.Adam())
    opt_state = Optimisers.setup(rule, ps)
    opt_statemin = deepcopy(opt_state)
    l_min = Inf
    ps_min = deepcopy(ps)
    dps = deepcopy(ps) .= 0
    wall_clock = [time_ns()]
    losses = zeros(maxiter)
    mean_loss = mean(losses)
    ind_losses = [individual_loss(ps, truth, param, x₀, ps_baseclosure, sts, NNs, param.scaled_Δt, length(timeframe)) for (truth, x₀, param, timeframe) in zip(truths[sim_index], x₀s[sim_index], params[sim_index], timeframes)]
    loss_prefactors = compute_loss_prefactor_density_contribution.(ind_losses, density_contributions[sim_index], S_scaling)

    for iter in 1:maxiter
        _, l = autodiff(Enzyme.ReverseWithPrimal, 
                        loss_multipleics, 
                        Active, 
                        DuplicatedNoNeed(ps, dps), 
                        DuplicatedNoNeed(truths[sim_index], deepcopy(truths[sim_index])), 
                        DuplicatedNoNeed(params[sim_index], deepcopy(params[sim_index])), 
                        DuplicatedNoNeed(x₀s[sim_index], deepcopy(x₀s[sim_index])), 
                        DuplicatedNoNeed(ps_baseclosure, deepcopy(ps_baseclosure)), 
                        Const(sts), 
                        Const(NNs), 
                        DuplicatedNoNeed(loss_prefactors, deepcopy(loss_prefactors)),
                        Const(params[sim_index][1].scaled_Δt),
                        Const(length(timeframes[1])))
        if iter <= 40
            Optimisers.adjust!(opt_state, eta=rule.eta * iter / 40)
        end
        
        opt_state, ps = Optimisers.update!(opt_state, ps, dps)

        losses[iter] = l
        if iter == 1
            l_min = l
        end
        
        @printf("%s, Δt %s, round %d, iter %d/%d, loss average %6.10e, minimum loss %6.5e, max NN weight %6.5e, gradient norm %6.5e\n",
                Dates.now(), prettytime(1e-9 * (time_ns() - wall_clock[1])), epoch, iter, maxiter, l, l_min,
                maximum(abs, ps), maximum(abs, dps))
        
        dps .= 0
        mean_loss = l
        if mean_loss < l_min
            l_min = mean_loss
            opt_statemin = deepcopy(opt_state)
            ps_min .= ps
        end

        if iter % 10000 == 0
            @info "Saving intermediate results"
            jldsave("$(FILE_DIR)/intermediate_training_results_round$(epoch)_epoch$(iter)_end$(timeframes[end]).jld2"; u=ps_min, state=opt_statemin, loss=l_min)
            # sols = [diagnose_fields(ps_min, param, x₀, ps_baseclosure, sts, NNs, data) for (data, x₀, param) in zip(train_data_plot.data, x₀s, params)]
            # for (index, sol) in enumerate(sols)
            #     animate_data(train_data_plot.data[index], sol.sols_dimensional, sol.fluxes, sol.diffusivities, sol.sols_dimensional_noNN, sol.fluxes_noNN, sol.diffusivities_noNN, index, FILE_DIR; epoch="intermediateround$(epoch)_$(iter)")
            # end
        end
        wall_clock = [time_ns()]
    end
    return ps_min, (; total=losses), opt_statemin
end

# optimizers = [Optimisers.Adam(1e-4), Optimisers.Adam(3e-5), Optimisers.Adam(1e-5)]
# maxiters = [10000, 10000, 10000]
# end_epochs = cumsum(maxiters)
# optimizers = vcat([Optimisers.Adam(3e-5) for _ in 1:3], [Optimisers.Adam(1e-5) for _ in 1:3], [Optimisers.Adam(3e-6) for _ in 1:5], Optimisers.Adam(1e-6), Optimisers.Adam(3e-7))
# maxiters = vcat([5000 for _ in 1:3], [7000 for _ in 1:10])
# end_epochs = cumsum(maxiters)

# sim_indices = [1, 2, 3, 4, 5, 6, 7, 8]

# training_timeframes = [timeframes[1][1:2],
#                       timeframes[1][1:3],
#                       timeframes[1][1:5],
#                       timeframes[1][1:7],
#                       timeframes[1][1:10],
#                       timeframes[1][1:12],
#                       timeframes[1][1:15],
#                       timeframes[1][1:18],
#                       timeframes[1][1:21],
#                       timeframes[1][1:24],
#                       timeframes[1][1:27],
#                       timeframes[1][1:27],
#                       timeframes[1][1:27]]

# plot_timeframes = [training_timeframe[1]:training_timeframe[end] for training_timeframe in training_timeframes]
# sols = nothing
# for (i, (epoch, optimizer, maxiter, training_timeframe, plot_timeframe)) in enumerate(zip(end_epochs, optimizers, maxiters, training_timeframes, plot_timeframes))
#     global ps = ps
#     global sols = sols
#     ps, losses, opt_state = train_NDE_multipleics(ps, params, ps_baseclosure, sts, NNs, truths, x₀s, train_data_plot, training_timeframe, S_scaling; sim_index=sim_indices, epoch=i, maxiter=maxiter, rule=optimizer)
    
#     jldsave("$(FILE_DIR)/training_results_epoch$(epoch)_end$(training_timeframe[end]).jld2"; u=ps, losses=losses, state=opt_state)

#     sols = [diagnose_fields(ps, param, x₀, ps_baseclosure, sts, NNs, data, param.scaled_original_time[2] - param.scaled_original_time[1], length(plot_timeframe)) for (data, x₀, param) in zip(train_data_plot.data[sim_indices], x₀s[sim_indices], params[sim_indices])]

#     for (i, index) in enumerate(sim_indices)
#         sol = sols[i]
#         animate_data(train_data_plot.data[index], sol.sols_dimensional, sol.fluxes, sol.diffusivities, sol.sols_dimensional_noNN, sol.fluxes_noNN, sol.diffusivities_noNN, index, FILE_DIR, length(plot_timeframe); suffix="epoch$(epoch)_end$(training_timeframe[end])")
#     end

#     plot_loss(losses, FILE_DIR; suffix="epoch$(epoch)_end$(training_timeframe[end])")

# end

sim_indices = 1:length(split_timeframes)
optimizers = [Optimisers.Adam(3e-4)]
maxiters = [5000]
end_epochs = cumsum(maxiters)
plot_timeframes = full_timeframes
density_contributions = [compute_density_contribution(train_data.data[sim]) for sim in simulation_indices_split]

sols = nothing
for (i, (epoch, optimizer, maxiter)) in enumerate(zip(end_epochs, optimizers, maxiters))
    global ps = ps
    global sols = sols
    ps, losses, opt_state = train_NDE_multipleics(ps, params_split, ps_baseclosure, sts, NNs, truths_split, x₀s_split, train_data_plot, split_timeframes, S_scaling, density_contributions; sim_index=sim_indices, epoch=i, maxiter=maxiter, rule=optimizer)
    
    jldsave("$(FILE_DIR)/training_results_epoch$(epoch)_end$(plot_timeframes[1][end]).jld2"; u=ps, losses=losses, state=opt_state)

    sols = [diagnose_fields(ps, param, x₀, ps_baseclosure, sts, NNs, data, param.scaled_original_time[2] - param.scaled_original_time[1], length(plot_timeframe)) for (data, x₀, param, plot_timeframe) in zip(train_data_plot.data, x₀s, params, plot_timeframes)]

    for (i, sol) in enumerate(sols)
        animate_data(train_data_plot.data[i], sol.sols_dimensional, sol.fluxes, sol.diffusivities, sol.sols_dimensional_noNN, sol.fluxes_noNN, sol.diffusivities_noNN, i, FILE_DIR, length(plot_timeframes[i]); suffix="epoch$(epoch)_split$(split_length)")
    end

    plot_loss(losses, FILE_DIR; suffix="epoch$(epoch)_end$(plot_timeframes[1][end])")

end

optimizers = [Optimisers.Adam(1e-4)]
maxiters = [5000]
end_epochs = end_epochs[end] .+ cumsum(maxiters)

sim_indices = [1, 2, 3, 4, 5, 6, 7, 8]

training_timeframes = [[timeframes[1][1:27] for _ in 1:length(sim_indices)]]

plot_timeframes = full_timeframes
density_contributions = [compute_density_contribution(train_data.data[sim]) for sim in sim_indices]

sols = nothing
for (i, (epoch, optimizer, maxiter, training_timeframe, plot_timeframe)) in enumerate(zip(end_epochs, optimizers, maxiters, training_timeframes, plot_timeframes))
    global ps = ps
    global sols = sols

    ps, losses, opt_state = train_NDE_multipleics(ps, params, ps_baseclosure, sts, NNs, truths, x₀s, train_data_plot, training_timeframe, S_scaling, density_contributions; sim_index=sim_indices, epoch=i, maxiter=maxiter, rule=optimizer)
    jldsave("$(FILE_DIR)/training_results_epoch$(epoch)_end$(training_timeframe[1][end]).jld2"; u=ps, losses=losses, state=opt_state)

    sols = [diagnose_fields(ps, param, x₀, ps_baseclosure, sts, NNs, data, param.scaled_original_time[2] - param.scaled_original_time[1], length(plot_timeframe)) for (data, x₀, param) in zip(train_data_plot.data[sim_indices], x₀s[sim_indices], params[sim_indices])]

    for (i, index) in enumerate(sim_indices)
        sol = sols[i]
        animate_data(train_data_plot.data[index], sol.sols_dimensional, sol.fluxes, sol.diffusivities, sol.sols_dimensional_noNN, sol.fluxes_noNN, sol.diffusivities_noNN, index, FILE_DIR, length(plot_timeframe); suffix="epoch$(epoch)_end$(training_timeframe[1][end])")
    end

    plot_loss(losses, FILE_DIR; suffix="epoch$(epoch)_end$(training_timeframe[1][end])")

end