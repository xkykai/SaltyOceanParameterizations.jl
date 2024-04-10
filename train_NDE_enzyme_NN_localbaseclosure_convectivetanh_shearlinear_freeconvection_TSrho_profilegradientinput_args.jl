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
        default = 256
      "--hidden_layer"
        help = "Number of hidden layers"
        arg_type = Int64
        default = 2
      "--activation"
        help = "Activation function"
        arg_type = String
        default = "relu"
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

FILE_DIR = "./training_output/NDE_enzyme_$(args["hidden_layer"])layer_$(args["hidden_layer_size"])_$(args["activation"])_localbaseclosure_convectivetanh_shearlinear_TSrho_profilegradientinput_args"
mkpath(FILE_DIR)

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
# wT_NN = Chain(Dense(67, 4, leakyrelu), Dense(4, 31))
# wS_NN = Chain(Dense(67, 4, leakyrelu), Dense(4, 31))

NN_layers = vcat(Dense(67, hidden_layer_size, activation), [Dense(hidden_layer_size, hidden_layer_size, relu) for _ in 1:N_hidden_layer-1]..., Dense(hidden_layer_size, 31))

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

    ∂T∂z_loss = mean((∂T∂z .- truth.∂T∂z)[1:end-3, :].^2)
    ∂S∂z_loss = mean((∂S∂z .- truth.∂S∂z)[1:end-3, :].^2)
    ∂ρ∂z_loss = mean((∂ρ∂z .- truth.∂ρ∂z)[1:end-3, :].^2)

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
ind_loss = (; T=sum([loss.T for loss in ind_losses]), 
              S=sum([loss.S for loss in ind_losses]), 
              ρ=sum([loss.ρ for loss in ind_losses]), 
              ∂T∂z=sum([loss.∂T∂z for loss in ind_losses]), 
              ∂S∂z=sum([loss.∂S∂z for loss in ind_losses]), 
              ∂ρ∂z=sum([loss.∂ρ∂z for loss in ind_losses]))

loss_prefactor = compute_loss_prefactor(ind_loss)

function loss_multipleics(ps, truths, params, x₀s, ps_baseclosure, sts, NNs, losses_prefactor=(; T=1, S=1, ρ=1, ∂T∂z=1, ∂S∂z=1, ∂ρ∂z=1))
    losses = [loss(ps, truth, param, x₀, ps_baseclosure, sts, NNs, losses_prefactor) for (truth, x₀, param) in zip(truths, x₀s, params)]
    return mean(losses)
end

function train_NDE_multipleics(ps, params, ps_baseclosure, sts, NNs, truths, x₀s; epoch=1, maxiter=2, rule=Optimisers.Adam(), loss_prefactor=(; T=1, S=1, ρ=1, ∂T∂z=1, ∂S∂z=1, ∂ρ∂z=1))
    opt_state = Optimisers.setup(rule, ps)
    dps = deepcopy(ps) .= 0
    wall_clock = [time_ns()]
    losses = zeros(maxiter)
    for iter in 1:maxiter
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
        if iter % 200 == 0
            jldsave("$(FILE_DIR)/intermediate_training_results_epoch$(epoch)_iter$(iter).jld2"; u=ps)
        end
        losses[iter] = l
        wall_clock = [time_ns()]
        dps .= 0
    end
    return ps, (; total=losses)
end

function solve_NDE_postprocessing(ps, params, x₀, ps_baseclosure, sts, NNs, timestep_multiple=2)
    eos = TEOS10EquationOfState()
    coarse_size = params.coarse_size
    Δt = (params.scaled_original_time[2] - params.scaled_original_time[1]) / timestep_multiple
    Nt_solve = (length(params.scaled_original_time) - 1) * timestep_multiple + 1
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

function diagnose_fields(ps, params, x₀, ps_baseclosure, sts, NNs, train_data_plot)
    sols = solve_NDE_postprocessing(ps, params, x₀, ps_baseclosure, sts, NNs)

    ps_noNN = deepcopy(ps) .= 0
    sols_noNN = solve_NDE_postprocessing(ps_noNN, params, x₀, ps_baseclosure, sts, NNs)

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

function animate_data(train_data, scaling, sols, fluxes, diffusivities, sols_noNN, fluxes_noNN, diffusivities_noNN, index, FILE_DIR; coarse_size=32, epoch=1)
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

    wTlim = (find_min(wT_residual, wT_diffusive_boundary, wT_total, train_data.flux.wT.column.unscaled, wT_noNN),
             find_max(wT_residual, wT_diffusive_boundary, wT_total, train_data.flux.wT.column.unscaled, wT_noNN))
    wSlim = (find_min(wS_residual, wS_diffusive_boundary, wS_total, train_data.flux.wS.column.unscaled, wS_noNN),
             find_max(wS_residual, wS_diffusive_boundary, wS_total, train_data.flux.wS.column.unscaled, wS_noNN))

    wTlim = (find_min(wT_residual, wT_diffusive_boundary, wT_total, train_data.flux.wT.column.unscaled, wT_noNN),
             find_max(wT_residual, wT_diffusive_boundary, wT_total, train_data.flux.wT.column.unscaled, wT_noNN))
    wSlim = (find_min(wS_residual, wS_diffusive_boundary, wS_total, train_data.flux.wS.column.unscaled, wS_noNN),
             find_max(wS_residual, wS_diffusive_boundary, wS_total, train_data.flux.wS.column.unscaled, wS_noNN))

    Rilim = (find_min(diffusivities.Ri, diffusivities.Ri_truth, diffusivities_noNN.Ri,), 
             find_max(diffusivities.Ri, diffusivities.Ri_truth, diffusivities_noNN.Ri,),)

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

    Ri_truthₙ = @lift diffusivities.Ri_truth[:, $n]
    Riₙ = @lift diffusivities.Ri[:, $n]
    Ri_noNNₙ = @lift diffusivities_noNN.Ri[:, $n]

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

    lines!(axT, T_truthₙ, zC, label="Truth")
    lines!(axT, T_noNNₙ, zC, label="Base closure only")
    lines!(axT, T_NDEₙ, zC, label="NDE")

    lines!(axS, S_truthₙ, zC, label="Truth")
    lines!(axS, S_noNNₙ, zC, label="Base closure only")
    lines!(axS, S_NDEₙ, zC, label="NDE")

    lines!(axρ, ρ_truthₙ, zC, label="Truth")
    lines!(axρ, ρ_noNNₙ, zC, label="Base closure only")
    lines!(axρ, ρ_NDEₙ, zC, label="NDE")

    lines!(axwT, wT_truthₙ, zF, label="Truth")
    lines!(axwT, wT_noNNₙ, zF, label="Base closure only")
    lines!(axwT, wT_diffusive_boundaryₙ, zF, label="Base closure")
    lines!(axwT, wT_residualₙ, zF, label="Residual")
    lines!(axwT, wT_totalₙ, zF, label="NDE")

    lines!(axwS, wS_truthₙ, zF, label="Truth")
    lines!(axwS, wS_noNNₙ, zF, label="Base closure only")
    lines!(axwS, wS_diffusive_boundaryₙ, zF, label="Base closure")
    lines!(axwS, wS_residualₙ, zF, label="Residual")
    lines!(axwS, wS_totalₙ, zF, label="NDE")

    lines!(axRi, Ri_truthₙ, zF, label="Truth")
    lines!(axRi, Ri_noNNₙ, zF, label="Base closure only")
    lines!(axRi, Riₙ, zF, label="NDE")

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

    CairoMakie.record(fig, "$(FILE_DIR)/training_$(index)_epoch$(epoch).mp4", 1:Nt, framerate=15) do nn
        n[] = nn
    end
end

function plot_loss(losses, FILE_DIR; epoch=1)
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
    save("$(FILE_DIR)/losses_epoch$(epoch).png", fig, px_per_unit=8)
end

optimizers = [Optimisers.Adam(3e-4), Optimisers.Adam(1e-4), Optimisers.Adam(5e-5), Optimisers.Adam(2e-5), Optimisers.Adam(1e-5)]
maxiters = [2500, 2500, 2500, 2500, 2500]

for (epoch, (optimizer, maxiter)) in enumerate(zip(optimizers, maxiters))
    global ps = ps
    ps, losses = train_NDE_multipleics(ps, params, ps_baseclosure, sts, NNs, truths, x₀s; maxiter=maxiter, rule=optimizer, loss_prefactor=loss_prefactor)
    
    sols = [diagnose_fields(ps, param, x₀, ps_baseclosure, sts, NNs, data) for (data, x₀, param) in zip(train_data_plot.data, x₀s, params)]

    for (index, sol) in enumerate(sols)
        animate_data(train_data_plot.data[index], train_data_plot.scaling, sol.sols_dimensional, sol.fluxes, sol.diffusivities, sol.sols_dimensional_noNN, sol.fluxes_noNN, sol.diffusivities_noNN, index, FILE_DIR; epoch=epoch)
    end
    plot_loss(losses, FILE_DIR; epoch=epoch)
end