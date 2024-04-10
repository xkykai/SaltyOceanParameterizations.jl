using SaltyOceanParameterizations
using SaltyOceanParameterizations.DataWrangling
using SaltyOceanParameterizations: calculate_Ri, local_Ri_ν_convectivetanh_shearlinear, local_Ri_κ_convectivetanh_shearlinear
using Oceananigans
using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL, OptimizationOptimisers, Random, SciMLSensitivity, LuxCUDA
using Statistics
using CairoMakie
using SeawaterPolynomials.TEOS10
using Printf
using Dates
using JLD2
using SciMLBase
using Colors
using Distributions
import SeawaterPolynomials.TEOS10: s, ΔS, Sₐᵤ
s(Sᴬ) = Sᴬ + ΔS >= 0 ? √((Sᴬ + ΔS) / Sₐᵤ) : NaN
using Glob
using Optimisers: adjust!

function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

FILE_DIR = "./training_output/NN_relu_2layer_256_local_diffusivity_convectivetanh_shearlinear_TS_freeconvection_mse_profilegradientinput"
mkpath(FILE_DIR)
@info "$(FILE_DIR)"

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

train_data_plot = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, full_timeframes)

rng = Random.default_rng(123)

wT_NN = Chain(Dense(67, 256, relu), Dense(256, 256, relu), Dense(256, 31))
wS_NN = Chain(Dense(67, 256, relu), Dense(256, 256, relu), Dense(256, 31))
# wT_NN = Chain(Dense(67, 4, relu), Dense(4, 4, relu), Dense(4, 31))
# wS_NN = Chain(Dense(67, 4, relu), Dense(4, 4, relu), Dense(4, 31))

ps_wT, st_wT = Lux.setup(rng, wT_NN)
ps_wS, st_wS = Lux.setup(rng, wS_NN)

ps_wT = ps_wT |> ComponentArray .|> Float64
ps_wS = ps_wS |> ComponentArray .|> Float64

ps_wT .= glorot_uniform(rng, Float64, length(ps_wT))
ps_wS .= glorot_uniform(rng, Float64, length(ps_wS))

ps_wT .*= 1e-5
ps_wS .*= 1e-5

NNs = (; wT=wT_NN, wS=wS_NN)
ps_training = ComponentArray(wT=ps_wT, wS=ps_wS)
st_NN = (; wT=st_wT, wS=st_wS)

struct Optimizer{I, L}
    initial :: I
    initial_learning_rate :: L
    learning_rate :: L
    warmup :: Int
    maxiter :: Int
end

function Optimizer(; initial, initial_learning_rate, learning_rate, warmup, maxiter)
    return Optimizer(initial, initial_learning_rate, learning_rate, warmup, maxiter)
end

function train_NDE(train_data, train_data_plot, NNs, ps_training, ps_baseclosure, st_NN, rng; 
                   coarse_size=32, dev=cpu_device(), optimizer=Optimizer(OptimizationOptimisers.ADAM(0.001), 1e-3, 1e-3, 1, 10), solver=ROCK2(), Ri_clamp_lims=(-Inf, Inf), epoch=1)
    train_data = train_data |> dev
    maxiter = optimizer.maxiter
    x₀s = [vcat(data.profile.T.scaled[:, 1], data.profile.S.scaled[:, 1]) for data in train_data.data] |> dev
    eos = TEOS10EquationOfState()
    ps_zeros = deepcopy(ps_training)
    ps_zeros .= 0

    params = [(                   f = data.coriolis.unscaled,
                           f_scaled = data.coriolis.scaled,
                                  τ = data.times[end] - data.times[1],
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
                            ) for (data, plot_data) in zip(train_data.data, train_data_plot.data)] |> dev

    function predict_residual_flux(T_hat, S_hat, ∂ρ∂z_hat, p, params, st)
        x′_T = vcat(T_hat, ∂ρ∂z_hat, params.wT.scaled.top, params.f_scaled)
        x′_S = vcat(S_hat, ∂ρ∂z_hat, params.wS.scaled.top, params.f_scaled)
        
        wT = vcat(0, first(NNs.wT(x′_T, p.wT, st.wT)), 0)
        wS = vcat(0, first(NNs.wS(x′_S, p.wS, st.wS)), 0)

        return wT, wS
    end

    function predict_residual_flux_dimensional(T_hat, S_hat, ∂ρ∂z_hat, p, params, st)
        wT_hat, wS_hat = predict_residual_flux(T_hat, S_hat, ∂ρ∂z_hat, p, params, st)
        
        wT = inv(params.scaling.wT).(wT_hat)
        wS = inv(params.scaling.wS).(wS_hat)

        wT = wT .- wT[1]
        wS = wS .- wS[1]

        return wT, wS
    end

    function predict_boundary_flux(params)
        wT = vcat(fill(params.wT.scaled.bottom, coarse_size), params.wT.scaled.top)
        wS = vcat(fill(params.wS.scaled.bottom, coarse_size), params.wS.scaled.top)

        return wT, wS
    end

    function predict_diffusivities(Ris, p)
        νs = local_Ri_ν_convectivetanh_shearlinear.(Ris, ps_baseclosure.ν_conv, ps_baseclosure.ν_shear, ps_baseclosure.m, ps_baseclosure.ΔRi)
        κs = local_Ri_κ_convectivetanh_shearlinear.(νs, ps_baseclosure.Pr)
        return νs, κs
    end

    function predict_diffusive_flux(ρ, T_hat, S_hat, p, params, st)
        Ris = calculate_Ri(zeros(coarse_size), zeros(coarse_size), ρ, params.Dᶠ, params.g, eos.reference_density, clamp_lims=Ri_clamp_lims)
        _, κs = predict_diffusivities(Ris, p)

        ∂T∂z_hat = params.Dᶠ_hat * T_hat
        ∂S∂z_hat = params.Dᶠ_hat * S_hat

        wT_diffusive = -κs .* ∂T∂z_hat
        wS_diffusive = -κs .* ∂S∂z_hat
        return wT_diffusive, wS_diffusive
    end

    function predict_diffusive_boundary_flux_dimensional(ρ, T_hat, S_hat, p, params, st)
        _wT_diffusive, _wS_diffusive = predict_diffusive_flux(ρ, T_hat, S_hat, p, params, st)
        _wT_boundary, _wS_boundary = predict_boundary_flux(params)

        wT_diffusive = params.scaling.T.σ / params.H .* _wT_diffusive
        wS_diffusive = params.scaling.S.σ / params.H .* _wS_diffusive

        wT_boundary = inv(params.scaling.wT).(_wT_boundary)
        wS_boundary = inv(params.scaling.wS).(_wS_boundary)

        wT = wT_diffusive .+ wT_boundary
        wS = wS_diffusive .+ wS_boundary

        return wT, wS
    end

    get_T(x::Vector, params) = x[1:params.coarse_size]
    get_S(x::Vector, params) = x[params.coarse_size+1:2*params.coarse_size]

    get_T(x::Matrix, params) = @view x[1:params.coarse_size, :]
    get_S(x::Matrix, params) = @view x[params.coarse_size+1:2*params.coarse_size, :]


    function get_scaled_profiles(x, params)
        T_hat = get_T(x, params)
        S_hat = get_S(x, params)

        return T_hat, S_hat
    end

    function unscale_profiles(T_hat, S_hat, params)
        T = inv(params.scaling.T).(T_hat)
        S = inv(params.scaling.S).(S_hat)

        return T, S
    end

    function calculate_unscaled_density(T, S)
        return TEOS10.ρ.(T, S, 0, Ref(eos))
    end

    function NDE(x, p, t, params, st)
        f = params.f
        Dᶜ_hat = params.Dᶜ_hat
        Dᶠ = params.Dᶠ
        scaling = params.scaling
        τ, H = params.τ, params.H

        T_hat, S_hat = get_scaled_profiles(x, params)
        T, S = unscale_profiles(T_hat, S_hat, params)

        ρ = calculate_unscaled_density(T, S)
        ∂ρ∂z_hat = scaling.∂ρ∂z.(Dᶠ * ρ)
        
        wT_residual, wS_residual = predict_residual_flux(T_hat, S_hat, ∂ρ∂z_hat, p, params, st)
        wT_boundary, wS_boundary = predict_boundary_flux(params)
        wT_diffusive, wS_diffusive = predict_diffusive_flux(ρ, T_hat, S_hat, p, params, st)

        dT = -τ / H^2 .* (Dᶜ_hat * wT_diffusive) .- τ / H * scaling.wT.σ / scaling.T.σ .* (Dᶜ_hat * (wT_boundary .+ wT_residual))
        dS = -τ / H^2 .* (Dᶜ_hat * wS_diffusive) .- τ / H * scaling.wS.σ / scaling.S.σ .* (Dᶜ_hat * (wS_boundary .+ wS_residual))

        return vcat(dT, dS)
    end

    function predict_NDE(p)
        probs = [ODEProblem((x, p′, t) -> NDE(x, p′, t, param, st_NN), x₀, (param.scaled_time[1], param.scaled_time[end]), p) for (x₀, param) in zip(x₀s, params)]
        sols = [Array(solve(prob, solver, saveat=param.scaled_time, reltol=1e-7, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP(), checkpointing=true))) for (param, prob) in zip(params, probs)]
        return sols
    end

    function predict_NDE_posttraining(p)
        probs = [ODEProblem((x, p′, t) -> NDE(x, p′, t, param, st_NN), x₀, (param.scaled_original_time[1], param.scaled_original_time[end]), p) for (x₀, param) in zip(x₀s, params)]
        sols = [solve(prob, solver, saveat=param.scaled_original_time, reltol=1e-7) for (param, prob) in zip(params, probs)]
        return sols
    end

    function predict_NDE_noNN()
        probs = [ODEProblem((x, p′, t) -> NDE(x, p′, t, param, st_NN), x₀, (param.scaled_original_time[1], param.scaled_original_time[end]), ps_zeros) for (x₀, param) in zip(x₀s, params)]
        sols = [solve(prob, solver, saveat=param.scaled_original_time, reltol=1e-7) for (param, prob) in zip(params, probs)]
        return sols
    end

    function predict_scaled_profiles(p)
        preds = predict_NDE(p)
        T_hats = [get_T(pred, param) for (pred, param) in zip(preds, params)]
        S_hats = [get_S(pred, param) for (pred, param) in zip(preds, params)]
        
        Ts = [inv(param.scaling.T).(T_hat) for (T_hat, param) in zip(T_hats, params)]
        Ss = [inv(param.scaling.S).(S_hat) for (S_hat, param) in zip(S_hats, params)]

        ρs = [calculate_unscaled_density(T, S) for (T, S) in zip(Ts, Ss)]
        ρ_hats = [param.scaling.ρ.(ρ) for (ρ, param) in zip(ρs, params)]

        ∂T∂z_hats = [hcat([param.scaling.∂T∂z.(param.Dᶠ * @view(T[:, i])) for i in axes(T, 2)]...) for (param, T) in zip(params, Ts)]
        ∂S∂z_hats = [hcat([param.scaling.∂S∂z.(param.Dᶠ * @view(S[:, i])) for i in axes(S, 2)]...) for (param, S) in zip(params, Ss)]
        ∂ρ∂z_hats = [hcat([param.scaling.∂ρ∂z.(param.Dᶠ * @view(ρ[:, i])) for i in axes(ρ, 2)]...) for (param, ρ) in zip(params, ρs)]

        return T_hats, S_hats, ρ_hats, ∂T∂z_hats, ∂S∂z_hats, ∂ρ∂z_hats, preds
    end


    function predict_losses(T, S, ρ, ∂T∂z, ∂S∂z, ∂ρ∂z, loss_scaling=(; T=1, S=1, ρ=1, ∂T∂z=1, ∂S∂z=1, ∂ρ∂z=1))
        T_loss = loss_scaling.T * mean(mean.([(data.profile.T.scaled .- T).^2 for (data, T) in zip(train_data.data, T)]))
        S_loss = loss_scaling.S * mean(mean.([(data.profile.S.scaled .- S).^2 for (data, S) in zip(train_data.data, S)]))
        ρ_loss = loss_scaling.ρ * mean(mean.([(data.profile.ρ.scaled .- ρ).^2 for (data, ρ) in zip(train_data.data, ρ)]))

        ∂T∂z_loss = loss_scaling.∂T∂z * mean(mean.([(data.profile.∂T∂z.scaled .- ∂T∂z).^2 for (data, ∂T∂z) in zip(train_data.data, ∂T∂z)]))
        ∂S∂z_loss = loss_scaling.∂S∂z * mean(mean.([(data.profile.∂S∂z.scaled .- ∂S∂z).^2 for (data, ∂S∂z) in zip(train_data.data, ∂S∂z)]))
        ∂ρ∂z_loss = loss_scaling.∂ρ∂z * mean(mean.([(data.profile.∂ρ∂z.scaled .- ∂ρ∂z).^2 for (data, ∂ρ∂z) in zip(train_data.data, ∂ρ∂z)]))

        return (; T=T_loss, S=S_loss, ρ=ρ_loss, ∂T∂z=∂T∂z_loss, ∂S∂z=∂S∂z_loss, ∂ρ∂z=∂ρ∂z_loss)
    end

    function compute_loss_prefactor(T_loss, S_loss, ρ_loss, ∂T∂z_loss, ∂S∂z_loss, ∂ρ∂z_loss)
        ρ_prefactor = 1
        T_prefactor = ρ_loss / T_loss
        S_prefactor = ρ_loss / S_loss

        ∂ρ∂z_prefactor = 1
        ∂T∂z_prefactor = ∂ρ∂z_loss / ∂T∂z_loss
        ∂S∂z_prefactor = ∂ρ∂z_loss / ∂S∂z_loss

        profile_loss = T_prefactor * T_loss + S_prefactor * S_loss + ρ_prefactor * ρ_loss
        gradient_loss = ∂T∂z_prefactor * ∂T∂z_loss + ∂S∂z_prefactor * ∂S∂z_loss + ∂ρ∂z_prefactor * ∂ρ∂z_loss

        gradient_prefactor = profile_loss / gradient_loss

        ∂ρ∂z_prefactor *= gradient_prefactor
        ∂T∂z_prefactor *= gradient_prefactor
        ∂S∂z_prefactor *= gradient_prefactor

        return (T=T_prefactor, S=S_prefactor, ρ=ρ_prefactor, ∂T∂z=∂T∂z_prefactor, ∂S∂z=∂S∂z_prefactor, ∂ρ∂z=∂ρ∂z_prefactor)
    end

    function compute_loss_prefactor(p)
        T_hats, S_hats, ρ_hats, ∂T∂z_hats, ∂S∂z_hats, ∂ρ∂z_hats, _ = predict_scaled_profiles(p)
        losses = predict_losses(T_hats, S_hats, ρ_hats, ∂T∂z_hats, ∂S∂z_hats, ∂ρ∂z_hats)
        return compute_loss_prefactor(losses...)
    end

    @info "Computing prefactor for losses"

    losses_prefactor = compute_loss_prefactor(ps_training)

    function loss_NDE(p)
        T_hats, S_hats, ρ_hats, ∂T∂z_hats, ∂S∂z_hats, ∂ρ∂z_hats, preds = predict_scaled_profiles(p)
        individual_loss = predict_losses(T_hats, S_hats, ρ_hats, ∂T∂z_hats, ∂S∂z_hats, ∂ρ∂z_hats, losses_prefactor)
        loss = sum(values(individual_loss))

        return loss, preds, individual_loss
    end

    iter = 0

    losses = zeros(maxiter+1)
    T_losses = zeros(maxiter+1)
    S_losses = zeros(maxiter+1)
    ρ_losses = zeros(maxiter+1)
    ∂T∂z_losses = zeros(maxiter+1)
    ∂S∂z_losses = zeros(maxiter+1)
    ∂ρ∂z_losses = zeros(maxiter+1)

    wall_clock = [time_ns()]

    iter_warmup = optimizer.warmup
    max_learningrate = optimizer.learning_rate
    min_learningrate = optimizer.initial_learning_rate

    callback = function (p, l, pred, ind_loss)
        @printf("%s, Δt %s, iter %d/%d, loss total %6.10e, T %6.5e, S %6.5e, ρ %6.5e, ∂T∂z %6.5e, ∂S∂z %6.5e, ∂ρ∂z %6.5e, max NN weight %6.5e\n",
                Dates.now(), prettytime(1e-9 * (time_ns() - wall_clock[1])), iter, maxiter, l, 
                ind_loss.T, ind_loss.S, ind_loss.ρ, 
                ind_loss.∂T∂z, ind_loss.∂S∂z, ind_loss.∂ρ∂z,
                maximum(abs, p.u))
        if iter % 200 == 0
            jldsave("$(FILE_DIR)/intermediate_training_results_epoch$(epoch)_iter$(iter).jld2"; u=p.u)
        end
        losses[iter+1] = l
        T_losses[iter+1] = ind_loss.T
        S_losses[iter+1] = ind_loss.S
        ρ_losses[iter+1] = ind_loss.ρ
        ∂T∂z_losses[iter+1] = ind_loss.∂T∂z
        ∂S∂z_losses[iter+1] = ind_loss.∂S∂z
        ∂ρ∂z_losses[iter+1] = ind_loss.∂ρ∂z

        iter += 1

        if iter <= iter_warmup
            adjust!(p.original, min_learningrate + (max_learningrate - min_learningrate) / iter_warmup * iter)
        end

        wall_clock[1] = time_ns()
        return false
    end

    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_NDE(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, ps_training)

    @info "Training NDE"
    res = Optimization.solve(optprob, optimizer.initial, callback=callback, maxiters=maxiter)

    sols_posttraining = predict_NDE_posttraining(res.u)
    sols_noNN = predict_NDE_noNN()

    wT_diffusive_boundary_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    wS_diffusive_boundary_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]

    wT_diffusive_boundary_noNN = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    wS_diffusive_boundary_noNN = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]

    wT_residual_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    wS_residual_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]

    νs_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    κs_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    
    νs_noNN = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    κs_noNN = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    
    Ri_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    Ri_noNN = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    Ri_truth = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]

    for (i, sol) in enumerate(sols_posttraining)
        for t in eachindex(sol.t)
            T_hat, S_hat = get_scaled_profiles(sol[:, t], params[i])
            T, S = unscale_profiles(T_hat, S_hat, params[i])

            ρ = calculate_unscaled_density(T, S)

            ∂ρ∂z_hat = params[i].scaling.∂ρ∂z.(params[i].Dᶠ * ρ)

            wT_diffusive_boundary, wS_diffusive_boundary = predict_diffusive_boundary_flux_dimensional(ρ, T_hat, S_hat, res.u, params[i], st_NN)
            wT_residual, wS_residual = predict_residual_flux_dimensional(T_hat, S_hat, ∂ρ∂z_hat, res.u, params[i], st_NN)

            Ris = calculate_Ri(zeros(coarse_size), zeros(coarse_size), ρ, params[i].Dᶠ, params[i].g, eos.reference_density, clamp_lims=Ri_clamp_lims)

            Ris_truth = calculate_Ri(zeros(coarse_size), 
                                     zeros(coarse_size),
                                     train_data_plot.data[i].profile.ρ.unscaled[:, t], 
                                     params[i].Dᶠ, params[i].g, eos.reference_density, clamp_lims=Ri_clamp_lims)

            νs, κs = predict_diffusivities(Ris, res.u)

            wT_residual_posttraining[i][:, t] .= wT_residual
            wS_residual_posttraining[i][:, t] .= wS_residual

            wT_diffusive_boundary_posttraining[i][:, t] .= wT_diffusive_boundary
            wS_diffusive_boundary_posttraining[i][:, t] .= wS_diffusive_boundary

            νs_posttraining[i][:, t] .= νs
            κs_posttraining[i][:, t] .= κs

            Ri_posttraining[i][:, t] .= Ris
            Ri_truth[i][:, t] .= Ris_truth
        end
    end

    for (i, sol) in enumerate(sols_noNN)
        for t in eachindex(sol.t)
            T_hat, S_hat = get_scaled_profiles(sol[:, t], params[i])

            T, S = unscale_profiles(T_hat, S_hat, params[i])
            ρ = calculate_unscaled_density(T, S)

            wT_diffusive_boundary, wS_diffusive_boundary = predict_diffusive_boundary_flux_dimensional(ρ, T_hat, S_hat, res.u, params[i], st_NN)
            Ris = calculate_Ri(zeros(coarse_size), zeros(coarse_size), ρ, params[i].Dᶠ, params[i].g, eos.reference_density, clamp_lims=Ri_clamp_lims)

            νs, κs = predict_diffusivities(Ris, res.u)

            wT_diffusive_boundary_noNN[i][:, t] .= wT_diffusive_boundary
            wS_diffusive_boundary_noNN[i][:, t] .= wS_diffusive_boundary

            νs_noNN[i][:, t] .= νs
            κs_noNN[i][:, t] .= κs

            Ri_noNN[i][:, t] .= Ris
        end
    end

    wT_total_posttraining = wT_residual_posttraining .+ wT_diffusive_boundary_posttraining
    wS_total_posttraining = wS_residual_posttraining .+ wS_diffusive_boundary_posttraining

    flux_posttraining = (wT = (diffusive_boundary=wT_diffusive_boundary_posttraining, residual=wT_residual_posttraining, total=wT_total_posttraining),
                         wS = (diffusive_boundary=wS_diffusive_boundary_posttraining, residual=wS_residual_posttraining, total=wS_total_posttraining))
                        
    flux_noNN = (; wT = (; total=wT_diffusive_boundary_noNN),
                  wS = (; total=wS_diffusive_boundary_noNN))

    diffusivities_posttraining = (ν=νs_posttraining, κ=κs_posttraining, Ri=Ri_posttraining, Ri_truth=Ri_truth)
    diffusivities_noNN = (ν=νs_noNN, κ=κs_noNN, Ri=Ri_noNN)

    losses = (total=losses, T=T_losses, S=S_losses, ρ=ρ_losses, ∂T∂z=∂T∂z_losses, ∂S∂z=∂S∂z_losses, ∂ρ∂z=∂ρ∂z_losses)

    return res, loss_NDE(res.u), sols_posttraining, flux_posttraining, losses, diffusivities_posttraining, sols_noNN, flux_noNN, diffusivities_noNN
end

#%%
function plot_loss(losses, FILE_DIR; epoch=1)
    colors = distinguishable_colors(10, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    
    fig = Figure(size=(1000, 600))
    axtotalloss = CairoMakie.Axis(fig[1, 1], title="Total Loss", xlabel="Iterations", ylabel="Loss", yscale=log10)
    axindividualloss = CairoMakie.Axis(fig[1, 2], title="Individual Loss", xlabel="Iterations", ylabel="Loss", yscale=log10)

    lines!(axtotalloss, losses.total, label="Total Loss", color=colors[1])

    lines!(axindividualloss, losses.T, label="T", color=colors[3])
    lines!(axindividualloss, losses.S, label="S", color=colors[4])
    lines!(axindividualloss, losses.ρ, label="ρ", color=colors[5])
    lines!(axindividualloss, losses.∂T∂z, label="∂T∂z", color=colors[8])
    lines!(axindividualloss, losses.∂S∂z, label="∂S∂z", color=colors[9])
    lines!(axindividualloss, losses.∂ρ∂z, label="∂ρ∂z", color=colors[10])

    axislegend(axindividualloss, position=:rt)
    save("$(FILE_DIR)/losses_epoch$(epoch).png", fig, px_per_unit=8)
end

#%%
function animate_data(train_data, scaling, sols, fluxes, diffusivities, sols_noNN, fluxes_noNN, diffusivities_noNN, index, FILE_DIR; coarse_size=32, epoch=1)    fig = Figure(size=(1920, 1080))
    axT = CairoMakie.Axis(fig[1, 1], title="T", xlabel="T (°C)", ylabel="z (m)")
    axS = CairoMakie.Axis(fig[1, 2], title="S", xlabel="S (g kg⁻¹)", ylabel="z (m)")
    axρ = CairoMakie.Axis(fig[1, 3], title="ρ", xlabel="ρ (kg m⁻³)", ylabel="z (m)")
    axwT = CairoMakie.Axis(fig[2, 1], title="wT", xlabel="wT (m s⁻¹ °C)", ylabel="z (m)")
    axwS = CairoMakie.Axis(fig[2, 2], title="wS", xlabel="wS (m s⁻¹ g kg⁻¹)", ylabel="z (m)")
    axRi = CairoMakie.Axis(fig[1, 4], title="Ri", xlabel="Ri", ylabel="z (m)")
    axdiffusivity = CairoMakie.Axis(fig[2, 3], title="Diffusivity", xlabel="Diffusivity (m² s⁻¹)", ylabel="z (m)")

    n = Observable(1)
    zC = train_data.data[index].metadata["zC"]
    zF = train_data.data[index].metadata["zF"]

    T_NDE = inv(train_data.scaling.T).(sols[index][1:coarse_size, :])
    S_NDE = inv(train_data.scaling.S).(sols[index][coarse_size+1:2*coarse_size, :])
    ρ_NDE = TEOS10.ρ.(T_NDE, S_NDE, 0, Ref(TEOS10EquationOfState()))

    wT_residual = fluxes.wT.residual[index]
    wS_residual = fluxes.wS.residual[index]

    wT_diffusive_boundary = fluxes.wT.diffusive_boundary[index]
    wS_diffusive_boundary = fluxes.wS.diffusive_boundary[index]

    wT_total = fluxes.wT.total[index]
    wS_total = fluxes.wS.total[index]

    T_noNN = inv(scaling.T).(sols_noNN[index][1:coarse_size, :])
    S_noNN = inv(scaling.S).(sols_noNN[index][coarse_size+1:2*coarse_size, :])
    ρ_noNN = TEOS10.ρ.(T_noNN, S_noNN, 0, Ref(TEOS10EquationOfState()))

    wT_noNN = fluxes_noNN.wT.total[index]
    wS_noNN = fluxes_noNN.wS.total[index]

    Tlim = (find_min(T_NDE, train_data.data[index].profile.T.unscaled, T_noNN), find_max(T_NDE, train_data.data[index].profile.T.unscaled, T_noNN))
    Slim = (find_min(S_NDE, train_data.data[index].profile.S.unscaled, S_noNN), find_max(S_NDE, train_data.data[index].profile.S.unscaled, S_noNN))
    ρlim = (find_min(ρ_NDE, train_data.data[index].profile.ρ.unscaled, ρ_noNN), find_max(ρ_NDE, train_data.data[index].profile.ρ.unscaled, ρ_noNN))

    wTlim = (find_min(wT_residual, wT_diffusive_boundary, wT_total, train_data.data[index].flux.wT.column.unscaled, wT_noNN),
             find_max(wT_residual, wT_diffusive_boundary, wT_total, train_data.data[index].flux.wT.column.unscaled, wT_noNN))
    wSlim = (find_min(wS_residual, wS_diffusive_boundary, wS_total, train_data.data[index].flux.wS.column.unscaled, wS_noNN),
             find_max(wS_residual, wS_diffusive_boundary, wS_total, train_data.data[index].flux.wS.column.unscaled, wS_noNN))

    wTlim = (find_min(wT_residual, wT_diffusive_boundary, wT_total, train_data.data[index].flux.wT.column.unscaled, wT_noNN),
             find_max(wT_residual, wT_diffusive_boundary, wT_total, train_data.data[index].flux.wT.column.unscaled, wT_noNN))
    wSlim = (find_min(wS_residual, wS_diffusive_boundary, wS_total, train_data.data[index].flux.wS.column.unscaled, wS_noNN),
             find_max(wS_residual, wS_diffusive_boundary, wS_total, train_data.data[index].flux.wS.column.unscaled, wS_noNN))

    Rilim = (find_min(diffusivities.Ri[index], diffusivities.Ri_truth[index], diffusivities_noNN.Ri[index],), 
             find_max(diffusivities.Ri[index], diffusivities.Ri_truth[index], diffusivities_noNN.Ri[index],),)

    diffusivitylim = (find_min(diffusivities.ν[index], diffusivities.κ[index], diffusivities_noNN.ν[index], diffusivities_noNN.κ[index]), 
                      find_max(diffusivities.ν[index], diffusivities.κ[index], diffusivities_noNN.ν[index], diffusivities_noNN.κ[index]),)

    T_truthₙ = @lift train_data.data[index].profile.T.unscaled[:, $n]
    S_truthₙ = @lift train_data.data[index].profile.S.unscaled[:, $n]
    ρ_truthₙ = @lift train_data.data[index].profile.ρ.unscaled[:, $n]

    wT_truthₙ = @lift train_data.data[index].flux.wT.column.unscaled[:, $n]
    wS_truthₙ = @lift train_data.data[index].flux.wS.column.unscaled[:, $n]

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

    Ri_truthₙ = @lift diffusivities.Ri_truth[index][:, $n]
    Riₙ = @lift diffusivities.Ri[index][:, $n]
    Ri_noNNₙ = @lift diffusivities_noNN.Ri[index][:, $n]

    νₙ = @lift diffusivities.ν[index][:, $n]
    κₙ = @lift diffusivities.κ[index][:, $n]

    ν_noNNₙ = @lift diffusivities_noNN.ν[index][:, $n]
    κ_noNNₙ = @lift diffusivities_noNN.κ[index][:, $n]

    Qᵀ = train_data.data[index].metadata["temperature_flux"]
    Qˢ = train_data.data[index].metadata["salinity_flux"]
    f = train_data.data[index].metadata["coriolis_parameter"]
    times = train_data.data[index].times
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

optimizers = [Optimizer(initial=OptimizationOptimisers.Adam(1e-6), initial_learning_rate=1e-6, learning_rate=1e-3, warmup=40, maxiter=1000),
              Optimizer(initial=OptimizationOptimisers.Adam(1e-6), initial_learning_rate=1e-6, learning_rate=3e-4, warmup=40, maxiter=1000)]

# optimizers = [Optimizer(initial=OptimizationOptimisers.Adam(1e-6), initial_learning_rate=1e-6, learning_rate=5e-5, warmup=10, maxiter=3)]

for (epoch, optimizer) in enumerate(optimizers)
    res, loss, sols, fluxes, losses, diffusivities, sols_noNN, fluxes_noNN, diffusivities_noNN = train_NDE(train_data, train_data_plot, NNs, ps_training, ps_baseclosure, st_NN, rng, solver=ROCK2(), optimizer=optimizer, epoch=epoch)
    u = res.u
    jldsave("$(FILE_DIR)/training_results_$(epoch).jld2"; res, u, loss, sols, fluxes, losses, NNs, st_NN, diffusivities, sols_noNN, fluxes_noNN, diffusivities_noNN)
    plot_loss(losses, FILE_DIR, epoch=epoch)
    for i in eachindex(field_datasets)
        animate_data(train_data_plot, train_data.scaling, sols, fluxes, diffusivities, sols_noNN, fluxes_noNN, diffusivities_noNN, i, FILE_DIR, epoch=epoch)
    end
    ps_training .= u
end

rm.(glob("$(FILE_DIR)/intermediate_training_results_*.jld2"))