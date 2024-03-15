using SaltyOceanParameterizations
using SaltyOceanParameterizations.DataWrangling
using SaltyOceanParameterizations: calculate_Ri
using Oceananigans
using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL, OptimizationOptimisers, Random, SciMLSensitivity, LuxCUDA
using Statistics
using CairoMakie
using SeawaterPolynomials.TEOS10
using Printf
using Dates
using JLD2
using Colors

function find_min(a...)
    return minimum(minimum.([a...]))
end
  
function find_max(a...)
    return maximum(maximum.([a...]))
end

FILE_DIR = "./training_output/slightly_nonlocal_diffusivity_NDE_gradient_swish_softplus_noclamp_small"
@info FILE_DIR
mkpath(FILE_DIR)

LES_FILE_DIRS = [
    "./LES_training/linearTS_dTdz_0.0013_dSdz_-0.0014_QU_-0.0002_QT_3.0e-5_QS_-3.0e-5_T_4.3_S_33.5_f_-0.00012_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.0002_QT_0.0003_QS_-3.0e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.0002_QT_0.0003_QS_-3.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_-0.025_dSdz_-0.0045_QU_-0.0002_QT_-0.0003_QS_-3.0e-5_T_-3.6_S_33.9_f_-0.000125_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
]

field_datasets = [FieldDataset(FILE_DIR, backend=OnDisk()) for FILE_DIR in LES_FILE_DIRS]

full_timeframes = [1:length(data["ubar"].times) for data in field_datasets]
timeframes = [5:5:length(data["ubar"].times) for data in field_datasets]
train_data = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, timeframes)
coarse_size = 32

train_data_plot = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, full_timeframes)

rng = Random.default_rng(123)

NN = Chain(Dense(2, 4, swish), Dense(4, 2, softplus))

ps_NN, st_NN = Lux.setup(rng, NN)
ps_NN = ps_NN |> ComponentArray .|> Float64

ps_NN ./= 100

function train_NDE(train_data, train_data_plot, NN, ps_NN, st_NN; coarse_size=32, dev=cpu_device(), maxiter=10, optimizer=OptimizationOptimisers.ADAM(0.001), solver=DP5(), Ri_clamp_lims=(-10, 10))
    train_data = train_data |> dev
    x₀s = [vcat(data.profile.u.scaled[:, 1], data.profile.v.scaled[:, 1], data.profile.T.scaled[:, 1], data.profile.S.scaled[:, 1]) for data in train_data.data] |> dev
    eos = TEOS10EquationOfState()

    params = [(                   f = data.metadata["coriolis_parameter"],
                                  τ = data.times[end] - data.times[1],
                        scaled_time = (data.times .- data.times[1]) ./ (data.times[end] - data.times[1]),
               scaled_original_time = data.metadata["original_times"] ./ (data.metadata["original_times"][end] - data.metadata["original_times"][1]),
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
                            scaling = train_data.scaling
               ) for data in train_data.data] |> dev

    function predict_diffusivities(Ris, p, st)
        ν₀ = 1e-5
        κ₀ = 1e-5

        diffusivities = [first(NN(Ris[i:i+1], p, st)) for i in 2:length(Ris)-1]

        νs = vcat(0, [diffusivity[1] / 10 + ν₀ for diffusivity in diffusivities], 0)
        κs = vcat(0, [diffusivity[2] / 10 + κ₀ for diffusivity in diffusivities], 0)
        return νs, κs
    end

    function predict_diffusive_flux(x, p, params, st)
        u_hat = x[1:coarse_size]
        v_hat = x[coarse_size+1:2*coarse_size]
        T_hat = x[2*coarse_size+1:3*coarse_size]
        S_hat = x[3*coarse_size+1:4*coarse_size]

        u = inv(params.scaling.u).(u_hat)
        v = inv(params.scaling.v).(v_hat)
        T = inv(params.scaling.T).(T_hat)
        S = inv(params.scaling.S).(S_hat)

        Ris = calculate_Ri(u, v, T, S, params.Dᶠ, params.g, eos.reference_density, clamp_lims=Ri_clamp_lims)

        νs, κs = predict_diffusivities(Ris, p, st)

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

    function predict_boundary_flux(params)
        uw_boundary = vcat(fill(params.uw.scaled.bottom, coarse_size), params.uw.scaled.top)
        vw_boundary = vcat(fill(params.vw.scaled.bottom, coarse_size), params.vw.scaled.top)
        wT_boundary = vcat(fill(params.wT.scaled.bottom, coarse_size), params.wT.scaled.top)
        wS_boundary = vcat(fill(params.wS.scaled.bottom, coarse_size), params.wS.scaled.top)

        return uw_boundary, vw_boundary, wT_boundary, wS_boundary
    end

    function predict_total_flux_dimensional(x, p, params, st)
        _uw_diffusive, _vw_diffusive, _wT_diffusive, _wS_diffusive = predict_diffusive_flux(x, p, params, st)
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

    function NDE(x, p, t, params, st)
        coarse_size = params.coarse_size
        f = params.f
        Dᶜ_hat = params.Dᶜ_hat
        scaling = params.scaling
        τ, H = params.τ, params.H

        u = x[1:coarse_size]
        v = x[coarse_size+1:2*coarse_size]

        uw_diffusive, vw_diffusive, wT_diffusive, wS_diffusive = predict_diffusive_flux(x, p, params, st)
        uw_boundary, vw_boundary, wT_boundary, wS_boundary = predict_boundary_flux(params)

        du = -τ / H^2 .* (Dᶜ_hat * uw_diffusive) .- τ / H * scaling.uw.σ / scaling.u.σ .* (Dᶜ_hat * uw_boundary) .+ f * τ ./ scaling.u.σ .* inv(scaling.v).(v)
        dv = -τ / H^2 .* (Dᶜ_hat * vw_diffusive) .- τ / H * scaling.vw.σ / scaling.v.σ .* (Dᶜ_hat * vw_boundary) .- f * τ ./ scaling.v.σ .* inv(scaling.u).(u)
        dT = -τ / H^2 .* (Dᶜ_hat * wT_diffusive) .- τ / H * scaling.wT.σ / scaling.T.σ .* (Dᶜ_hat * wT_boundary)
        dS = -τ / H^2 .* (Dᶜ_hat * wS_diffusive) .- τ / H * scaling.wS.σ / scaling.S.σ .* (Dᶜ_hat * wS_boundary)

        return vcat(du, dv, dT, dS)
    end

    function predict_NDE(p)
        probs = [ODEProblem((x, p′, t) -> NDE(x, p′, t, param, st_NN), x₀, (param.scaled_time[1], param.scaled_time[end]), p) for (x₀, param) in zip(x₀s, params)]
        sols = [Array(solve(prob, solver, saveat=param.scaled_time, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), reltol=1e-3)) for (param, prob) in zip(params, probs)]
        return sols
    end

    function predict_NDE_posttraining(p)
        probs = [ODEProblem((x, p′, t) -> NDE(x, p′, t, param, st_NN), x₀, (param.scaled_original_time[1], param.scaled_original_time[end]), p) for (x₀, param) in zip(x₀s, params)]
        sols = [solve(prob, solver, saveat=param.scaled_original_time, reltol=1e-3) for (param, prob) in zip(params, probs)]
        return sols
    end

    function compute_loss_prefactor(p)
        preds = predict_NDE(p)

        us = [@view(pred[1:coarse_size, :]) for pred in preds]
        vs = [@view(pred[coarse_size+1:2*coarse_size, :]) for pred in preds]
        Ts = [@view(pred[2*coarse_size+1:3*coarse_size, :]) for pred in preds]
        Ss = [@view(pred[3*coarse_size+1:4*coarse_size, :]) for pred in preds]
        ρs = [param.scaling.ρ.(TEOS10.ρ.(inv(param.scaling.T).(T), inv(param.scaling.S).(S), 0, Ref(eos))) for (T, S, param) in zip(Ts, Ss, params)]

        ∂u∂zs = [hcat([param.scaling.∂u∂z.(param.Dᶠ * inv(param.scaling.u).(@view(u[:, i]))) for i in axes(u, 2)]...) for (param, u) in zip(params, us)]
        ∂v∂zs = [hcat([param.scaling.∂v∂z.(param.Dᶠ * inv(param.scaling.v).(@view(v[:, i]))) for i in axes(v, 2)]...) for (param, v) in zip(params, vs)]
        ∂T∂zs = [hcat([param.scaling.∂T∂z.(param.Dᶠ * inv(param.scaling.T).(@view(T[:, i]))) for i in axes(T, 2)]...) for (param, T) in zip(params, Ts)]
        ∂S∂zs = [hcat([param.scaling.∂S∂z.(param.Dᶠ * inv(param.scaling.S).(@view(S[:, i]))) for i in axes(S, 2)]...) for (param, S) in zip(params, Ss)]
        ∂ρ∂zs = [hcat([param.scaling.∂ρ∂z.(param.Dᶠ * inv(param.scaling.ρ).(@view(ρ[:, i]))) for i in axes(ρ, 2)]...) for (param, ρ) in zip(params, ρs)]

        u_loss = mean(mean.([(data.profile.u.scaled .- u).^2 for (data, u) in zip(train_data.data, us)]))
        v_loss = mean(mean.([(data.profile.v.scaled .- v).^2 for (data, v) in zip(train_data.data, vs)]))
        T_loss = mean(mean.([(data.profile.T.scaled .- T).^2 for (data, T) in zip(train_data.data, Ts)]))
        S_loss = mean(mean.([(data.profile.S.scaled .- S).^2 for (data, S) in zip(train_data.data, Ss)]))
        ρ_loss = mean(mean.([(data.profile.ρ.scaled .- ρ).^2 for (data, ρ) in zip(train_data.data, ρs)]))
        ∂u∂z_loss = mean(mean.([(data.profile.∂u∂z.scaled .- ∂u∂z).^2 for (data, ∂u∂z) in zip(train_data.data, ∂u∂zs)]))
        ∂v∂z_loss = mean(mean.([(data.profile.∂v∂z.scaled .- ∂v∂z).^2 for (data, ∂v∂z) in zip(train_data.data, ∂v∂zs)]))
        ∂T∂z_loss = mean(mean.([(data.profile.∂T∂z.scaled .- ∂T∂z).^2 for (data, ∂T∂z) in zip(train_data.data, ∂T∂zs)]))
        ∂S∂z_loss = mean(mean.([(data.profile.∂S∂z.scaled .- ∂S∂z).^2 for (data, ∂S∂z) in zip(train_data.data, ∂S∂zs)]))
        ∂ρ∂z_loss = mean(mean.([(data.profile.∂ρ∂z.scaled .- ∂ρ∂z).^2 for (data, ∂ρ∂z) in zip(train_data.data, ∂ρ∂zs)]))

        ρ_prefactor = 1
        T_prefactor = ρ_loss / T_loss
        S_prefactor = ρ_loss / S_loss
        u_prefactor = ρ_loss / u_loss * (0.05/0.3)
        v_prefactor = ρ_loss / v_loss * (0.05/0.3)

        ∂ρ∂z_prefactor = 1
        ∂T∂z_prefactor = ∂ρ∂z_loss / ∂T∂z_loss
        ∂S∂z_prefactor = ∂ρ∂z_loss / ∂S∂z_loss
        ∂u∂z_prefactor = ∂ρ∂z_loss / ∂u∂z_loss * (0.05/0.3)
        ∂v∂z_prefactor = ∂ρ∂z_loss / ∂v∂z_loss * (0.05/0.3)

        profile_loss = u_prefactor * u_loss + v_prefactor * v_loss + T_prefactor * T_loss + S_prefactor * S_loss + ρ_prefactor * ρ_loss
        gradient_loss = ∂u∂z_prefactor * ∂u∂z_loss + ∂v∂z_prefactor * ∂v∂z_loss + ∂T∂z_prefactor * ∂T∂z_loss + ∂S∂z_prefactor * ∂S∂z_loss + ∂ρ∂z_prefactor * ∂ρ∂z_loss

        gradient_prefactor = profile_loss / gradient_loss

        ∂ρ∂z_prefactor *= gradient_prefactor
        ∂T∂z_prefactor *= gradient_prefactor
        ∂S∂z_prefactor *= gradient_prefactor
        ∂u∂z_prefactor *= gradient_prefactor
        ∂v∂z_prefactor *= gradient_prefactor

        return (ρ=ρ_prefactor, T=T_prefactor, S=S_prefactor, u=u_prefactor, v=v_prefactor, ∂ρ∂z=∂ρ∂z_prefactor, ∂T∂z=∂T∂z_prefactor, ∂S∂z=∂S∂z_prefactor, ∂u∂z=∂u∂z_prefactor, ∂v∂z=∂v∂z_prefactor)
    end

    losses_prefactor = compute_loss_prefactor(ps_NN)

    function loss_NDE(p)
        preds = predict_NDE(p)

        us = [@view(pred[1:coarse_size, :]) for pred in preds]
        vs = [@view(pred[coarse_size+1:2*coarse_size, :]) for pred in preds]
        Ts = [@view(pred[2*coarse_size+1:3*coarse_size, :]) for pred in preds]
        Ss = [@view(pred[3*coarse_size+1:4*coarse_size, :]) for pred in preds]
        ρs = [param.scaling.ρ.(TEOS10.ρ.(inv(param.scaling.T).(T), inv(param.scaling.S).(S), 0, Ref(eos))) for (T, S, param) in zip(Ts, Ss, params)]

        ∂u∂zs = [hcat([param.scaling.∂u∂z.(param.Dᶠ * inv(param.scaling.u).(@view(u[:, i]))) for i in axes(u, 2)]...) for (param, u) in zip(params, us)]
        ∂v∂zs = [hcat([param.scaling.∂v∂z.(param.Dᶠ * inv(param.scaling.v).(@view(v[:, i]))) for i in axes(v, 2)]...) for (param, v) in zip(params, vs)]
        ∂T∂zs = [hcat([param.scaling.∂T∂z.(param.Dᶠ * inv(param.scaling.T).(@view(T[:, i]))) for i in axes(T, 2)]...) for (param, T) in zip(params, Ts)]
        ∂S∂zs = [hcat([param.scaling.∂S∂z.(param.Dᶠ * inv(param.scaling.S).(@view(S[:, i]))) for i in axes(S, 2)]...) for (param, S) in zip(params, Ss)]
        ∂ρ∂zs = [hcat([param.scaling.∂ρ∂z.(param.Dᶠ * inv(param.scaling.ρ).(@view(ρ[:, i]))) for i in axes(ρ, 2)]...) for (param, ρ) in zip(params, ρs)]

        u_loss = losses_prefactor.u * mean(mean.([(data.profile.u.scaled .- u).^2 for (data, u) in zip(train_data.data, us)]))
        v_loss = losses_prefactor.v * mean(mean.([(data.profile.v.scaled .- v).^2 for (data, v) in zip(train_data.data, vs)]))
        T_loss = losses_prefactor.T * mean(mean.([(data.profile.T.scaled .- T).^2 for (data, T) in zip(train_data.data, Ts)]))
        S_loss = losses_prefactor.S * mean(mean.([(data.profile.S.scaled .- S).^2 for (data, S) in zip(train_data.data, Ss)]))
        ρ_loss = losses_prefactor.ρ * mean(mean.([(data.profile.ρ.scaled .- ρ).^2 for (data, ρ) in zip(train_data.data, ρs)]))
        ∂u∂z_loss = losses_prefactor.∂u∂z * mean(mean.([(data.profile.∂u∂z.scaled .- ∂u∂z).^2 for (data, ∂u∂z) in zip(train_data.data, ∂u∂zs)]))
        ∂v∂z_loss = losses_prefactor.∂v∂z * mean(mean.([(data.profile.∂v∂z.scaled .- ∂v∂z).^2 for (data, ∂v∂z) in zip(train_data.data, ∂v∂zs)]))
        ∂T∂z_loss = losses_prefactor.∂T∂z * mean(mean.([(data.profile.∂T∂z.scaled .- ∂T∂z).^2 for (data, ∂T∂z) in zip(train_data.data, ∂T∂zs)]))
        ∂S∂z_loss = losses_prefactor.∂S∂z * mean(mean.([(data.profile.∂S∂z.scaled .- ∂S∂z).^2 for (data, ∂S∂z) in zip(train_data.data, ∂S∂zs)]))
        ∂ρ∂z_loss = losses_prefactor.∂ρ∂z * mean(mean.([(data.profile.∂ρ∂z.scaled .- ∂ρ∂z).^2 for (data, ∂ρ∂z) in zip(train_data.data, ∂ρ∂zs)]))

        loss = u_loss + v_loss + T_loss + S_loss + ρ_loss + ∂u∂z_loss + ∂v∂z_loss + ∂T∂z_loss + ∂S∂z_loss + ∂ρ∂z_loss

        individual_loss = (u=u_loss, v=v_loss, T=T_loss, S=S_loss, ρ=ρ_loss, ∂u∂z=∂u∂z_loss, ∂v∂z=∂v∂z_loss, ∂T∂z=∂T∂z_loss, ∂S∂z=∂S∂z_loss, ∂ρ∂z=∂ρ∂z_loss)
        return loss, preds, individual_loss
    end

    iter = 0

    losses = zeros(maxiter+1)
    u_losses = zeros(maxiter+1)
    v_losses = zeros(maxiter+1)
    T_losses = zeros(maxiter+1)
    S_losses = zeros(maxiter+1)
    ρ_losses = zeros(maxiter+1)
    ∂u∂z_losses = zeros(maxiter+1)
    ∂v∂z_losses = zeros(maxiter+1)
    ∂T∂z_losses = zeros(maxiter+1)
    ∂S∂z_losses = zeros(maxiter+1)
    ∂ρ∂z_losses = zeros(maxiter+1)

    wall_clock = [time_ns()]

    callback = function (p, l, pred, ind_loss)
        @printf("%s, Δt %s, iter %d/%d, loss total %6.10e, u %6.5e, v %6.5e, T %6.5e, S %6.5e, ρ %6.5e, ∂u∂z %6.5e, ∂v∂z %6.5e, ∂T∂z %6.5e, ∂S∂z %6.5e, ∂ρ∂z %6.5e,\n",
                Dates.now(), prettytime(1e-9 * (time_ns() - wall_clock[1])), iter, maxiter, l, 
                ind_loss.u, ind_loss.v, ind_loss.T, ind_loss.S, ind_loss.ρ, 
                ind_loss.∂u∂z, ind_loss.∂v∂z, ind_loss.∂T∂z, ind_loss.∂S∂z, ind_loss.∂ρ∂z)
        losses[iter+1] = l
        u_losses[iter+1] = ind_loss.u
        v_losses[iter+1] = ind_loss.v
        T_losses[iter+1] = ind_loss.T
        S_losses[iter+1] = ind_loss.S
        ρ_losses[iter+1] = ind_loss.ρ
        ∂u∂z_losses[iter+1] = ind_loss.∂u∂z
        ∂v∂z_losses[iter+1] = ind_loss.∂v∂z
        ∂T∂z_losses[iter+1] = ind_loss.∂T∂z
        ∂S∂z_losses[iter+1] = ind_loss.∂S∂z
        ∂ρ∂z_losses[iter+1] = ind_loss.∂ρ∂z

        iter += 1
        wall_clock[1] = time_ns()
        return false
    end

    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_NDE(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, ps_NN)

    res = Optimization.solve(optprob, optimizer, callback=callback, maxiters=maxiter)

    sols_posttraining = predict_NDE_posttraining(res.u)

    uw_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    vw_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    wT_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    wS_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    νs_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    κs_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    Ri_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    Ri_truth = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]

    for (i, sol) in enumerate(sols_posttraining)
        for t in eachindex(sol.t)
            uw, vw, wT, wS = predict_total_flux_dimensional(sol[:, t], res.u, params[i], st_NN)
            Ris = calculate_Ri(inv(params[i].scaling.u).(sol[1:coarse_size, t]), 
                               inv(params[i].scaling.v).(sol[coarse_size+1:2*coarse_size, t]), 
                               inv(params[i].scaling.T).(sol[2*coarse_size+1:3*coarse_size, t]), 
                               inv(params[i].scaling.S).(sol[3*coarse_size+1:4*coarse_size, t]), 
                               params[i].Dᶠ, params[i].g, eos.reference_density, clamp_lims=Ri_clamp_lims)

            Ris_truth = calculate_Ri(train_data_plot.data[i].profile.u.unscaled[:, t], 
                                     train_data_plot.data[i].profile.v.unscaled[:, t], 
                                     train_data_plot.data[i].profile.T.unscaled[:, t], 
                                     train_data_plot.data[i].profile.S.unscaled[:, t], 
                                     params[i].Dᶠ, params[i].g, eos.reference_density, clamp_lims=Ri_clamp_lims)
            νs, κs = predict_diffusivities(Ris, res.u, st_NN)

            uw_posttraining[i][:, t] .= uw
            vw_posttraining[i][:, t] .= vw
            wT_posttraining[i][:, t] .= wT
            wS_posttraining[i][:, t] .= wS
            νs_posttraining[i][:, t] .= νs
            κs_posttraining[i][:, t] .= κs
            Ri_posttraining[i][:, t] .= Ris
            Ri_truth[i][:, t] .= Ris_truth
        end
    end

    flux_posttraining = (uw=uw_posttraining, vw=vw_posttraining, wT=wT_posttraining, wS=wS_posttraining)
    losses = (total=losses, u=u_losses, v=v_losses, T=T_losses, S=S_losses, ρ=ρ_losses, ∂u∂z=∂u∂z_losses, ∂v∂z=∂v∂z_losses, ∂T∂z=∂T∂z_losses, ∂S∂z=∂S∂z_losses, ∂ρ∂z=∂ρ∂z_losses)
    diffusivities_posttraining = (ν=νs_posttraining, κ=κs_posttraining, Ri=Ri_posttraining, Ri_truth=Ri_truth)

    return res, loss_NDE(res.u), sols_posttraining, flux_posttraining, losses, diffusivities_posttraining
end

function plot_loss(losses, FILE_DIR; epoch=1)
    colors = distinguishable_colors(10, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    fig = Figure(size=(1000, 600))
    axtotalloss = CairoMakie.Axis(fig[1, 1], title="Total Loss", xlabel="Iterations", ylabel="Loss", yscale=log10)
    axindividualloss = CairoMakie.Axis(fig[1, 2], title="Individual Loss", xlabel="Iterations", ylabel="Loss", yscale=log10)

    lines!(axtotalloss, losses.total, label="Total Loss")

    lines!(axindividualloss, losses.u, label="u", color=colors[1])
    lines!(axindividualloss, losses.v, label="v", color=colors[2])
    lines!(axindividualloss, losses.T, label="T", color=colors[3])
    lines!(axindividualloss, losses.S, label="S", color=colors[4])
    lines!(axindividualloss, losses.ρ, label="ρ", color=colors[5])
    lines!(axindividualloss, losses.∂u∂z, label="∂u∂z", color=colors[6])
    lines!(axindividualloss, losses.∂v∂z, label="∂v∂z", color=colors[7])
    lines!(axindividualloss, losses.∂T∂z, label="∂T∂z", color=colors[8])
    lines!(axindividualloss, losses.∂S∂z, label="∂S∂z", color=colors[9])
    lines!(axindividualloss, losses.∂ρ∂z, label="∂ρ∂z", color=colors[10])

    axislegend(axindividualloss, position=:rt)
    save("$(FILE_DIR)/losses_epoch$(epoch).png", fig, px_per_unit=8)
end

function animate_data(train_data, sols, fluxes, diffusivities, index, FILE_DIR; coarse_size=32, epoch=1)
    fig = Figure(size=(1920, 1080))
    axu = CairoMakie.Axis(fig[1, 1], title="u", xlabel="u (m s⁻¹)", ylabel="z (m)")
    axv = CairoMakie.Axis(fig[1, 2], title="v", xlabel="v (m s⁻¹)", ylabel="z (m)")
    axT = CairoMakie.Axis(fig[1, 3], title="T", xlabel="T (°C)", ylabel="z (m)")
    axS = CairoMakie.Axis(fig[1, 4], title="S", xlabel="S (g kg⁻¹)", ylabel="z (m)")
    axρ = CairoMakie.Axis(fig[1, 5], title="ρ", xlabel="ρ (kg m⁻³)", ylabel="z (m)")
    axuw = CairoMakie.Axis(fig[2, 1], title="uw", xlabel="uw (m² s⁻²)", ylabel="z (m)")
    axvw = CairoMakie.Axis(fig[2, 2], title="vw", xlabel="vw (m² s⁻²)", ylabel="z (m)")
    axwT = CairoMakie.Axis(fig[2, 3], title="wT", xlabel="wT (m s⁻¹ °C)", ylabel="z (m)")
    axwS = CairoMakie.Axis(fig[2, 4], title="wS", xlabel="wS (m s⁻¹ g kg⁻¹)", ylabel="z (m)")
    axRi = CairoMakie.Axis(fig[2, 5], title="Ri", xlabel="Ri", ylabel="z (m)")
    axdiffusivity = CairoMakie.Axis(fig[2, 6], title="Diffusivity", xlabel="Diffusivity (m² s⁻¹)", ylabel="z (m)")

    n = Observable(1)
    zC = train_data.data[index].metadata["zC"]
    zF = train_data.data[index].metadata["zF"]

    u_NDE = inv(train_data.scaling.u).(sols[index][1:coarse_size, :])
    v_NDE = inv(train_data.scaling.v).(sols[index][coarse_size+1:2*coarse_size, :])
    T_NDE = inv(train_data.scaling.T).(sols[index][2*coarse_size+1:3*coarse_size, :])
    S_NDE = inv(train_data.scaling.S).(sols[index][3*coarse_size+1:4*coarse_size, :])
    ρ_NDE = TEOS10.ρ.(T_NDE, S_NDE, 0, Ref(TEOS10EquationOfState()))

    uw_NDE = fluxes.uw[index]
    vw_NDE = fluxes.vw[index]
    wT_NDE = fluxes.wT[index]
    wS_NDE = fluxes.wS[index]

    ulim = (find_min(u_NDE, train_data.data[index].profile.u.unscaled), find_max(u_NDE, train_data.data[index].profile.u.unscaled))
    vlim = (find_min(v_NDE, train_data.data[index].profile.v.unscaled), find_max(v_NDE, train_data.data[index].profile.v.unscaled))
    Tlim = (find_min(T_NDE, train_data.data[index].profile.T.unscaled), find_max(T_NDE, train_data.data[index].profile.T.unscaled))
    Slim = (find_min(S_NDE, train_data.data[index].profile.S.unscaled), find_max(S_NDE, train_data.data[index].profile.S.unscaled))
    ρlim = (find_min(ρ_NDE, train_data.data[index].profile.ρ.unscaled), find_max(ρ_NDE, train_data.data[index].profile.ρ.unscaled))

    uwlim = (find_min(uw_NDE, train_data.data[index].flux.uw.column.unscaled), find_max(uw_NDE, train_data.data[index].flux.uw.column.unscaled))
    vwlim = (find_min(vw_NDE, train_data.data[index].flux.vw.column.unscaled), find_max(vw_NDE, train_data.data[index].flux.vw.column.unscaled))
    wTlim = (find_min(wT_NDE, train_data.data[index].flux.wT.column.unscaled), find_max(wT_NDE, train_data.data[index].flux.wT.column.unscaled))
    wSlim = (find_min(wS_NDE, train_data.data[index].flux.wS.column.unscaled), find_max(wS_NDE, train_data.data[index].flux.wS.column.unscaled))

    Rilim = (find_min(diffusivities.Ri[index], diffusivities.Ri_truth[index]), find_max(diffusivities.Ri[index], diffusivities.Ri_truth[index]))
    diffusivitylim = (find_min(diffusivities.ν[index], diffusivities.κ[index]), find_max(diffusivities.ν[index], diffusivities.κ[index]))

    u_truthₙ = @lift train_data.data[index].profile.u.unscaled[:, $n]
    v_truthₙ = @lift train_data.data[index].profile.v.unscaled[:, $n]
    T_truthₙ = @lift train_data.data[index].profile.T.unscaled[:, $n]
    S_truthₙ = @lift train_data.data[index].profile.S.unscaled[:, $n]
    ρ_truthₙ = @lift train_data.data[index].profile.ρ.unscaled[:, $n]

    uw_truthₙ = @lift train_data.data[index].flux.uw.column.unscaled[:, $n]
    vw_truthₙ = @lift train_data.data[index].flux.vw.column.unscaled[:, $n]
    wT_truthₙ = @lift train_data.data[index].flux.wT.column.unscaled[:, $n]
    wS_truthₙ = @lift train_data.data[index].flux.wS.column.unscaled[:, $n]

    u_NDEₙ = @lift u_NDE[:, $n]
    v_NDEₙ = @lift v_NDE[:, $n]
    T_NDEₙ = @lift T_NDE[:, $n]
    S_NDEₙ = @lift S_NDE[:, $n]
    ρ_NDEₙ = @lift ρ_NDE[:, $n]

    uw_NDEₙ = @lift uw_NDE[:, $n]
    vw_NDEₙ = @lift vw_NDE[:, $n]
    wT_NDEₙ = @lift wT_NDE[:, $n]
    wS_NDEₙ = @lift wS_NDE[:, $n]

    Ri_truthₙ = @lift diffusivities.Ri_truth[index][:, $n]
    Riₙ = @lift diffusivities.Ri[index][:, $n]
    νₙ = @lift diffusivities.ν[index][:, $n]
    κₙ = @lift diffusivities.κ[index][:, $n]

    Qᵁ = train_data.data[index].metadata["momentum_flux"]
    Qᵀ = train_data.data[index].metadata["temperature_flux"]
    Qˢ = train_data.data[index].metadata["salinity_flux"]
    f = train_data.data[index].metadata["coriolis_parameter"]
    times = train_data.data[index].metadata["original_times"]
    Nt = length(times)

    time_str = @lift "Qᵁ = $(Qᵁ) m² s⁻², Qᵀ = $(Qᵀ) m s⁻¹ °C, Qˢ = $(Qˢ) m s⁻¹ g kg⁻¹, f = $(f) s⁻¹, Time = $(round(times[$n]/24/60^2, digits=3)) days"

    lines!(axu, u_truthₙ, zC, label="Truth")
    lines!(axu, u_NDEₙ, zC, label="NDE")

    lines!(axv, v_truthₙ, zC, label="Truth")
    lines!(axv, v_NDEₙ, zC, label="NDE")

    lines!(axT, T_truthₙ, zC, label="Truth")
    lines!(axT, T_NDEₙ, zC, label="NDE")

    lines!(axS, S_truthₙ, zC, label="Truth")
    lines!(axS, S_NDEₙ, zC, label="NDE")

    lines!(axρ, ρ_truthₙ, zC, label="Truth")
    lines!(axρ, ρ_NDEₙ, zC, label="NDE")

    lines!(axuw, uw_truthₙ, zF, label="Truth")
    lines!(axuw, uw_NDEₙ, zF, label="NDE")

    lines!(axvw, vw_truthₙ, zF, label="Truth")
    lines!(axvw, vw_NDEₙ, zF, label="NDE")

    lines!(axwT, wT_truthₙ, zF, label="Truth")
    lines!(axwT, wT_NDEₙ, zF, label="NDE")

    lines!(axwS, wS_truthₙ, zF, label="Truth")
    lines!(axwS, wS_NDEₙ, zF, label="NDE")

    lines!(axRi, Ri_truthₙ, zF, label="Truth")
    lines!(axRi, Riₙ, zF, label="NDE")

    lines!(axdiffusivity, νₙ, zF, label="ν")
    lines!(axdiffusivity, κₙ, zF, label="κ")

    axislegend(axu, position=:rb)
    axislegend(axdiffusivity, position=:rb)

    Label(fig[0, :], time_str, font=:bold, tellwidth=false)

    xlims!(axu, ulim)
    xlims!(axv, vlim)
    xlims!(axT, Tlim)
    xlims!(axS, Slim)
    xlims!(axρ, ρlim)
    xlims!(axuw, uwlim)
    xlims!(axvw, vwlim)
    xlims!(axwT, wTlim)
    xlims!(axwS, wSlim)
    xlims!(axRi, Rilim)
    xlims!(axdiffusivity, diffusivitylim)

    CairoMakie.record(fig, "$(FILE_DIR)/training_$(index)_epoch$(epoch).mp4", 1:Nt, framerate=15) do nn
        # xlims!(axu, nothing, nothing)
        # xlims!(axv, nothing, nothing)
        # xlims!(axT, nothing, nothing)
        # xlims!(axS, nothing, nothing)
        # xlims!(axρ, nothing, nothing)
        # xlims!(axuw, nothing, nothing)
        # xlims!(axvw, nothing, nothing)
        # xlims!(axwT, nothing, nothing)
        # xlims!(axwS, nothing, nothing)
        n[] = nn
    end
end

epoch = 1
res, loss, sols, fluxes, losses, diffusivities = train_NDE(train_data, train_data_plot, NN, ps_NN, st_NN, Ri_clamp_lims=(-Inf, Inf),
                                                           maxiter=500, optimizer=OptimizationOptimisers.ADAM(0.005), solver=ROCK2())

jldsave("$(FILE_DIR)/training_results_$(epoch).jld2"; res, loss, sols, fluxes, losses, diffusivities)
plot_loss(losses, FILE_DIR, epoch=epoch)
for i in eachindex(field_datasets)
    animate_data(train_data_plot, sols, fluxes, diffusivities, i, FILE_DIR, epoch=epoch)
end

epoch += 1

res, loss, sols, fluxes, losses, diffusivities = train_NDE(train_data, train_data_plot, NN, res.u, st_NN, Ri_clamp_lims=(-Inf, Inf),
                                                           maxiter=500, optimizer=OptimizationOptimisers.ADAM(0.001), solver=ROCK2())
@info "Training complete"

jldsave("$(FILE_DIR)/training_results_$(epoch).jld2"; res, loss, sols, fluxes, losses)
plot_loss(losses, FILE_DIR, epoch=epoch)
for i in eachindex(field_datasets)
    animate_data(train_data_plot, sols, fluxes, diffusivities, i, FILE_DIR, epoch=epoch)
end
@info "Animation complete"

epoch += 1

res, loss, sols, fluxes, losses, diffusivities = train_NDE(train_data, train_data_plot, NN, res.u, st_NN, Ri_clamp_lims=(-Inf, Inf),
                                                           maxiter=500, optimizer=OptimizationOptimisers.ADAM(0.0005), solver=ROCK2())
@info "Training complete"

jldsave("$(FILE_DIR)/training_results_$(epoch).jld2"; res, loss, sols, fluxes, losses)
plot_loss(losses, FILE_DIR, epoch=epoch)
for i in eachindex(field_datasets)
    animate_data(train_data_plot, sols, fluxes, diffusivities, i, FILE_DIR, epoch=epoch)
end
@info "Animation complete"
#%%