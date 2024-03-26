using SaltyOceanParameterizations
using SaltyOceanParameterizations.DataWrangling
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
import SeawaterPolynomials.TEOS10: s, ΔS, Sₐᵤ
s(Sᴬ) = Sᴬ + ΔS >= 0 ? √((Sᴬ + ΔS) / Sₐᵤ) : NaN

function find_min(a...)
    return minimum(minimum.([a...]))
end
  
function find_max(a...)
    return maximum(maximum.([a...]))
end

FILE_DIR = "./training_output/freeconvection_NN_leakyrelu_1024_convective_adjustment_ROCK2_reltol1e-5_test"

mkpath(FILE_DIR)
@info "$(FILE_DIR)"

LES_FILE_DIRS = [
    "./LES_training/linearTS_dTdz_0.0013_dSdz_-0.0014_QU_0.0_QT_3.0e-5_QS_-3.0e-5_T_4.3_S_33.5_f_-0.00012_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_0.0_QT_0.0003_QS_-3.0e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0003_QS_-3.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_-0.025_dSdz_-0.0045_QU_0.0_QT_-0.0003_QS_-3.0e-5_T_-3.6_S_33.9_f_-0.000125_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
]

field_datasets = [FieldDataset(FILE_DIR, backend=OnDisk()) for FILE_DIR in LES_FILE_DIRS]

full_timeframes = [25:length(data["ubar"].times) for data in field_datasets]
timeframes = [25:5:length(data["ubar"].times) for data in field_datasets]
train_data = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, timeframes)
coarse_size = 32

train_data_plot = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, full_timeframes)

rng = Random.default_rng(123)

NN = Chain(Dense(99, 1024, leakyrelu), Dense(1024, 62))

ps, st = Lux.setup(rng, NN)

ps = ps |> ComponentArray .|> Float64
ps .= glorot_uniform(rng, Float64, length(ps))

ps .*= 1e-3

NNs = (; NDE=NN)
ps_training = ComponentArray(NDE=ps)
st_NN = (; NDE=st)

function convective_adjustment(dρdz)
    return ifelse(dρdz > 0, 0.2, 1e-5)
end

function train_NDE(train_data, train_data_plot, NNs, ps_training, st_NN, rng; 
                   coarse_size=32, dev=cpu_device(), maxiter=10, optimizer=OptimizationOptimisers.ADAM(0.001), solver=ROCK2(), Ri_clamp_lims=(-Inf, Inf))
    train_data = train_data |> dev
    x₀s = [vcat(data.profile.T.scaled[:, 1], data.profile.S.scaled[:, 1]) for data in train_data.data] |> dev
    eos = TEOS10EquationOfState()

    params = [(                   f = data.metadata["coriolis_parameter"],
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

    function predict_residual_flux(T_hat, S_hat, ρ_hat, p, params, st)
        x′ = vcat(T_hat, S_hat, ρ_hat, params.wT.scaled.top, params.wS.scaled.top, params.f_scaled)
        
        interior_size = coarse_size-1
        NN_pred = first(NNs.NDE(x′, p.NDE, st.NDE))

        wT = vcat(0, NN_pred[1:interior_size], 0)
        wS = vcat(0, NN_pred[interior_size+1:2*interior_size], 0)

        return wT, wS
    end

    function predict_residual_flux_dimensional(T_hat, S_hat, ρ_hat, p, params, st)
        wT_hat, wS_hat = predict_residual_flux(T_hat, S_hat, ρ_hat, p, params, st)
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

    function predict_diffusivities(∂ρ∂z)
        κs = convective_adjustment.(∂ρ∂z)
        return κs
    end

    function predict_diffusive_flux(ρ, T_hat, S_hat, p, params, st)
        ∂ρ∂z = params.Dᶠ * ρ
        κs = predict_diffusivities(∂ρ∂z)

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

    function calculate_scaled_profile_gradients(T, S, ρ, params)
        Dᶠ = params.Dᶠ

        ∂T∂z_hat = params.scaling.∂T∂z.(Dᶠ * T)
        ∂S∂z_hat = params.scaling.∂S∂z.(Dᶠ * S)
        ∂ρ∂z_hat = params.scaling.∂ρ∂z.(Dᶠ * ρ)

        return ∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat
    end

    function NDE(x, p, t, params, st)
        f = params.f
        Dᶜ_hat = params.Dᶜ_hat
        scaling = params.scaling
        τ, H = params.τ, params.H

        T_hat, S_hat = get_scaled_profiles(x, params)
        T, S = unscale_profiles(T_hat, S_hat, params)
        
        ρ = calculate_unscaled_density(T, S)
        ρ_hat = params.scaling.ρ.(ρ)

        wT_residual, wS_residual = predict_residual_flux(T_hat, S_hat, ρ_hat, p, params, st)
        wT_boundary, wS_boundary = predict_boundary_flux(params)
        wT_diffusive, wS_diffusive = predict_diffusive_flux(ρ, T_hat, S_hat, p, params, st)

        dT = -τ / H^2 .* (Dᶜ_hat * wT_diffusive) .- τ / H * scaling.wT.σ / scaling.T.σ .* (Dᶜ_hat * (wT_boundary .+ wT_residual))
        dS = -τ / H^2 .* (Dᶜ_hat * wS_diffusive) .- τ / H * scaling.wS.σ / scaling.S.σ .* (Dᶜ_hat * (wS_boundary .+ wS_residual))

        return vcat(dT, dS)
    end

    function predict_NDE(p)
        probs = [ODEProblem((x, p′, t) -> NDE(x, p′, t, param, st_NN), x₀, (param.scaled_time[1], param.scaled_time[end]), p) for (x₀, param) in zip(x₀s, params)]
        sols = [Array(solve(prob, solver, saveat=param.scaled_time, reltol=1e-5)) for (param, prob) in zip(params, probs)]
        return sols
    end

    function predict_NDE_posttraining(p)
        probs = [ODEProblem((x, p′, t) -> NDE(x, p′, t, param, st_NN), x₀, (param.scaled_original_time[1], param.scaled_original_time[end]), p) for (x₀, param) in zip(x₀s, params)]
        sols = [solve(prob, solver, saveat=param.scaled_original_time, reltol=1e-5) for (param, prob) in zip(params, probs)]
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

        return (T=T_loss, S=S_loss, ρ=ρ_loss, ∂T∂z=∂T∂z_loss, ∂S∂z=∂S∂z_loss, ∂ρ∂z=∂ρ∂z_loss)
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

        return (ρ=ρ_prefactor, T=T_prefactor, S=S_prefactor, ∂ρ∂z=∂ρ∂z_prefactor, ∂T∂z=∂T∂z_prefactor, ∂S∂z=∂S∂z_prefactor)
    end

    function compute_loss_prefactor(p)
        T, S, ρ, ∂T∂z, ∂S∂z, ∂ρ∂z, _ = predict_scaled_profiles(p)
        losses = predict_losses(T, S, ρ, ∂T∂z, ∂S∂z, ∂ρ∂z)
        return compute_loss_prefactor(losses...)
    end

    @info "Computing prefactor for losses"

    losses_prefactor = compute_loss_prefactor(ps_training)

    function loss_NDE(p)
        T, S, ρ, ∂T∂z, ∂S∂z, ∂ρ∂z, preds = predict_scaled_profiles(p)
        individual_loss = predict_losses(T, S, ρ, ∂T∂z, ∂S∂z, ∂ρ∂z, losses_prefactor)
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

    callback = function (p, l, pred, ind_loss)
        @printf("%s, Δt %s, iter %d/%d, loss total %6.10e, T %6.5e, S %6.5e, ρ %6.5e, ∂T∂z %6.5e, ∂S∂z %6.5e, ∂ρ∂z %6.5e,\n",
                Dates.now(), prettytime(1e-9 * (time_ns() - wall_clock[1])), iter, maxiter, l, 
                ind_loss.T, ind_loss.S, ind_loss.ρ, 
                ind_loss.∂T∂z, ind_loss.∂S∂z, ind_loss.∂ρ∂z)
        losses[iter+1] = l
        T_losses[iter+1] = ind_loss.T
        S_losses[iter+1] = ind_loss.S
        ρ_losses[iter+1] = ind_loss.ρ
        ∂T∂z_losses[iter+1] = ind_loss.∂T∂z
        ∂S∂z_losses[iter+1] = ind_loss.∂S∂z
        ∂ρ∂z_losses[iter+1] = ind_loss.∂ρ∂z

        iter += 1
        wall_clock[1] = time_ns()
        return false
    end

    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_NDE(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, ps_training)

    @info "Training NDE"
    res = Optimization.solve(optprob, optimizer, callback=callback, maxiters=maxiter)

    sols_posttraining = predict_NDE_posttraining(res.u)

    uw_diffusive_boundary_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    vw_diffusive_boundary_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    wT_diffusive_boundary_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    wS_diffusive_boundary_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]

    uw_residual_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    vw_residual_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    wT_residual_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    wS_residual_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]

    νs_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    κs_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    Ri_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    Ri_truth = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]

    for (i, sol) in enumerate(sols_posttraining)
        for t in eachindex(sol.t)
            T_hat, S_hat = get_scaled_profiles(sol[:, t], params[i])
            T, S = unscale_profiles(T_hat, S_hat, params[i])

            ρ = calculate_unscaled_density(T, S)
            ρ_hat = params[i].scaling.ρ.(ρ)

            wT_diffusive_boundary, wS_diffusive_boundary = predict_diffusive_boundary_flux_dimensional(ρ, T_hat, S_hat, res.u, params[i], st_NN)
            wT_residual, wS_residual = predict_residual_flux_dimensional(T_hat, S_hat, ρ_hat, res.u, params[i], st_NN)

            ∂ρ∂z = params[i].Dᶠ * ρ
            κs = predict_diffusivities(∂ρ∂z)

            wT_residual_posttraining[i][:, t] .= wT_residual
            wS_residual_posttraining[i][:, t] .= wS_residual

            wT_diffusive_boundary_posttraining[i][:, t] .= wT_diffusive_boundary
            wS_diffusive_boundary_posttraining[i][:, t] .= wS_diffusive_boundary

            κs_posttraining[i][:, t] .= κs

        end
    end
    
    uw_total_posttraining = uw_residual_posttraining .+ uw_diffusive_boundary_posttraining
    vw_total_posttraining = vw_residual_posttraining .+ vw_diffusive_boundary_posttraining
    wT_total_posttraining = wT_residual_posttraining .+ wT_diffusive_boundary_posttraining
    wS_total_posttraining = wS_residual_posttraining .+ wS_diffusive_boundary_posttraining

    flux_posttraining = (uw = (diffusive_boundary=uw_diffusive_boundary_posttraining, residual=uw_residual_posttraining, total=uw_total_posttraining),
                         vw = (diffusive_boundary=vw_diffusive_boundary_posttraining, residual=vw_residual_posttraining, total=vw_total_posttraining),
                         wT = (diffusive_boundary=wT_diffusive_boundary_posttraining, residual=wT_residual_posttraining, total=wT_total_posttraining),
                         wS = (diffusive_boundary=wS_diffusive_boundary_posttraining, residual=wS_residual_posttraining, total=wS_total_posttraining))
                         
    diffusivities_posttraining = (ν=νs_posttraining, κ=κs_posttraining, Ri=Ri_posttraining, Ri_truth=Ri_truth)

    losses = (total=losses, T=T_losses, S=S_losses, ρ=ρ_losses, ∂T∂z=∂T∂z_losses, ∂S∂z=∂S∂z_losses, ∂ρ∂z=∂ρ∂z_losses)

    return res, loss_NDE(res.u), sols_posttraining, flux_posttraining, losses, diffusivities_posttraining
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

    u_NDE = zeros(coarse_size, length(train_data.data[index].times))
    v_NDE = zeros(coarse_size, length(train_data.data[index].times))
    T_NDE = inv(train_data.scaling.T).(sols[index][1:coarse_size, :])
    S_NDE = inv(train_data.scaling.S).(sols[index][coarse_size+1:2*coarse_size, :])
    ρ_NDE = TEOS10.ρ.(T_NDE, S_NDE, 0, Ref(TEOS10EquationOfState()))

    uw_residual = fluxes.uw.residual[index]
    vw_residual = fluxes.vw.residual[index]
    wT_residual = fluxes.wT.residual[index]
    wS_residual = fluxes.wS.residual[index]

    uw_diffusive_boundary = fluxes.uw.diffusive_boundary[index]
    vw_diffusive_boundary = fluxes.vw.diffusive_boundary[index]
    wT_diffusive_boundary = fluxes.wT.diffusive_boundary[index]
    wS_diffusive_boundary = fluxes.wS.diffusive_boundary[index]

    uw_total = fluxes.uw.total[index]
    vw_total = fluxes.vw.total[index]
    wT_total = fluxes.wT.total[index]
    wS_total = fluxes.wS.total[index]

    ulim = (find_min(u_NDE, train_data.data[index].profile.u.unscaled), find_max(u_NDE, train_data.data[index].profile.u.unscaled))
    vlim = (find_min(v_NDE, train_data.data[index].profile.v.unscaled), find_max(v_NDE, train_data.data[index].profile.v.unscaled))
    Tlim = (find_min(T_NDE, train_data.data[index].profile.T.unscaled), find_max(T_NDE, train_data.data[index].profile.T.unscaled))
    Slim = (find_min(S_NDE, train_data.data[index].profile.S.unscaled), find_max(S_NDE, train_data.data[index].profile.S.unscaled))
    ρlim = (find_min(ρ_NDE, train_data.data[index].profile.ρ.unscaled), find_max(ρ_NDE, train_data.data[index].profile.ρ.unscaled))

    uwlim = (find_min(uw_residual, uw_diffusive_boundary, uw_total, train_data.data[index].flux.uw.column.unscaled), 
             find_max(uw_residual, uw_diffusive_boundary, uw_total, train_data.data[index].flux.uw.column.unscaled))
    vwlim = (find_min(vw_residual, vw_diffusive_boundary, vw_total, train_data.data[index].flux.vw.column.unscaled),
             find_max(vw_residual, vw_diffusive_boundary, vw_total, train_data.data[index].flux.vw.column.unscaled))
    wTlim = (find_min(wT_residual, wT_diffusive_boundary, wT_total, train_data.data[index].flux.wT.column.unscaled),
             find_max(wT_residual, wT_diffusive_boundary, wT_total, train_data.data[index].flux.wT.column.unscaled))
    wSlim = (find_min(wS_residual, wS_diffusive_boundary, wS_total, train_data.data[index].flux.wS.column.unscaled),
             find_max(wS_residual, wS_diffusive_boundary, wS_total, train_data.data[index].flux.wS.column.unscaled))

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

    uw_residualₙ = @lift uw_residual[:, $n]
    vw_residualₙ = @lift vw_residual[:, $n]
    wT_residualₙ = @lift wT_residual[:, $n]
    wS_residualₙ = @lift wS_residual[:, $n]

    uw_diffusive_boundaryₙ = @lift uw_diffusive_boundary[:, $n]
    vw_diffusive_boundaryₙ = @lift vw_diffusive_boundary[:, $n]
    wT_diffusive_boundaryₙ = @lift wT_diffusive_boundary[:, $n]
    wS_diffusive_boundaryₙ = @lift wS_diffusive_boundary[:, $n]

    uw_totalₙ = @lift uw_total[:, $n]
    vw_totalₙ = @lift vw_total[:, $n]
    wT_totalₙ = @lift wT_total[:, $n]
    wS_totalₙ = @lift wS_total[:, $n]

    Ri_truthₙ = @lift diffusivities.Ri_truth[index][:, $n]
    Riₙ = @lift diffusivities.Ri[index][:, $n]
    νₙ = @lift diffusivities.ν[index][:, $n]
    κₙ = @lift diffusivities.κ[index][:, $n]

    Qᵁ = train_data.data[index].metadata["momentum_flux"]
    Qᵀ = train_data.data[index].metadata["temperature_flux"]
    Qˢ = train_data.data[index].metadata["salinity_flux"]
    f = train_data.data[index].metadata["coriolis_parameter"]
    times = train_data.data[index].times
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
    lines!(axuw, uw_totalₙ, zF, label="NDE")
    lines!(axuw, uw_residualₙ, zF, label="Residual")
    lines!(axuw, uw_diffusive_boundaryₙ, zF, label="Base closure")

    lines!(axvw, vw_truthₙ, zF, label="Truth")
    lines!(axvw, vw_totalₙ, zF, label="NDE")
    lines!(axvw, vw_residualₙ, zF, label="Residual")
    lines!(axvw, vw_diffusive_boundaryₙ, zF, label="Base closure")

    lines!(axwT, wT_truthₙ, zF, label="Truth")
    lines!(axwT, wT_totalₙ, zF, label="NDE")
    lines!(axwT, wT_residualₙ, zF, label="Residual")
    lines!(axwT, wT_diffusive_boundaryₙ, zF, label="Base closure")

    lines!(axwS, wS_truthₙ, zF, label="Truth")
    lines!(axwS, wS_totalₙ, zF, label="NDE")
    lines!(axwS, wS_residualₙ, zF, label="Residual")
    lines!(axwS, wS_diffusive_boundaryₙ, zF, label="Base closure")

    lines!(axRi, Ri_truthₙ, zF, label="Truth")
    lines!(axRi, Riₙ, zF, label="NDE")

    lines!(axdiffusivity, νₙ, zF, label="ν")
    lines!(axdiffusivity, κₙ, zF, label="κ")

    axislegend(axu, position=:rb)
    axislegend(axuw, position=:rb)
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
    # xlims!(axRi, Rilim)
    xlims!(axdiffusivity, diffusivitylim)

    CairoMakie.record(fig, "$(FILE_DIR)/training_$(index)_epoch$(epoch).mp4", 1:Nt, framerate=15) do nn
        n[] = nn
    end
end

epoch = 1

res, loss, sols, fluxes, losses, diffusivities = train_NDE(train_data, train_data_plot, NNs, ps_training, st_NN, rng, maxiter=1000, solver=ROCK2(), optimizer=OptimizationOptimisers.ADAM(5e-5))

u = res.u
jldsave("$(FILE_DIR)/training_results_$(epoch).jld2"; res, u, loss, sols, fluxes, losses, NNs, st_NN, diffusivities)
plot_loss(losses, FILE_DIR, epoch=epoch)
for i in eachindex(field_datasets)
    animate_data(train_data_plot, sols, fluxes, diffusivities, i, FILE_DIR, epoch=epoch)
end

epoch += 1

res, loss, sols, fluxes, losses, diffusivities = train_NDE(train_data, train_data_plot, NNs, res.u, st_NN, rng, maxiter=1000, solver=ROCK2(), optimizer=OptimizationOptimisers.ADAM(5e-5))
u = res.u
jldsave("$(FILE_DIR)/training_results_$(epoch).jld2"; res, u, loss, sols, fluxes, losses, NNs, st_NN, diffusivities)
plot_loss(losses, FILE_DIR, epoch=epoch)
for i in eachindex(field_datasets)
    animate_data(train_data_plot, sols, fluxes, diffusivities, i, FILE_DIR, epoch=epoch)
end

epoch += 1

res, loss, sols, fluxes, losses, diffusivities = train_NDE(train_data, train_data_plot, NNs, res.u, st_NN, rng, maxiter=1000, solver=ROCK2(), optimizer=OptimizationOptimisers.ADAM(2e-5))

u = res.u
jldsave("$(FILE_DIR)/training_results_$(epoch).jld2"; res, u, loss, sols, fluxes, losses, NNs, st_NN, diffusivities)
plot_loss(losses, FILE_DIR, epoch=epoch)
for i in eachindex(field_datasets)
    animate_data(train_data_plot, sols, fluxes, diffusivities, i, FILE_DIR, epoch=epoch)
end

epoch += 1

res, loss, sols, fluxes, losses, diffusivities = train_NDE(train_data, train_data_plot, NNs, res.u, st_NN, rng, maxiter=1000, solver=ROCK2(), optimizer=OptimizationOptimisers.ADAM(2e-5))

u = res.u
jldsave("$(FILE_DIR)/training_results_$(epoch).jld2"; res, u, loss, sols, fluxes, losses, NNs, st_NN, diffusivities)
plot_loss(losses, FILE_DIR, epoch=epoch)
for i in eachindex(field_datasets)
    animate_data(train_data_plot, sols, fluxes, diffusivities, i, FILE_DIR, epoch=epoch)
end


