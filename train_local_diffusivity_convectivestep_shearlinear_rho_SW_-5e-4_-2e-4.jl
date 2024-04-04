using SaltyOceanParameterizations
using SaltyOceanParameterizations.DataWrangling
using SaltyOceanParameterizations: calculate_Ri, local_Ri_ν_convectivestep_shearlinear, local_Ri_κ_convectivestep_shearlinear
using Oceananigans
using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL, OptimizationOptimisers, Random, SciMLSensitivity, LuxCUDA
using Statistics
using CairoMakie
using SeawaterPolynomials.TEOS10
using Printf
using Dates
using JLD2
import SeawaterPolynomials.TEOS10: s, ΔS, Sₐᵤ
s(Sᴬ) = Sᴬ + ΔS >= 0 ? √((Sᴬ + ΔS) / Sₐᵤ) : NaN

function find_min(a...)
    return minimum(minimum.([a...]))
end
  
function find_max(a...)
    return maximum(maximum.([a...]))
end

FILE_DIR = "./training_output/local_diffusivity_convectivestep_shearlinear_rho_SW_-5e-4_-2e-4"
mkpath(FILE_DIR)

LES_FILE_DIRS = [
    "./LES_training/linearTS_b_dTdz_0.0013_dSdz_-0.0014_QU_-0.0005_QB_0.0_T_4.3_S_33.5_f_-0.00012_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_b_dTdz_0.013_dSdz_0.00075_QU_-0.0005_QB_0.0_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_b_dTdz_0.014_dSdz_0.0021_QU_-0.0005_QB_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_b_dTdz_-0.025_dSdz_-0.0045_QU_-0.0005_QB_0.0_T_-3.6_S_33.9_f_-0.000125_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_b_dTdz_0.0013_dSdz_-0.0014_QU_-0.0002_QB_0.0_T_4.3_S_33.5_f_-0.00012_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_b_dTdz_0.013_dSdz_0.00075_QU_-0.0002_QB_0.0_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_b_dTdz_0.014_dSdz_0.0021_QU_-0.0002_QB_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_b_dTdz_-0.025_dSdz_-0.0045_QU_-0.0002_QB_0.0_T_-3.6_S_33.9_f_-0.000125_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
]

field_datasets = [FieldDataset(FILE_DIR, backend=OnDisk()) for FILE_DIR in LES_FILE_DIRS]

full_timeframes = [25:length(data["ubar"].times) for data in field_datasets]
timeframes = [25:5:length(data["ubar"].times) for data in field_datasets]
train_data = LESDatasetsB(field_datasets, ZeroMeanUnitVarianceScaling, timeframes)
coarse_size = 32

train_data_plot = LESDatasetsB(field_datasets, ZeroMeanUnitVarianceScaling, full_timeframes)

ps = ComponentArray(ν_conv=0.4, ν_shear=6.484e-02, m=-1.583e-01, Pr=1.27)

function optimize_parameters(train_data, train_data_plot, ps; coarse_size=32, dev=cpu_device(), maxiter=10, optimizer=OptimizationOptimisers.ADAM(0.01), solver=DP5(), Ri_clamp_lims=(-Inf, Inf), sensealg=ForwardDiffSensitivity())
    train_data = train_data |> dev
    x₀s = [vcat(data.profile.u.scaled[:, 1], data.profile.v.scaled[:, 1], data.profile.ρ.scaled[:, 1]) for data in train_data.data] |> dev
    eos = TEOS10EquationOfState()

    params = [(                   f = data.metadata["coriolis_parameter"],
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
                                 uw = (scaled = (top=data.flux.uw.surface.scaled, bottom=data.flux.uw.bottom.scaled),
                                       unscaled = (top=data.flux.uw.surface.unscaled, bottom=data.flux.uw.bottom.unscaled)),
                                 vw = (scaled = (top=data.flux.vw.surface.scaled, bottom=data.flux.vw.bottom.scaled),
                                       unscaled = (top=data.flux.vw.surface.unscaled, bottom=data.flux.vw.bottom.unscaled)),
                                 wb = (scaled = (top=data.flux.wb.surface.scaled, bottom=data.flux.wb.bottom.scaled),
                                       unscaled = (top=data.flux.wb.surface.unscaled, bottom=data.flux.wb.bottom.unscaled)),
                                 wρ = (scaled = (top=data.flux.wρ.surface.scaled, bottom=data.flux.wρ.bottom.scaled),
                                       unscaled = (top=data.flux.wρ.surface.unscaled, bottom=data.flux.wρ.bottom.unscaled)),
                            scaling = train_data.scaling
               ) for (data, plot_data) in zip(train_data.data, train_data_plot.data)] |> dev

    function predict_diffusivities(Ris, p)
        νs = local_Ri_ν_convectivestep_shearlinear.(Ris, p.ν_conv, p.ν_shear, p.m)
        κs = local_Ri_κ_convectivestep_shearlinear.(νs, p.Pr)
        return νs, κs
    end

    function predict_diffusive_flux(x, p, params)
        u_hat = x[1:coarse_size]
        v_hat = x[coarse_size+1:2*coarse_size]
        ρ_hat = x[2*coarse_size+1:3*coarse_size]

        u = inv(params.scaling.u).(u_hat)
        v = inv(params.scaling.v).(v_hat)
        ρ = inv(params.scaling.ρ).(ρ_hat)

        Ris = calculate_Ri(u, v, ρ, params.Dᶠ, params.g, eos.reference_density, clamp_lims=Ri_clamp_lims)
        νs, κs = predict_diffusivities(Ris, p)

        ∂u∂z_hat = params.Dᶠ_hat * u_hat
        ∂v∂z_hat = params.Dᶠ_hat * v_hat
        ∂ρ∂z_hat = params.Dᶠ_hat * ρ_hat

        uw_diffusive = -νs .* ∂u∂z_hat
        vw_diffusive = -νs .* ∂v∂z_hat
        wρ_diffusive = -κs .* ∂ρ∂z_hat
        return uw_diffusive, vw_diffusive, wρ_diffusive
    end

    function predict_boundary_flux(params)
        uw_boundary = vcat(fill(params.uw.scaled.bottom, coarse_size), params.uw.scaled.top)
        vw_boundary = vcat(fill(params.vw.scaled.bottom, coarse_size), params.vw.scaled.top)
        wρ_boundary = vcat(fill(params.wρ.scaled.bottom, coarse_size), params.wρ.scaled.top)

        return uw_boundary, vw_boundary, wρ_boundary
    end

    function predict_total_flux_dimensional(x, p, params)
        _uw_diffusive, _vw_diffusive, _wρ_diffusive = predict_diffusive_flux(x, p, params)
        _uw_boundary, _vw_boundary, _wρ_boundary = predict_boundary_flux(params)

        uw_diffusive = params.scaling.u.σ / params.H .* _uw_diffusive
        vw_diffusive = params.scaling.v.σ / params.H .* _vw_diffusive
        wρ_diffusive = params.scaling.ρ.σ / params.H .* _wρ_diffusive

        uw_boundary = inv(params.scaling.uw).(_uw_boundary)
        vw_boundary = inv(params.scaling.vw).(_vw_boundary)
        wρ_boundary = inv(params.scaling.wρ).(_wρ_boundary)

        uw = uw_diffusive .+ uw_boundary
        vw = vw_diffusive .+ vw_boundary
        wρ = wρ_diffusive .+ wρ_boundary

        return uw, vw, wρ
    end

    function DE(x, p, t, params)
        coarse_size = params.coarse_size
        f = params.f
        Dᶜ_hat = params.Dᶜ_hat
        scaling = params.scaling
        τ, H = params.τ, params.H

        u = x[1:coarse_size]
        v = x[coarse_size+1:2*coarse_size]

        uw_diffusive, vw_diffusive, wρ_diffusive = predict_diffusive_flux(x, p, params)
        uw_boundary, vw_boundary, wρ_boundary = predict_boundary_flux(params)

        du = -τ / H^2 .* (Dᶜ_hat * uw_diffusive) .- τ / H * scaling.uw.σ / scaling.u.σ .* (Dᶜ_hat * uw_boundary) .+ f * τ ./ scaling.u.σ .* inv(scaling.v).(v)
        dv = -τ / H^2 .* (Dᶜ_hat * vw_diffusive) .- τ / H * scaling.vw.σ / scaling.v.σ .* (Dᶜ_hat * vw_boundary) .- f * τ ./ scaling.v.σ .* inv(scaling.u).(u)
        dρ = -τ / H^2 .* (Dᶜ_hat * wρ_diffusive) .- τ / H * scaling.wρ.σ / scaling.ρ.σ .* (Dᶜ_hat * wρ_boundary)

        return vcat(du, dv, dρ)
    end

    function predict_DE(p)
        probs = [ODEProblem((x, p′, t) -> DE(x, p′, t, param), x₀, (param.scaled_time[1], param.scaled_time[end]), p) for (x₀, param) in zip(x₀s, params)]
        sols = [Array(solve(prob, solver, saveat=param.scaled_time, reltol=1e-6, sensealg=sensealg)) for (param, prob) in zip(params, probs)]
        return sols
    end

    function predict_DE_posttraining(p)
        probs = [ODEProblem((x, p′, t) -> DE(x, p′, t, param), x₀, (param.scaled_original_time[1], param.scaled_original_time[end]), p) for (x₀, param) in zip(x₀s, params)]
        sols = [solve(prob, solver, saveat=param.scaled_original_time, reltol=1e-6) for (param, prob) in zip(params, probs)]
        return sols
    end

    function compute_loss_prefactor(p)
        preds = predict_DE(p)

        us = [@view(pred[1:coarse_size, :]) for pred in preds]
        vs = [@view(pred[coarse_size+1:2*coarse_size, :]) for pred in preds]
        ρs = [@view(pred[2*coarse_size+1:3*coarse_size, :]) for pred in preds]

        u_loss = mean(mean.([(data.profile.u.scaled .- u).^2 for (data, u) in zip(train_data.data, us)]))
        v_loss = mean(mean.([(data.profile.v.scaled .- v).^2 for (data, v) in zip(train_data.data, vs)]))
        ρ_loss = mean(mean.([(data.profile.ρ.scaled .- ρ).^2 for (data, ρ) in zip(train_data.data, ρs)]))

        ρ_prefactor = 1
        u_prefactor = ρ_loss / u_loss
        v_prefactor = ρ_loss / v_loss

        return (u=u_prefactor, v=v_prefactor, ρ=ρ_prefactor)
    end

    losses_prefactor = compute_loss_prefactor(ps)

    function loss_DE(p)
        preds = predict_DE(p)

        us = [@view(pred[1:coarse_size, :]) for pred in preds]
        vs = [@view(pred[coarse_size+1:2*coarse_size, :]) for pred in preds]
        ρs = [@view(pred[2*coarse_size+1:3*coarse_size, :]) for pred in preds]

        u_loss = losses_prefactor.u * mean(mean.([(data.profile.u.scaled .- u).^2 for (data, u) in zip(train_data.data, us)]))
        v_loss = losses_prefactor.v * mean(mean.([(data.profile.v.scaled .- v).^2 for (data, v) in zip(train_data.data, vs)]))
        ρ_loss = losses_prefactor.ρ * mean(mean.([(data.profile.ρ.scaled .- ρ).^2 for (data, ρ) in zip(train_data.data, ρs)]))

        loss = u_loss + v_loss + ρ_loss

        individual_loss = (u=u_loss, v=v_loss, ρ=ρ_loss)
        return loss, preds, individual_loss, p
    end

    iter = 0

    losses = fill(1e-8, maxiter+1)
    u_losses = fill(1e-8, maxiter+1)
    v_losses = fill(1e-8, maxiter+1)
    ρ_losses = fill(1e-8, maxiter+1)

    wall_clock = [time_ns()]

    callback = function (opt_state, l, pred, ind_loss, p)
        @printf("%s, Δt %s, iter %d/%d, loss total %6.10e, u %6.5e, v %6.5e, ρ %6.5e, ν_conv %6.3e, ν_shear %6.3e, m %6.3e, Pr %6.3e \n",
                Dates.now(), prettytime(1e-9 * (time_ns() - wall_clock[1])), iter, maxiter, l, ind_loss.u, ind_loss.v, ind_loss.ρ,
                p.ν_conv, p.ν_shear, p.m, p.Pr)
        losses[iter+1] = l
        u_losses[iter+1] = ind_loss.u
        v_losses[iter+1] = ind_loss.v
        ρ_losses[iter+1] = ind_loss.ρ

        iter += 1
        wall_clock[1] = time_ns()
        return false
    end

    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_DE(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, ps)

    res = Optimization.solve(optprob, optimizer, callback=callback, maxiters=maxiter)

    sols_posttraining = predict_DE_posttraining(res.u)

    uw_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    vw_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    wρ_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    νs_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    κs_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    Ri_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    Ri_truth = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]

    for (i, sol) in enumerate(sols_posttraining)
        for t in eachindex(sol.t)
            uw, vw, wρ = predict_total_flux_dimensional(sol[:, t], res.u, params[i])
            Ris = calculate_Ri(inv(params[i].scaling.u).(sol[1:coarse_size, t]), 
                               inv(params[i].scaling.v).(sol[coarse_size+1:2*coarse_size, t]), 
                               inv(params[i].scaling.ρ).(sol[2*coarse_size+1:3*coarse_size, t]), 
                               params[i].Dᶠ, params[i].g, eos.reference_density, clamp_lims=Ri_clamp_lims)

            Ris_truth = calculate_Ri(train_data_plot.data[i].profile.u.unscaled[:, t], 
                                     train_data_plot.data[i].profile.v.unscaled[:, t], 
                                     train_data_plot.data[i].profile.ρ.unscaled[:, t], 
                                     params[i].Dᶠ, params[i].g, eos.reference_density, clamp_lims=Ri_clamp_lims)

            νs, κs = predict_diffusivities(Ris, res.u)

            uw_posttraining[i][:, t] .= uw
            vw_posttraining[i][:, t] .= vw
            wρ_posttraining[i][:, t] .= wρ
            νs_posttraining[i][:, t] .= νs
            κs_posttraining[i][:, t] .= κs
            Ri_posttraining[i][:, t] .= Ris
            Ri_truth[i][:, t] .= Ris_truth
        end
    end

    losses = losses[1:iter]
    u_losses = u_losses[1:iter]
    v_losses = v_losses[1:iter]
    ρ_losses = ρ_losses[1:iter]

    flux_posttraining = (uw=uw_posttraining, vw=vw_posttraining, wρ=wρ_posttraining)
    losses = (total=losses, u=u_losses, v=v_losses, ρ=ρ_losses)
    diffusivities_posttraining = (ν=νs_posttraining, κ=κs_posttraining, Ri=Ri_posttraining, Ri_truth=Ri_truth)

    return res, loss_DE(res.u), sols_posttraining, flux_posttraining, losses, diffusivities_posttraining
end

function plot_loss(losses, FILE_DIR; epoch=1)
    fig = Figure(size=(1000, 600))
    axtotalloss = CairoMakie.Axis(fig[1, 1], title="Total Loss", xlabel="Iterations", ylabel="Loss", yscale=log10)
    axindividualloss = CairoMakie.Axis(fig[1, 2], title="Individual Loss", xlabel="Iterations", ylabel="Loss", yscale=log10)

    lines!(axtotalloss, losses.total, label="Total Loss")

    lines!(axindividualloss, losses.u, label="u")
    lines!(axindividualloss, losses.v, label="v")
    lines!(axindividualloss, losses.ρ, label="ρ")

    axislegend(axindividualloss, position=:rt)
    save("$(FILE_DIR)/losses_epoch$(epoch).png", fig, px_per_unit=8)
end

function animate_data(truth_data, scaling, sols, fluxes, diffusivities, index, FILE_DIR; coarse_size=32, epoch=1)
    fig = Figure(size=(1920, 1080))
    axu = CairoMakie.Axis(fig[1, 1], title="u", xlabel="u (m s⁻¹)", ylabel="z (m)")
    axv = CairoMakie.Axis(fig[1, 2], title="v", xlabel="v (m s⁻¹)", ylabel="z (m)")
    axρ = CairoMakie.Axis(fig[1, 3], title="ρ", xlabel="ρ (kg m⁻³)", ylabel="z (m)")
    axuw = CairoMakie.Axis(fig[2, 1], title="uw", xlabel="uw (m² s⁻²)", ylabel="z (m)")
    axvw = CairoMakie.Axis(fig[2, 2], title="vw", xlabel="vw (m² s⁻²)", ylabel="z (m)")
    axwρ = CairoMakie.Axis(fig[2, 3], title="wρ", xlabel="wρ (m s⁻¹ kg m⁻³)", ylabel="z (m)")
    axRi = CairoMakie.Axis(fig[1, 4], title="Ri", xlabel="Ri", ylabel="z (m)")
    axdiffusivity = CairoMakie.Axis(fig[2, 4], title="Diffusivity", xlabel="Diffusivity (m² s⁻¹)", ylabel="z (m)")

    n = Observable(1)
    zC = truth_data.data[index].metadata["zC"]
    zF = truth_data.data[index].metadata["zF"]

    u_training = inv(scaling.u).(sols[index][1:coarse_size, :])
    v_training = inv(scaling.v).(sols[index][coarse_size+1:2*coarse_size, :])
    ρ_training = inv(scaling.ρ).(sols[index][2*coarse_size+1:3*coarse_size, :])

    uw_training = fluxes.uw[index]
    vw_training = fluxes.vw[index]
    wρ_training = fluxes.wρ[index]

    ulim = (find_min(u_training, truth_data.data[index].profile.u.unscaled) - 1e-7, find_max(u_training, truth_data.data[index].profile.u.unscaled) + 1e-7)
    vlim = (find_min(v_training, truth_data.data[index].profile.v.unscaled) - 1e-7, find_max(v_training, truth_data.data[index].profile.v.unscaled) + 1e-7)
    ρlim = (find_min(ρ_training, truth_data.data[index].profile.ρ.unscaled), find_max(ρ_training, truth_data.data[index].profile.ρ.unscaled))

    uwlim = (find_min(uw_training, truth_data.data[index].flux.uw.column.unscaled) - 1e-7, find_max(uw_training, truth_data.data[index].flux.uw.column.unscaled) + 1e-7)
    vwlim = (find_min(vw_training, truth_data.data[index].flux.vw.column.unscaled) - 1e-7, find_max(vw_training, truth_data.data[index].flux.vw.column.unscaled) + 1e-7)
    wρlim = (find_min(wρ_training, truth_data.data[index].flux.wρ.column.unscaled), find_max(wρ_training, truth_data.data[index].flux.wρ.column.unscaled))

    Rilim = (find_min(diffusivities.Ri[index], diffusivities.Ri_truth[index]), find_max(diffusivities.Ri[index], diffusivities.Ri_truth[index]))
    diffusivitylim = (find_min(diffusivities.ν[index], diffusivities.κ[index]), find_max(diffusivities.ν[index], diffusivities.κ[index]))

    u_truthₙ = @lift truth_data.data[index].profile.u.unscaled[:, $n]
    v_truthₙ = @lift truth_data.data[index].profile.v.unscaled[:, $n]
    ρ_truthₙ = @lift truth_data.data[index].profile.ρ.unscaled[:, $n]

    uw_truthₙ = @lift truth_data.data[index].flux.uw.column.unscaled[:, $n]
    vw_truthₙ = @lift truth_data.data[index].flux.vw.column.unscaled[:, $n]
    wρ_truthₙ = @lift truth_data.data[index].flux.wρ.column.unscaled[:, $n]

    u_trainingₙ = @lift u_training[:, $n]
    v_trainingₙ = @lift v_training[:, $n]
    ρ_trainingₙ = @lift ρ_training[:, $n]

    uw_trainingₙ = @lift uw_training[:, $n]
    vw_trainingₙ = @lift vw_training[:, $n]
    wρ_trainingₙ = @lift wρ_training[:, $n]

    Ri_truthₙ = @lift diffusivities.Ri_truth[index][:, $n]
    Riₙ = @lift diffusivities.Ri[index][:, $n]
    νₙ = @lift diffusivities.ν[index][:, $n]
    κₙ = @lift diffusivities.κ[index][:, $n]

    Qᵁ = truth_data.data[index].metadata["momentum_flux"]
    Qᴿ = truth_data.data[index].metadata["density_flux"]
    f = truth_data.data[index].metadata["coriolis_parameter"]
    times = truth_data.data[index].times
    Nt = length(times)

    time_str = @lift "Qᵁ = $(Qᵁ) m² s⁻², Qᴿ = $(Qᴿ) m s⁻¹ kg m⁻³, f = $(f) s⁻¹, Time = $(round(times[$n]/24/60^2, digits=3)) days"

    lines!(axu, u_truthₙ, zC, label="Truth")
    lines!(axu, u_trainingₙ, zC, label="NDE")

    lines!(axv, v_truthₙ, zC, label="Truth")
    lines!(axv, v_trainingₙ, zC, label="NDE")

    lines!(axρ, ρ_truthₙ, zC, label="Truth")
    lines!(axρ, ρ_trainingₙ, zC, label="NDE")

    lines!(axuw, uw_truthₙ, zF, label="Truth")
    lines!(axuw, uw_trainingₙ, zF, label="NDE")

    lines!(axvw, vw_truthₙ, zF, label="Truth")
    lines!(axvw, vw_trainingₙ, zF, label="NDE")

    lines!(axwρ, wρ_truthₙ, zF, label="Truth")
    lines!(axwρ, wρ_trainingₙ, zF, label="NDE")

    lines!(axRi, Ri_truthₙ, zF, label="Truth")
    lines!(axRi, Riₙ, zF, label="NDE")

    lines!(axdiffusivity, νₙ, zF, label="ν")
    lines!(axdiffusivity, κₙ, zF, label="κ")

    axislegend(axu, position=:rb)
    axislegend(axdiffusivity, position=:rb)

    Label(fig[0, :], time_str, font=:bold, tellwidth=false)

    xlims!(axu, ulim)
    xlims!(axv, vlim)
    xlims!(axρ, ρlim)
    xlims!(axuw, uwlim)
    xlims!(axvw, vwlim)
    xlims!(axwρ, wρlim)
    xlims!(axRi, Rilim)
    xlims!(axdiffusivity, diffusivitylim)

    CairoMakie.record(fig, "$(FILE_DIR)/training_$(index)_epoch$(epoch).mp4", 1:Nt, framerate=15) do nn
        n[] = nn
    end
end

optimizers = [OptimizationOptimisers.ADAM(1e-3),
              OptimizationOptimisers.ADAM(5e-4),
              OptimizationOptimisers.ADAM(2e-4),
              OptimizationOptimisers.ADAM(1e-4),
              OptimizationOptimJL.BFGS()]

maxiters = [500, 500, 1000, 1000, 300]
# optimizers = [OptimizationOptimisers.Adam(5e-4)]
# maxiters = [5]

for (epoch, (optimizer, maxiter)) in enumerate(zip(optimizers, maxiters))
    res, loss, sols, fluxes, losses, diffusivities = optimize_parameters(train_data, train_data_plot, ps, maxiter=maxiter, optimizer=optimizer, Ri_clamp_lims=(-Inf, Inf), solver=VCABM3())
    u = res.u
    jldsave("$(FILE_DIR)/training_results_$(epoch).jld2"; res, u, loss, sols, fluxes, losses, diffusivities)
    for i in eachindex(field_datasets)
        animate_data(train_data_plot, train_data.scaling, sols, fluxes, diffusivities, i, FILE_DIR, epoch=epoch)
    end
    plot_loss(losses, FILE_DIR, epoch=epoch)
    ps .= u
end

# xs = -1:0.01:10
# νs = local_Ri_ν_convectivestep_shearlinear.(xs, ps.ν_conv, ps.ν_shear, ps.m)

# lines(xs, νs, color=:blue, linewidth=2, label="NDE")