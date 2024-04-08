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

FILE_DIR = "./training_output/CNN_kernel4_4channels_swish_local_diffusivity_convectivetanh_shearlinear_glorot_freeconvection_mse_Adam1e-3"
mkpath(FILE_DIR)
@info "$(FILE_DIR)"

LES_FILE_DIRS = [
    "./LES_training/linearTS_b_dTdz_0.0013_dSdz_-0.0014_QU_0.0_QB_8.0e-7_T_4.3_S_33.5_f_-0.00012_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_b_dTdz_0.013_dSdz_0.00075_QU_0.0_QB_8.0e-7_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    # "./LES_training/linearTS_b_dTdz_0.014_dSdz_0.0021_QU_0.0_QB_8.0e-7_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    # "./LES_training/linearTS_b_dTdz_-0.025_dSdz_-0.0045_QU_0.0_QB_8.0e-7_T_-3.6_S_33.9_f_-0.000125_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
]

BASECLOSURE_FILE_DIR = "./training_output/local_diffusivity_convectivetanh_shearlinear_rho_SW_FC_WWSC_SWWC/training_results_5.jld2"

field_datasets = [FieldDataset(FILE_DIR, backend=OnDisk()) for FILE_DIR in LES_FILE_DIRS]

ps_baseclosure = jldopen(BASECLOSURE_FILE_DIR, "r")["u"]

timeframes = [25:10:length(data["ubar"].times) for data in field_datasets]
full_timeframes = [25:length(data["ubar"].times) for data in field_datasets]
train_data = LESDatasetsB(field_datasets, ZeroMeanUnitVarianceScaling, timeframes)
coarse_size = 32

train_data_plot = LESDatasetsB(field_datasets, ZeroMeanUnitVarianceScaling, full_timeframes)

rng = Random.default_rng(123)

CNN = Chain(Conv(Tuple(4), 1 => 4, swish),
            FlattenLayer())

decoder = Chain(Dense(118, 256, swish),
                Dense(256, 256, swish),
                Dense(256, 31))

ps_CNN, st_CNN = Lux.setup(rng, CNN)
ps_decoder, st_decoder = Lux.setup(rng, decoder)

ps_CNN = ps_CNN |> ComponentArray .|> Float64
ps_decoder = ps_decoder |> ComponentArray .|> Float64

CNN(rand(32, 1, 1), ps_CNN, st_CNN)[1]
ps_CNN .= glorot_uniform(rng, Float64, length(ps_CNN))
ps_decoder .= glorot_uniform(rng, Float64, length(ps_decoder))

ps_decoder .*= 1e-5

NNs = (; CNN=CNN, decoder=decoder)
ps_training = ComponentArray(CNN=ps_CNN, decoder=ps_decoder)
st_NN = (; CNN=st_CNN, decoder=st_decoder)

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
    x₀s = [data.profile.ρ.scaled[:, 1] for data in train_data.data] |> dev
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
                                 wρ = (scaled = (top=data.flux.wρ.surface.scaled, bottom=data.flux.wρ.bottom.scaled),
                                       unscaled = (top=data.flux.wρ.surface.unscaled, bottom=data.flux.wρ.bottom.unscaled)),
                            scaling = train_data.scaling,
                            ) for (data, plot_data) in zip(train_data.data, train_data_plot.data)] |> dev

    function predict_residual_flux(ρ_hat, p, params, st)
        embedding = vcat(vec(first(NNs.CNN(reshape(ρ_hat, params.coarse_size, 1, 1), p.CNN, st.CNN))), params.wρ.scaled.top, params.f_scaled)

        NN_pred = first(NNs.decoder(embedding, p.decoder, st.decoder))

        wρ = vcat(0, NN_pred, 0)

        return wρ
    end

    function predict_residual_flux_dimensional(ρ_hat, p, params, st)
        wρ_hat = predict_residual_flux(ρ_hat, p, params, st)
        wρ = inv(params.scaling.wρ).(wρ_hat)

        wρ = wρ .- wρ[1]

        return wρ
    end

    function predict_boundary_flux(params)
        wρ = vcat(fill(params.wρ.scaled.bottom, coarse_size), params.wρ.scaled.top)

        return wρ
    end

    function predict_diffusivities(Ris, p)
        νs = local_Ri_ν_convectivetanh_shearlinear.(Ris, ps_baseclosure.ν_conv, ps_baseclosure.ν_shear, ps_baseclosure.m, ps_baseclosure.ΔRi)
        κs = local_Ri_κ_convectivetanh_shearlinear.(νs, ps_baseclosure.Pr)
        return νs, κs
    end

    function predict_diffusive_flux(ρ, ρ_hat, p, params, st)
        Ris = calculate_Ri(zeros(coarse_size), zeros(coarse_size), ρ, params.Dᶠ, params.g, eos.reference_density, clamp_lims=Ri_clamp_lims)
        νs, κs = predict_diffusivities(Ris, p)

        ∂ρ∂z_hat = params.Dᶠ_hat * ρ_hat

        wρ_diffusive = -κs .* ∂ρ∂z_hat
        return wρ_diffusive
    end

    function predict_diffusive_boundary_flux_dimensional(ρ, ρ_hat, p, params, st)
        _wρ_diffusive = predict_diffusive_flux(ρ, ρ_hat, p, params, st)
        _wρ_boundary = predict_boundary_flux(params)

        wρ_diffusive = params.scaling.ρ.σ / params.H .* _wρ_diffusive

        wρ_boundary = inv(params.scaling.wρ).(_wρ_boundary)

        wρ = wρ_diffusive .+ wρ_boundary

        return wρ
    end

    function unscale_profiles(ρ_hat, params)
        ρ = inv(params.scaling.ρ).(ρ_hat)

        return ρ
    end

    function calculate_scaled_profile_gradients(ρ, params)
        Dᶠ = params.Dᶠ

        ∂ρ∂z_hat = params.scaling.∂ρ∂z.(Dᶠ * ρ)

        return ∂ρ∂z_hat
    end

    function NDE(x, p, t, params, st)
        f = params.f
        Dᶜ_hat = params.Dᶜ_hat
        scaling = params.scaling
        τ, H = params.τ, params.H

        ρ_hat = x
        ρ = unscale_profiles(ρ_hat, params)
        
        wρ_residual = predict_residual_flux(ρ_hat, p, params, st)
        wρ_boundary = predict_boundary_flux(params)
        wρ_diffusive = predict_diffusive_flux(ρ, ρ_hat, p, params, st)

        dρ = -τ / H^2 .* (Dᶜ_hat * wρ_diffusive) .- τ / H * scaling.wρ.σ / scaling.ρ.σ .* (Dᶜ_hat * (wρ_boundary .+ wρ_residual))

        return dρ
    end

    function predict_NDE(p)
        probs = [ODEProblem((x, p′, t) -> NDE(x, p′, t, param, st_NN), x₀, (param.scaled_time[1], param.scaled_time[end]), p) for (x₀, param) in zip(x₀s, params)]
        sols = [Array(solve(prob, solver, saveat=param.scaled_time, reltol=1e-5, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP(), checkpointing=true))) for (param, prob) in zip(params, probs)]
        return sols
    end

    function predict_NDE_posttraining(p)
        probs = [ODEProblem((x, p′, t) -> NDE(x, p′, t, param, st_NN), x₀, (param.scaled_original_time[1], param.scaled_original_time[end]), p) for (x₀, param) in zip(x₀s, params)]
        sols = [solve(prob, solver, saveat=param.scaled_original_time, reltol=1e-5) for (param, prob) in zip(params, probs)]
        return sols
    end

    function predict_NDE_noNN()
        probs = [ODEProblem((x, p′, t) -> NDE(x, p′, t, param, st_NN), x₀, (param.scaled_original_time[1], param.scaled_original_time[end]), ps_zeros) for (x₀, param) in zip(x₀s, params)]
        sols = [solve(prob, solver, saveat=param.scaled_original_time, reltol=1e-5) for (param, prob) in zip(params, probs)]
        return sols
    end

    function predict_scaled_profiles(p)
        preds = predict_NDE(p)
        ρ_hats = preds
        
        ρs = [inv(param.scaling.ρ).(ρ_hat) for (ρ_hat, param) in zip(ρ_hats, params)]

        ∂ρ∂z_hats = [hcat([param.scaling.∂ρ∂z.(param.Dᶠ * @view(ρ[:, i])) for i in axes(ρ, 2)]...) for (param, ρ) in zip(params, ρs)]

        return ρ_hats, ∂ρ∂z_hats, preds
    end

    function predict_losses(ρ, ∂ρ∂z, loss_scaling=(; ρ=1, ∂ρ∂z=1))
        ρ_loss = loss_scaling.ρ * mean(mean.([(data.profile.ρ.scaled .- ρ).^2 for (data, ρ) in zip(train_data.data, ρ)]))

        ∂ρ∂z_loss = loss_scaling.∂ρ∂z * mean(mean.([(data.profile.∂ρ∂z.scaled .- ∂ρ∂z).^2 for (data, ∂ρ∂z) in zip(train_data.data, ∂ρ∂z)]))

        return (ρ=ρ_loss, ∂ρ∂z=∂ρ∂z_loss)
    end

    function compute_loss_prefactor(ρ_loss, ∂ρ∂z_loss)
        ρ_prefactor = 1

        ∂ρ∂z_prefactor = 1

        profile_loss = ρ_prefactor * ρ_loss
        gradient_loss = ∂ρ∂z_prefactor * ∂ρ∂z_loss

        gradient_prefactor = profile_loss / gradient_loss

        ∂ρ∂z_prefactor *= gradient_prefactor

        return (ρ=ρ_prefactor, ∂ρ∂z=∂ρ∂z_prefactor)
    end

    function compute_loss_prefactor(p)
        ρ, ∂ρ∂z, _ = predict_scaled_profiles(p)
        losses = predict_losses(ρ, ∂ρ∂z)
        return compute_loss_prefactor(losses...)
    end

    @info "Computing prefactor for losses"

    losses_prefactor = compute_loss_prefactor(ps_training)

    function loss_NDE(p)
        ρ, ∂ρ∂z, preds = predict_scaled_profiles(p)
        individual_loss = predict_losses(ρ, ∂ρ∂z, losses_prefactor)
        loss = sum(values(individual_loss))

        return loss, preds, individual_loss
    end

    iter = 0

    losses = zeros(maxiter+1)
    ρ_losses = zeros(maxiter+1)
    ∂ρ∂z_losses = zeros(maxiter+1)

    wall_clock = [time_ns()]

    iter_warmup = optimizer.warmup
    max_learningrate = optimizer.learning_rate
    min_learningrate = optimizer.initial_learning_rate
    # optimizer.eta = min_learningrate

    callback = function (p, l, pred, ind_loss)
        @printf("%s, Δt %s, iter %d/%d, loss total %6.10e, ρ %6.5e, ∂ρ∂z %6.5e, max NN weight %6.5e\n",
                Dates.now(), prettytime(1e-9 * (time_ns() - wall_clock[1])), iter, maxiter, l, 
                ind_loss.ρ, 
                ind_loss.∂ρ∂z,
                maximum(abs, p.u))
        if iter % 200 == 0
            jldsave("$(FILE_DIR)/intermediate_training_results_epoch$(epoch)_iter$(iter).jld2"; u=p.u)
        end
        losses[iter+1] = l
        ρ_losses[iter+1] = ind_loss.ρ
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

    wρ_diffusive_boundary_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]

    wρ_diffusive_boundary_noNN = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]

    wρ_residual_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]

    νs_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    κs_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    
    νs_noNN = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    κs_noNN = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    
    Ri_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    Ri_noNN = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    Ri_truth = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]

    for (i, sol) in enumerate(sols_posttraining)
        for t in eachindex(sol.t)
            ρ_hat = sol[:, t]
            ρ = unscale_profiles(ρ_hat, params[i])

            wρ_diffusive_boundary = predict_diffusive_boundary_flux_dimensional(ρ, ρ_hat, res.u, params[i], st_NN)
            wρ_residual = predict_residual_flux_dimensional(ρ_hat, res.u, params[i], st_NN)

            Ris = calculate_Ri(zeros(coarse_size), zeros(coarse_size), ρ, params[i].Dᶠ, params[i].g, eos.reference_density, clamp_lims=Ri_clamp_lims)

            Ris_truth = calculate_Ri(zeros(coarse_size), 
                                     zeros(coarse_size),
                                     train_data_plot.data[i].profile.ρ.unscaled[:, t], 
                                     params[i].Dᶠ, params[i].g, eos.reference_density, clamp_lims=Ri_clamp_lims)

            νs, κs = predict_diffusivities(Ris, res.u)

            wρ_residual_posttraining[i][:, t] .= wρ_residual

            wρ_diffusive_boundary_posttraining[i][:, t] .= wρ_diffusive_boundary

            νs_posttraining[i][:, t] .= νs
            κs_posttraining[i][:, t] .= κs

            Ri_posttraining[i][:, t] .= Ris
            Ri_truth[i][:, t] .= Ris_truth
        end
    end

    for (i, sol) in enumerate(sols_noNN)
        for t in eachindex(sol.t)
            ρ_hat = sol[:, t]
            ρ = unscale_profiles(ρ_hat, params[i])

            wρ_diffusive_boundary = predict_diffusive_boundary_flux_dimensional(ρ, ρ_hat, res.u, params[i], st_NN)

            Ris = calculate_Ri(zeros(coarse_size), zeros(coarse_size), ρ, params[i].Dᶠ, params[i].g, eos.reference_density, clamp_lims=Ri_clamp_lims)

            νs, κs = predict_diffusivities(Ris, res.u)

            wρ_diffusive_boundary_noNN[i][:, t] .= wρ_diffusive_boundary

            νs_noNN[i][:, t] .= νs
            κs_noNN[i][:, t] .= κs

            Ri_noNN[i][:, t] .= Ris
        end
    end

    wρ_total_posttraining = wρ_residual_posttraining .+ wρ_diffusive_boundary_posttraining

    flux_posttraining = (; wρ = (diffusive_boundary=wρ_diffusive_boundary_posttraining, residual=wρ_residual_posttraining, total=wρ_total_posttraining))

    flux_noNN = (; wρ = (; total=wρ_diffusive_boundary_noNN))

    diffusivities_posttraining = (ν=νs_posttraining, κ=κs_posttraining, Ri=Ri_posttraining, Ri_truth=Ri_truth)
    diffusivities_noNN = (ν=νs_noNN, κ=κs_noNN, Ri=Ri_noNN)

    losses = (total=losses, ρ=ρ_losses, ∂ρ∂z=∂ρ∂z_losses)

    return res, loss_NDE(res.u), sols_posttraining, flux_posttraining, losses, diffusivities_posttraining, sols_noNN, flux_noNN, diffusivities_noNN
end

#%%
function plot_loss(losses, FILE_DIR; epoch=1)
    colors = distinguishable_colors(10, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    
    fig = Figure(size=(1000, 600))
    axtotalloss = CairoMakie.Axis(fig[1, 1], title="Total Loss", xlabel="Iterations", ylabel="Loss", yscale=log10)
    axindividualloss = CairoMakie.Axis(fig[1, 2], title="Individual Loss", xlabel="Iterations", ylabel="Loss", yscale=log10)

    lines!(axtotalloss, losses.total, label="Total Loss", color=colors[1])

    lines!(axindividualloss, losses.ρ, label="ρ", color=colors[5])
    lines!(axindividualloss, losses.∂ρ∂z, label="∂ρ∂z", color=colors[10])

    axislegend(axindividualloss, position=:rt)
    save("$(FILE_DIR)/losses_epoch$(epoch).png", fig, px_per_unit=8)
end

#%%
function animate_data(train_data, scaling, sols, fluxes, diffusivities, sols_noNN, fluxes_noNN, diffusivities_noNN, index, FILE_DIR; coarse_size=32, epoch=1)
    fig = Figure(size=(1200, 1080))
    axρ = CairoMakie.Axis(fig[1, 1], title="ρ", xlabel="ρ (kg m⁻³)", ylabel="z (m)")
    axwρ = CairoMakie.Axis(fig[2, 1], title="wρ", xlabel="wρ (m s⁻¹ kg m⁻³)", ylabel="z (m)")
    axRi = CairoMakie.Axis(fig[1, 2], title="Ri", xlabel="Ri", ylabel="z (m)")
    axdiffusivity = CairoMakie.Axis(fig[2, 2], title="Diffusivity", xlabel="Diffusivity (m² s⁻¹)", ylabel="z (m)")

    n = Observable(1)
    zC = train_data.data[index].metadata["zC"]
    zF = train_data.data[index].metadata["zF"]

    ρ_NDE = inv(scaling.ρ).(sols[index])

    wρ_residual = fluxes.wρ.residual[index]

    wρ_diffusive_boundary = fluxes.wρ.diffusive_boundary[index]

    wρ_total = fluxes.wρ.total[index]

    ρ_noNN = inv(scaling.ρ).(sols_noNN[index])

    wρ_noNN = fluxes_noNN.wρ.total[index]

    ρlim = (find_min(ρ_NDE, train_data.data[index].profile.ρ.unscaled, ρ_noNN), find_max(ρ_NDE, train_data.data[index].profile.ρ.unscaled, ρ_noNN))

    wρlim = (find_min(wρ_residual, wρ_diffusive_boundary, wρ_total, train_data.data[index].flux.wρ.column.unscaled, wρ_noNN),
             find_max(wρ_residual, wρ_diffusive_boundary, wρ_total, train_data.data[index].flux.wρ.column.unscaled, wρ_noNN))

    Rilim = (find_min(diffusivities.Ri[index], diffusivities.Ri_truth[index], diffusivities_noNN.Ri[index],), 
             find_max(diffusivities.Ri[index], diffusivities.Ri_truth[index], diffusivities_noNN.Ri[index],),)

    diffusivitylim = (find_min(diffusivities.ν[index], diffusivities.κ[index], diffusivities_noNN.ν[index], diffusivities_noNN.κ[index]), 
                      find_max(diffusivities.ν[index], diffusivities.κ[index], diffusivities_noNN.ν[index], diffusivities_noNN.κ[index]),)

    ρ_truthₙ = @lift train_data.data[index].profile.ρ.unscaled[:, $n]

    wρ_truthₙ = @lift train_data.data[index].flux.wρ.column.unscaled[:, $n]

    ρ_NDEₙ = @lift ρ_NDE[:, $n]

    ρ_noNNₙ = @lift ρ_noNN[:, $n]

    wρ_residualₙ = @lift wρ_residual[:, $n]

    wρ_diffusive_boundaryₙ = @lift wρ_diffusive_boundary[:, $n]

    wρ_totalₙ = @lift wρ_total[:, $n]

    wρ_noNNₙ = @lift wρ_noNN[:, $n]

    Ri_truthₙ = @lift diffusivities.Ri_truth[index][:, $n]
    Riₙ = @lift diffusivities.Ri[index][:, $n]
    Ri_noNNₙ = @lift diffusivities_noNN.Ri[index][:, $n]

    νₙ = @lift diffusivities.ν[index][:, $n]
    κₙ = @lift diffusivities.κ[index][:, $n]

    ν_noNNₙ = @lift diffusivities_noNN.ν[index][:, $n]
    κ_noNNₙ = @lift diffusivities_noNN.κ[index][:, $n]

    Qᵁ = train_data.data[index].metadata["momentum_flux"]
    Qᴿ = train_data.data[index].metadata["density_flux"]
    f = train_data.data[index].metadata["coriolis_parameter"]
    times = train_data.data[index].times
    Nt = length(times)

    time_str = @lift "Qᵁ = $(Qᵁ) m² s⁻², Qᴿ = $(Qᴿ) m s⁻¹ kg m⁻³, f = $(f) s⁻¹, Time = $(round(times[$n]/24/60^2, digits=3)) days"

    lines!(axρ, ρ_truthₙ, zC, label="Truth")
    lines!(axρ, ρ_noNNₙ, zC, label="Base closure only")
    lines!(axρ, ρ_NDEₙ, zC, label="NDE")

    lines!(axwρ, wρ_truthₙ, zF, label="Truth")
    lines!(axwρ, wρ_noNNₙ, zF, label="Base closure only")
    lines!(axwρ, wρ_diffusive_boundaryₙ, zF, label="Base closure")
    lines!(axwρ, wρ_residualₙ, zF, label="Residual")
    lines!(axwρ, wρ_totalₙ, zF, label="NDE")

    lines!(axRi, Ri_truthₙ, zF, label="Truth")
    lines!(axRi, Ri_noNNₙ, zF, label="Base closure only")
    lines!(axRi, Riₙ, zF, label="NDE")

    lines!(axdiffusivity, ν_noNNₙ, zF, label="ν, Base closure only")
    lines!(axdiffusivity, κ_noNNₙ, zF, label="κ, Base closure only")
    lines!(axdiffusivity, νₙ, zF, label="ν, NDE")
    lines!(axdiffusivity, κₙ, zF, label="κ, NDE")

    axislegend(axρ, position=:lb)
    axislegend(axwρ, position=:rb)
    axislegend(axdiffusivity, position=:rb)
    
    Label(fig[0, :], time_str, font=:bold, tellwidth=false)

    xlims!(axρ, ρlim)
    xlims!(axwρ, wρlim)
    xlims!(axRi, Rilim)
    xlims!(axdiffusivity, diffusivitylim)

    CairoMakie.record(fig, "$(FILE_DIR)/training_$(index)_epoch$(epoch).mp4", 1:Nt, framerate=15) do nn
        n[] = nn
    end
end

optimizers = [Optimizer(initial=OptimizationOptimisers.Adam(1e-3), initial_learning_rate=1e-3, learning_rate=1e-3, warmup=1, maxiter=1000),
              Optimizer(initial=OptimizationOptimisers.Adam(1e-6), initial_learning_rate=1e-6, learning_rate=3e-4, warmup=40, maxiter=1000),]

# optimizers = [Optimizer(initial=OptimizationOptimisers.Adam(1e-3), initial_learning_rate=1e-3, learning_rate=1e-3, warmup=1, maxiter=2)]

for (epoch, optimizer) in enumerate(optimizers)
    res, loss, sols, fluxes, losses, diffusivities, sols_noNN, fluxes_noNN, diffusivities_noNN = train_NDE(train_data, train_data_plot, NNs, ps_training, ps_baseclosure, st_NN, rng, solver=ROCK4(), optimizer=optimizer, epoch=epoch)
    u = res.u
    jldsave("$(FILE_DIR)/training_results_$(epoch).jld2"; res, u, loss, sols, fluxes, losses, NNs, st_NN, diffusivities, sols_noNN, fluxes_noNN, diffusivities_noNN)
    plot_loss(losses, FILE_DIR, epoch=epoch)
    for i in eachindex(field_datasets)
        animate_data(train_data_plot, train_data.scaling, sols, fluxes, diffusivities, sols_noNN, fluxes_noNN, diffusivities_noNN, i, FILE_DIR, epoch=epoch)
    end
    ps_training .= u
end

rm.(glob("$(FILE_DIR)/intermediate_training_results_*.jld2"))