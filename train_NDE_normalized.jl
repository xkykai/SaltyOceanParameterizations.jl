using SaltyOceanParameterizations
using SaltyOceanParameterizations.DataWrangling
using SaltyOceanParameterizations: calculate_Ri, local_Ri_ν_piecewise_linear, local_Ri_κ_piecewise_linear
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

function find_min(a...)
    return minimum(minimum.([a...]))
end
  
function find_max(a...)
    return maximum(maximum.([a...]))
end

# FILE_DIR = "./training_output/SW_FC_UNet_2358_2level_1layer_256_swish_local_diffusivity_piecewise_linear_rho_lossequal_mae_Adam_glorot_lossequal_mae_ADAM5e-4_test"
# FILE_DIR = "./training_output/SW_FC_UNet_2358_2level_1layer_256_swish_local_diffusivity_piecewise_linear_rho_rho0.8_gradient_Adam_glorot_lossequal_mae_ADAM5e-4_test"
FILE_DIR = "./training_output/SW_FC_UNet_2358_2level_1layer_256_swish_nobaseclosure_glorot_lossequal_mae_normalized_test"
mkpath(FILE_DIR)
@info "$(FILE_DIR)"

LES_FILE_DIRS = [
    # "./LES_training/linearTS_b_dTdz_0.0013_dSdz_-0.0014_QU_0.0_QB_8.0e-7_T_4.3_S_33.5_f_-0.00012_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_b_dTdz_0.013_dSdz_0.00075_QU_0.0_QB_8.0e-7_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_b_dTdz_0.014_dSdz_0.0021_QU_0.0_QB_8.0e-7_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    # "./LES_training/linearTS_b_dTdz_-0.025_dSdz_-0.0045_QU_0.0_QB_8.0e-7_T_-3.6_S_33.9_f_-0.000125_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",

    "./LES_training/linearTS_b_dTdz_0.0013_dSdz_-0.0014_QU_-0.0005_QB_0.0_T_4.3_S_33.5_f_-0.00012_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    # "./LES_training/linearTS_b_dTdz_0.013_dSdz_0.00075_QU_-0.0005_QB_0.0_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    # "./LES_training/linearTS_b_dTdz_0.014_dSdz_0.0021_QU_-0.0005_QB_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_b_dTdz_-0.025_dSdz_-0.0045_QU_-0.0005_QB_0.0_T_-3.6_S_33.9_f_-0.000125_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
]

# BASECLOSURE_FILE_DIR = "./training_output/local_diffusivity_piecewise_linear_rho_noclamp_lossequal_SW_FC_largeinitialdiffusivity_mae/training_results_4.jld2"
BASECLOSURE_FILE_DIR = "./training_output/local_diffusivity_piecewise_linear_rho_noclamp_rho0.8_gradient_Adam_SW_FC_largeinitialdiffusivity/training_results_4.jld2"

field_datasets = [FieldDataset(FILE_DIR, backend=OnDisk()) for FILE_DIR in LES_FILE_DIRS]

ps_baseclosure = jldopen(BASECLOSURE_FILE_DIR, "r")["u"]

timeframes = [25:10:length(data["ubar"].times) for data in field_datasets]
full_timeframes = [25:length(data["ubar"].times) for data in field_datasets]
train_data = LESDatasetsB(field_datasets, ZeroMeanUnitVarianceScaling, timeframes)
coarse_size = 32

train_data_plot = LESDatasetsB(field_datasets, ZeroMeanUnitVarianceScaling, full_timeframes)

rng = Random.default_rng(123)

level3_channel = 32
level2_channel = 16
level1_channel = 8
output_channel = 8
input_channel = 7

level3 = Chain(MaxPool(Tuple(2), stride=2), 
               Conv(Tuple(5), level2_channel => level3_channel, swish, pad=SamePad()),
               Conv(Tuple(5), level3_channel => level3_channel, swish, pad=SamePad()),
               Upsample(2),
               Conv(Tuple(5), level3_channel => level2_channel, swish, pad=SamePad()))

level2down = Chain(MaxPool(Tuple(2), stride=2), 
                    Conv(Tuple(5), level1_channel => level2_channel, swish, pad=SamePad()),
                    Conv(Tuple(5), level2_channel => level2_channel, swish, pad=SamePad()))

level2up = Chain(Conv(Tuple(5), level3_channel => level2_channel, swish, pad=SamePad()),
                 Conv(Tuple(5), level2_channel => level2_channel, swish, pad=SamePad()),
                 Upsample(2),
                 Conv(Tuple(5), level2_channel => level1_channel, swish, pad=SamePad()))

level1down = Chain(Conv(Tuple(5), input_channel => level1_channel, swish, pad=SamePad()),
                   Conv(Tuple(5), level1_channel => level1_channel, swish, pad=SamePad()))

level1up = Chain(Conv(Tuple(5), level2_channel => level1_channel, swish, pad=SamePad()),
                 Conv(Tuple(5), level1_channel => level1_channel, swish, pad=SamePad()),
                 Conv(Tuple(5), level1_channel => output_channel, swish, pad=SamePad()))

decoder = Chain(FlattenLayer(),
                Dense(32*output_channel, 256, swish),
                Dense(256, 93))

function concat_two_layers(output, input)
    return cat(output, input, dims=2)
end

level2_block = Chain(level2down, SkipConnection(level3, concat_two_layers), level2up)

UNet = Chain(level1down, SkipConnection(level2_block, concat_two_layers), level1up)

NN = Chain(UNet, decoder)

ps, st = Lux.setup(rng, NN)

ps = ps |> ComponentArray .|> Float64

ps .= glorot_uniform(rng, Float64, length(ps))

# ps .*= 0

NNs = (; NDE=NN)
ps_training = ComponentArray(NDE=ps)
st_NN = (; NDE=st)

function train_NDE(train_data, train_data_plot, NNs, ps_training, ps_baseclosure, st_NN, rng; 
                   coarse_size=32, dev=cpu_device(), maxiter=10, optimizer=OptimizationOptimisers.ADAM(0.001), solver=ROCK2(), Ri_clamp_lims=(-Inf, Inf), epoch=1)
    train_data = train_data |> dev
    x₀s = [vcat(data.profile.u.scaled[:, 1], data.profile.v.scaled[:, 1], data.profile.ρ.scaled[:, 1]) for data in train_data.data] |> dev
    eos = TEOS10EquationOfState()
    ps_zeros = deepcopy(ps_training)
    ps_zeros.NDE .= 0

    function construct_gaussian_fourier_features(scalars, output_dims, rng)
        N = length(scalars)
        if maximum(scalars) == minimum(scalars)
            scalars_normalized = zeros(length(scalars))
        else
            scalars_normalized = (scalars .- minimum(scalars)) ./ (maximum(scalars) - minimum(scalars))
        end
        m = Int(output_dims / 2)
        gaussian = Normal()

        bs = [[rand(rng, gaussian) for _ in 1:m] for _ in 1:N]

        features = [zeros(output_dims) for _ in 1:N]

        for (scalar, feature, b) in zip(scalars_normalized, features, bs)
            for i in 1:m
                index = i*2 - 1
                feature[index] = cos(2π * scalar * b[i])
                feature[index+1] = sin(2π * scalar * b[i])
            end
        end

        return features
    end

    fs_scaled = [data.coriolis.scaled for data in train_data.data] |> dev
    uws_top_scaled = [data.flux.uw.surface.scaled for data in train_data.data] |> dev
    vws_top_scaled = [data.flux.vw.surface.scaled for data in train_data.data] |> dev
    wρs_top_scaled = [data.flux.wρ.surface.scaled for data in train_data.data] |> dev

    f_features = construct_gaussian_fourier_features(fs_scaled, coarse_size, rng) |> dev
    uw_features = construct_gaussian_fourier_features(uws_top_scaled, coarse_size, rng) |> dev
    vw_features = construct_gaussian_fourier_features(vws_top_scaled, coarse_size, rng) |> dev
    wρ_features = construct_gaussian_fourier_features(wρs_top_scaled, coarse_size, rng) |> dev

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
                                 wρ = (scaled = (top=data.flux.wρ.surface.scaled, bottom=data.flux.wρ.bottom.scaled),
                                       unscaled = (top=data.flux.wρ.surface.unscaled, bottom=data.flux.wρ.bottom.unscaled)),
                            scaling = train_data.scaling,
                            feature = (f=f_feature, uw=uw_feature, vw=vw_feature, wρ=wρ_feature),
                      profile_range = (u = vec(maximum(data.profile.u.scaled, dims=2) .- minimum(data.profile.u.scaled, dims=2) .+ 1e-8),
                                       v = vec(maximum(data.profile.v.scaled, dims=2) .- minimum(data.profile.v.scaled, dims=2) .+ 1e-8),
                                       ρ = vec(maximum(data.profile.ρ.scaled, dims=2) .- minimum(data.profile.ρ.scaled, dims=2)),
                                       ∂u∂z = vec(maximum(data.profile.∂u∂z.scaled, dims=2) .- minimum(data.profile.∂u∂z.scaled, dims=2) .+ 1e-8),
                                       ∂v∂z = vec(maximum(data.profile.∂v∂z.scaled, dims=2) .- minimum(data.profile.∂v∂z.scaled, dims=2) .+ 1e-8),
                                       ∂ρ∂z = vec(maximum(data.profile.∂ρ∂z.scaled, dims=2) .- minimum(data.profile.∂ρ∂z.scaled, dims=2)) .+ 1e-8),
                            ) for (data, plot_data, f_feature, uw_feature, vw_feature, wρ_feature) in zip(train_data.data, train_data_plot.data, f_features, uw_features, vw_features, wρ_features)] |> dev

    function predict_residual_flux(u_hat, v_hat, ρ_hat, p, params, st)
        x′ = reshape(hcat(u_hat, v_hat, ρ_hat, params.feature.uw, params.feature.vw, params.feature.wρ, params.feature.f), coarse_size, 7, 1)
        
        interior_size = coarse_size-1
        NN_pred = first(NNs.NDE(x′, p.NDE, st.NDE))

        uw = vcat(0, NN_pred[1:interior_size, 1], 0)
        vw = vcat(0, NN_pred[interior_size+1:2*interior_size, 1], 0)
        wρ = vcat(0, NN_pred[2*interior_size+1:3*interior_size, 1], 0)

        return uw, vw, wρ
    end

    function predict_residual_flux_dimensional(u_hat, v_hat, ρ_hat, p, params, st)
        uw_hat, vw_hat, wρ_hat = predict_residual_flux(u_hat, v_hat, ρ_hat, p, params, st)
        uw = inv(params.scaling.uw).(uw_hat)
        vw = inv(params.scaling.vw).(vw_hat)
        wρ = inv(params.scaling.wρ).(wρ_hat)

        uw = uw .- uw[1]
        vw = vw .- vw[1]
        wρ = wρ .- wρ[1]

        return uw, vw, wρ
    end

    function predict_boundary_flux(params)
        uw = vcat(fill(params.uw.scaled.bottom, coarse_size), params.uw.scaled.top)
        vw = vcat(fill(params.vw.scaled.bottom, coarse_size), params.vw.scaled.top)
        wρ = vcat(fill(params.wρ.scaled.bottom, coarse_size), params.wρ.scaled.top)

        return uw, vw, wρ
    end

    function predict_diffusivities(Ris, p)
        # νs = local_Ri_ν_piecewise_linear.(Ris, ps_baseclosure.ν₁, ps_baseclosure.m)
        # κs = local_Ri_κ_piecewise_linear.(νs, ps_baseclosure.Pr)
        # return νs, κs
        return zeros(coarse_size+1), zeros(coarse_size+1)
    end

    function predict_diffusive_flux(u, v, ρ, u_hat, v_hat, ρ_hat, p, params, st)
        # Ris = calculate_Ri(u, v, ρ, params.Dᶠ, params.g, eos.reference_density, clamp_lims=Ri_clamp_lims)
        # νs, κs = predict_diffusivities(Ris, p)

        # ∂u∂z_hat = params.Dᶠ_hat * u_hat
        # ∂v∂z_hat = params.Dᶠ_hat * v_hat
        # ∂ρ∂z_hat = params.Dᶠ_hat * ρ_hat

        # uw_diffusive = -νs .* ∂u∂z_hat
        # vw_diffusive = -νs .* ∂v∂z_hat
        # wρ_diffusive = -κs .* ∂ρ∂z_hat
        # return uw_diffusive, vw_diffusive, wρ_diffusive
        return zeros(coarse_size+1), zeros(coarse_size+1), zeros(coarse_size+1)
    end

    function predict_diffusive_boundary_flux_dimensional(u, v, ρ, u_hat, v_hat, ρ_hat, p, params, st)
        _uw_diffusive, _vw_diffusive, _wρ_diffusive = predict_diffusive_flux(u, v, ρ, u_hat, v_hat, ρ_hat, p, params, st)
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

    get_u(x::Vector, params) = x[1:params.coarse_size]
    get_v(x::Vector, params) = x[params.coarse_size+1:2*params.coarse_size]
    get_ρ(x::Vector, params) = x[2*params.coarse_size+1:3*params.coarse_size]

    get_u(x::Matrix, params) = @view x[1:params.coarse_size, :]
    get_v(x::Matrix, params) = @view x[params.coarse_size+1:2*params.coarse_size, :]
    get_ρ(x::Matrix, params) = @view x[2*params.coarse_size+1:3*params.coarse_size, :]

    function get_scaled_profiles(x, params)
        u_hat = get_u(x, params)
        v_hat = get_v(x, params)
        ρ_hat = get_ρ(x, params)

        return u_hat, v_hat, ρ_hat
    end

    function unscale_profiles(u_hat, v_hat, ρ_hat, params)
        u = inv(params.scaling.u).(u_hat)
        v = inv(params.scaling.v).(v_hat)
        ρ = inv(params.scaling.ρ).(ρ_hat)

        return u, v, ρ
    end

    function calculate_scaled_profile_gradients(u, v, ρ, params)
        Dᶠ = params.Dᶠ

        ∂u∂z_hat = params.scaling.∂u∂z.(Dᶠ * u)
        ∂v∂z_hat = params.scaling.∂v∂z.(Dᶠ * v)
        ∂ρ∂z_hat = params.scaling.∂ρ∂z.(Dᶠ * ρ)

        return ∂u∂z_hat, ∂v∂z_hat, ∂ρ∂z_hat
    end

    function NDE(x, p, t, params, st)
        f = params.f
        Dᶜ_hat = params.Dᶜ_hat
        scaling = params.scaling
        τ, H = params.τ, params.H

        u_hat, v_hat, ρ_hat = get_scaled_profiles(x, params)
        u, v, ρ = unscale_profiles(u_hat, v_hat, ρ_hat, params)
        
        uw_residual, vw_residual, wρ_residual = predict_residual_flux(u_hat, v_hat, ρ_hat, p, params, st)
        uw_boundary, vw_boundary, wρ_boundary = predict_boundary_flux(params)
        uw_diffusive, vw_diffusive, wρ_diffusive = predict_diffusive_flux(u, v, ρ, u_hat, v_hat, ρ_hat, p, params, st)

        du = -τ / H^2 .* (Dᶜ_hat * uw_diffusive) .- τ / H * scaling.uw.σ / scaling.u.σ .* (Dᶜ_hat * (uw_boundary .+ uw_residual)) .+ f * τ ./ scaling.u.σ .* v
        dv = -τ / H^2 .* (Dᶜ_hat * vw_diffusive) .- τ / H * scaling.vw.σ / scaling.v.σ .* (Dᶜ_hat * (vw_boundary .+ vw_residual)) .- f * τ ./ scaling.v.σ .* u
        dρ = -τ / H^2 .* (Dᶜ_hat * wρ_diffusive) .- τ / H * scaling.wρ.σ / scaling.ρ.σ .* (Dᶜ_hat * (wρ_boundary .+ wρ_residual))

        return vcat(du, dv, dρ)
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
        u_hats = [get_u(pred, param) for (pred, param) in zip(preds, params)]
        v_hats = [get_v(pred, param) for (pred, param) in zip(preds, params)]
        ρ_hats = [get_ρ(pred, param) for (pred, param) in zip(preds, params)]
        
        us = [inv(param.scaling.u).(u_hat) for (u_hat, param) in zip(u_hats, params)]
        vs = [inv(param.scaling.v).(v_hat) for (v_hat, param) in zip(v_hats, params)]
        ρs = [inv(param.scaling.ρ).(ρ_hat) for (ρ_hat, param) in zip(ρ_hats, params)]

        ∂u∂z_hats = [hcat([param.scaling.∂u∂z.(param.Dᶠ * @view(u[:, i])) for i in axes(u, 2)]...) for (param, u) in zip(params, us)]
        ∂v∂z_hats = [hcat([param.scaling.∂v∂z.(param.Dᶠ * @view(v[:, i])) for i in axes(v, 2)]...) for (param, v) in zip(params, vs)]
        ∂ρ∂z_hats = [hcat([param.scaling.∂ρ∂z.(param.Dᶠ * @view(ρ[:, i])) for i in axes(ρ, 2)]...) for (param, ρ) in zip(params, ρs)]

        return u_hats, v_hats, ρ_hats, ∂u∂z_hats, ∂v∂z_hats, ∂ρ∂z_hats, preds
    end

    function predict_losses(u, v, ρ, ∂u∂z, ∂v∂z, ∂ρ∂z, loss_scaling=(; u=1, v=1, ρ=1, ∂u∂z=1, ∂v∂z=1, ∂ρ∂z=1))
        u_loss = loss_scaling.u * mean(mean.([abs.(data.profile.u.scaled .- u) ./ param.profile_range.u  for (data, u, param) in zip(train_data.data, u, params)]))
        v_loss = loss_scaling.v * mean(mean.([abs.(data.profile.v.scaled .- v) ./ param.profile_range.v  for (data, v, param) in zip(train_data.data, v, params)]))
        ρ_loss = loss_scaling.ρ * mean(mean.([abs.(data.profile.ρ.scaled .- ρ) ./ param.profile_range.ρ  for (data, ρ, param) in zip(train_data.data, ρ, params)]))

        ∂u∂z_loss = loss_scaling.∂u∂z * mean(mean.([abs.(data.profile.∂u∂z.scaled .- ∂u∂z) ./ param.profile_range.∂u∂z for (data, ∂u∂z, param) in zip(train_data.data, ∂u∂z, params)]))
        ∂v∂z_loss = loss_scaling.∂v∂z * mean(mean.([abs.(data.profile.∂v∂z.scaled .- ∂v∂z) ./ param.profile_range.∂v∂z for (data, ∂v∂z, param) in zip(train_data.data, ∂v∂z, params)]))
        ∂ρ∂z_loss = loss_scaling.∂ρ∂z * mean(mean.([abs.(data.profile.∂ρ∂z.scaled .- ∂ρ∂z) ./ param.profile_range.∂ρ∂z for (data, ∂ρ∂z, param) in zip(train_data.data, ∂ρ∂z, params)]))

        return (u=u_loss, v=v_loss, ρ=ρ_loss, ∂u∂z=∂u∂z_loss, ∂v∂z=∂v∂z_loss, ∂ρ∂z=∂ρ∂z_loss)
    end

    function compute_loss_prefactor(u_loss, v_loss, ρ_loss, ∂u∂z_loss, ∂v∂z_loss, ∂ρ∂z_loss)
        ρ_prefactor = 1
        u_prefactor = ρ_loss / u_loss
        v_prefactor = ρ_loss / v_loss

        ∂ρ∂z_prefactor = 1
        ∂u∂z_prefactor = ∂ρ∂z_loss / ∂u∂z_loss
        ∂v∂z_prefactor = ∂ρ∂z_loss / ∂v∂z_loss

        profile_loss = u_prefactor * u_loss + v_prefactor * v_loss + ρ_prefactor * ρ_loss
        gradient_loss = ∂u∂z_prefactor * ∂u∂z_loss + ∂v∂z_prefactor * ∂v∂z_loss + ∂ρ∂z_prefactor * ∂ρ∂z_loss

        gradient_prefactor = profile_loss / gradient_loss

        ∂ρ∂z_prefactor *= gradient_prefactor
        ∂u∂z_prefactor *= gradient_prefactor
        ∂v∂z_prefactor *= gradient_prefactor

        return (ρ=ρ_prefactor, u=u_prefactor, v=v_prefactor, ∂ρ∂z=∂ρ∂z_prefactor, ∂u∂z=∂u∂z_prefactor, ∂v∂z=∂v∂z_prefactor)
    end

    function compute_loss_prefactor(p)
        u, v, ρ, ∂u∂z, ∂v∂z, ∂ρ∂z, _ = predict_scaled_profiles(p)
        losses = predict_losses(u, v, ρ, ∂u∂z, ∂v∂z, ∂ρ∂z)
        return compute_loss_prefactor(losses...)
    end

    @info "Computing prefactor for losses"

    losses_prefactor = compute_loss_prefactor(ps_training)

    function loss_NDE(p)
        u, v, ρ, ∂u∂z, ∂v∂z, ∂ρ∂z, preds = predict_scaled_profiles(p)
        individual_loss = predict_losses(u, v, ρ, ∂u∂z, ∂v∂z, ∂ρ∂z, losses_prefactor)
        loss = sum(values(individual_loss))

        return loss, preds, individual_loss
    end

    iter = 0

    losses = zeros(maxiter+1)
    u_losses = zeros(maxiter+1)
    v_losses = zeros(maxiter+1)
    ρ_losses = zeros(maxiter+1)
    ∂u∂z_losses = zeros(maxiter+1)
    ∂v∂z_losses = zeros(maxiter+1)
    ∂ρ∂z_losses = zeros(maxiter+1)

    wall_clock = [time_ns()]

    callback = function (p, l, pred, ind_loss)
        @printf("%s, Δt %s, iter %d/%d, loss total %6.10e, u %6.5e, v %6.5e, ρ %6.5e, ∂u∂z %6.5e, ∂v∂z %6.5e, ∂ρ∂z %6.5e,\n",
                Dates.now(), prettytime(1e-9 * (time_ns() - wall_clock[1])), iter, maxiter, l, 
                ind_loss.u, ind_loss.v, ind_loss.ρ, 
                ind_loss.∂u∂z, ind_loss.∂v∂z, ind_loss.∂ρ∂z)
        if iter % 10 == 0
            jldsave("$(FILE_DIR)/intermediate_training_results_epoch$(epoch)_iter$(iter).jld2"; u=p.u)
        end
        losses[iter+1] = l
        u_losses[iter+1] = ind_loss.u
        v_losses[iter+1] = ind_loss.v
        ρ_losses[iter+1] = ind_loss.ρ
        ∂u∂z_losses[iter+1] = ind_loss.∂u∂z
        ∂v∂z_losses[iter+1] = ind_loss.∂v∂z
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
    sols_noNN = predict_NDE_noNN()

    uw_diffusive_boundary_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    vw_diffusive_boundary_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    wρ_diffusive_boundary_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]

    uw_diffusive_boundary_noNN = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    vw_diffusive_boundary_noNN = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    wρ_diffusive_boundary_noNN = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]

    uw_residual_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    vw_residual_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
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
            u_hat, v_hat, ρ_hat = get_scaled_profiles(sol[:, t], params[i])
            u, v, ρ = unscale_profiles(u_hat, v_hat, ρ_hat, params[i])

            uw_diffusive_boundary, vw_diffusive_boundary, wρ_diffusive_boundary = predict_diffusive_boundary_flux_dimensional(u, v, ρ, u_hat, v_hat, ρ_hat, res.u, params[i], st_NN)
            uw_residual, vw_residual, wρ_residual = predict_residual_flux_dimensional(u_hat, v_hat, ρ_hat, res.u, params[i], st_NN)

            Ris = calculate_Ri(u, v, ρ, params[i].Dᶠ, params[i].g, eos.reference_density, clamp_lims=Ri_clamp_lims)

            Ris_truth = calculate_Ri(train_data_plot.data[i].profile.u.unscaled[:, t], 
                                     train_data_plot.data[i].profile.v.unscaled[:, t],
                                     train_data_plot.data[i].profile.ρ.unscaled[:, t], 
                                     params[i].Dᶠ, params[i].g, eos.reference_density, clamp_lims=Ri_clamp_lims)

            νs, κs = predict_diffusivities(Ris, res.u)

            uw_residual_posttraining[i][:, t] .= uw_residual
            vw_residual_posttraining[i][:, t] .= vw_residual
            wρ_residual_posttraining[i][:, t] .= wρ_residual

            uw_diffusive_boundary_posttraining[i][:, t] .= uw_diffusive_boundary
            vw_diffusive_boundary_posttraining[i][:, t] .= vw_diffusive_boundary
            wρ_diffusive_boundary_posttraining[i][:, t] .= wρ_diffusive_boundary

            νs_posttraining[i][:, t] .= νs
            κs_posttraining[i][:, t] .= κs

            Ri_posttraining[i][:, t] .= Ris
            Ri_truth[i][:, t] .= Ris_truth
        end
    end

    for (i, sol) in enumerate(sols_noNN)
        for t in eachindex(sol.t)
            u_hat, v_hat, ρ_hat = get_scaled_profiles(sol[:, t], params[i])
            u, v, ρ = unscale_profiles(u_hat, v_hat, ρ_hat, params[i])

            uw_diffusive_boundary, vw_diffusive_boundary, wρ_diffusive_boundary = predict_diffusive_boundary_flux_dimensional(u, v, ρ, u_hat, v_hat, ρ_hat, res.u, params[i], st_NN)

            Ris = calculate_Ri(u, v, ρ, params[i].Dᶠ, params[i].g, eos.reference_density, clamp_lims=Ri_clamp_lims)

            νs, κs = predict_diffusivities(Ris, res.u)

            uw_diffusive_boundary_noNN[i][:, t] .= uw_diffusive_boundary
            vw_diffusive_boundary_noNN[i][:, t] .= vw_diffusive_boundary
            wρ_diffusive_boundary_noNN[i][:, t] .= wρ_diffusive_boundary

            νs_noNN[i][:, t] .= νs
            κs_noNN[i][:, t] .= κs

            Ri_noNN[i][:, t] .= Ris
        end
    end

    uw_total_posttraining = uw_residual_posttraining .+ uw_diffusive_boundary_posttraining
    vw_total_posttraining = vw_residual_posttraining .+ vw_diffusive_boundary_posttraining
    wρ_total_posttraining = wρ_residual_posttraining .+ wρ_diffusive_boundary_posttraining

    flux_posttraining = (uw = (diffusive_boundary=uw_diffusive_boundary_posttraining, residual=uw_residual_posttraining, total=uw_total_posttraining),
                         vw = (diffusive_boundary=vw_diffusive_boundary_posttraining, residual=vw_residual_posttraining, total=vw_total_posttraining),
                         wρ = (diffusive_boundary=wρ_diffusive_boundary_posttraining, residual=wρ_residual_posttraining, total=wρ_total_posttraining))

    flux_noNN = (uw = (; total=uw_diffusive_boundary_noNN),
                 vw = (; total=vw_diffusive_boundary_noNN),
                 wρ = (; total=wρ_diffusive_boundary_noNN))

    diffusivities_posttraining = (ν=νs_posttraining, κ=κs_posttraining, Ri=Ri_posttraining, Ri_truth=Ri_truth)
    diffusivities_noNN = (ν=νs_noNN, κ=κs_noNN, Ri=Ri_noNN)

    losses = (total=losses, u=u_losses, v=v_losses, ρ=ρ_losses, ∂u∂z=∂u∂z_losses, ∂v∂z=∂v∂z_losses, ∂ρ∂z=∂ρ∂z_losses)

    return res, loss_NDE(res.u), sols_posttraining, flux_posttraining, losses, diffusivities_posttraining, sols_noNN, flux_noNN, diffusivities_noNN
end

#%%
function plot_loss(losses, FILE_DIR; epoch=1)
    colors = distinguishable_colors(10, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    
    fig = Figure(size=(1000, 600))
    axtotalloss = CairoMakie.Axis(fig[1, 1], title="Total Loss", xlabel="Iterations", ylabel="Loss", yscale=log10)
    axindividualloss = CairoMakie.Axis(fig[1, 2], title="Individual Loss", xlabel="Iterations", ylabel="Loss", yscale=log10)

    lines!(axtotalloss, losses.total, label="Total Loss", color=colors[1])

    lines!(axindividualloss, losses.u, label="u", color=colors[1])
    lines!(axindividualloss, losses.v, label="v", color=colors[2])
    lines!(axindividualloss, losses.ρ, label="ρ", color=colors[5])
    lines!(axindividualloss, losses.∂u∂z, label="∂u∂z", color=colors[6])
    lines!(axindividualloss, losses.∂v∂z, label="∂v∂z", color=colors[7])
    lines!(axindividualloss, losses.∂ρ∂z, label="∂ρ∂z", color=colors[10])

    axislegend(axindividualloss, position=:rt)
    save("$(FILE_DIR)/losses_epoch$(epoch).png", fig, px_per_unit=8)
end

#%%
function animate_data(train_data, scaling, sols, fluxes, diffusivities, sols_noNN, fluxes_noNN, diffusivities_noNN, index, FILE_DIR; coarse_size=32, epoch=1)
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
    zC = train_data.data[index].metadata["zC"]
    zF = train_data.data[index].metadata["zF"]

    u_NDE = inv(scaling.u).(sols[index][1:coarse_size, :])
    v_NDE = inv(scaling.v).(sols[index][coarse_size+1:2*coarse_size, :])
    ρ_NDE = inv(scaling.ρ).(sols[index][2*coarse_size+1:3*coarse_size, :])

    uw_residual = fluxes.uw.residual[index]
    vw_residual = fluxes.vw.residual[index]
    wρ_residual = fluxes.wρ.residual[index]

    uw_diffusive_boundary = fluxes.uw.diffusive_boundary[index]
    vw_diffusive_boundary = fluxes.vw.diffusive_boundary[index]
    wρ_diffusive_boundary = fluxes.wρ.diffusive_boundary[index]

    uw_total = fluxes.uw.total[index]
    vw_total = fluxes.vw.total[index]
    wρ_total = fluxes.wρ.total[index]

    u_noNN = inv(scaling.u).(sols_noNN[index][1:coarse_size, :])
    v_noNN = inv(scaling.v).(sols_noNN[index][coarse_size+1:2*coarse_size, :])
    ρ_noNN = inv(scaling.ρ).(sols_noNN[index][2*coarse_size+1:3*coarse_size, :])

    uw_noNN = fluxes_noNN.uw.total[index]
    vw_noNN = fluxes_noNN.vw.total[index]
    wρ_noNN = fluxes_noNN.wρ.total[index]

    ulim = (find_min(u_NDE, train_data.data[index].profile.u.unscaled, u_noNN) - 1e-7, find_max(u_NDE, train_data.data[index].profile.u.unscaled, u_noNN) + 1e-7)
    vlim = (find_min(v_NDE, train_data.data[index].profile.v.unscaled, v_noNN) - 1e-7, find_max(v_NDE, train_data.data[index].profile.v.unscaled, v_noNN) + 1e-7)
    ρlim = (find_min(ρ_NDE, train_data.data[index].profile.ρ.unscaled, ρ_noNN), find_max(ρ_NDE, train_data.data[index].profile.ρ.unscaled, ρ_noNN))

    uwlim = (find_min(uw_residual, uw_diffusive_boundary, uw_total, train_data.data[index].flux.uw.column.unscaled) - 1e-7, 
             find_max(uw_residual, uw_diffusive_boundary, uw_total, train_data.data[index].flux.uw.column.unscaled) + 1e-7)
    vwlim = (find_min(vw_residual, vw_diffusive_boundary, vw_total, train_data.data[index].flux.vw.column.unscaled) - 1e-7,
             find_max(vw_residual, vw_diffusive_boundary, vw_total, train_data.data[index].flux.vw.column.unscaled) + 1e-7)
    wρlim = (find_min(wρ_residual, wρ_diffusive_boundary, wρ_total, train_data.data[index].flux.wρ.column.unscaled, wρ_noNN),
             find_max(wρ_residual, wρ_diffusive_boundary, wρ_total, train_data.data[index].flux.wρ.column.unscaled, wρ_noNN))

    Rilim = (find_min(diffusivities.Ri[index], diffusivities.Ri_truth[index], diffusivities_noNN.Ri[index],), 
             find_max(diffusivities.Ri[index], diffusivities.Ri_truth[index], diffusivities_noNN.Ri[index],),)

    diffusivitylim = (find_min(diffusivities.ν[index], diffusivities.κ[index], diffusivities_noNN.ν[index], diffusivities_noNN.κ[index]), 
                      find_max(diffusivities.ν[index], diffusivities.κ[index], diffusivities_noNN.ν[index], diffusivities_noNN.κ[index]),)

    u_truthₙ = @lift train_data.data[index].profile.u.unscaled[:, $n]
    v_truthₙ = @lift train_data.data[index].profile.v.unscaled[:, $n]
    ρ_truthₙ = @lift train_data.data[index].profile.ρ.unscaled[:, $n]

    uw_truthₙ = @lift train_data.data[index].flux.uw.column.unscaled[:, $n]
    vw_truthₙ = @lift train_data.data[index].flux.vw.column.unscaled[:, $n]
    wρ_truthₙ = @lift train_data.data[index].flux.wρ.column.unscaled[:, $n]

    u_NDEₙ = @lift u_NDE[:, $n]
    v_NDEₙ = @lift v_NDE[:, $n]
    ρ_NDEₙ = @lift ρ_NDE[:, $n]

    u_noNNₙ = @lift u_noNN[:, $n]
    v_noNNₙ = @lift v_noNN[:, $n]
    ρ_noNNₙ = @lift ρ_noNN[:, $n]

    uw_residualₙ = @lift uw_residual[:, $n]
    vw_residualₙ = @lift vw_residual[:, $n]
    wρ_residualₙ = @lift wρ_residual[:, $n]

    uw_diffusive_boundaryₙ = @lift uw_diffusive_boundary[:, $n]
    vw_diffusive_boundaryₙ = @lift vw_diffusive_boundary[:, $n]
    wρ_diffusive_boundaryₙ = @lift wρ_diffusive_boundary[:, $n]

    uw_totalₙ = @lift uw_total[:, $n]
    vw_totalₙ = @lift vw_total[:, $n]
    wρ_totalₙ = @lift wρ_total[:, $n]

    uw_noNNₙ = @lift uw_noNN[:, $n]
    vw_noNNₙ = @lift vw_noNN[:, $n]
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

    lines!(axu, u_truthₙ, zC, label="Truth")
    lines!(axu, u_noNNₙ, zC, label="Base closure only")
    lines!(axu, u_NDEₙ, zC, label="NDE")

    lines!(axv, v_truthₙ, zC, label="Truth")
    lines!(axv, v_noNNₙ, zC, label="Base closure only")
    lines!(axv, v_NDEₙ, zC, label="NDE")

    lines!(axρ, ρ_truthₙ, zC, label="Truth")
    lines!(axρ, ρ_noNNₙ, zC, label="Base closure only")
    lines!(axρ, ρ_NDEₙ, zC, label="NDE")

    lines!(axuw, uw_truthₙ, zF, label="Truth")
    lines!(axuw, uw_noNNₙ, zF, label="Base closure only")
    lines!(axuw, uw_diffusive_boundaryₙ, zF, label="Base closure")
    lines!(axuw, uw_residualₙ, zF, label="Residual")
    lines!(axuw, uw_totalₙ, zF, label="NDE")

    lines!(axvw, vw_truthₙ, zF, label="Truth")
    lines!(axvw, vw_noNNₙ, zF, label="Base closure only")
    lines!(axvw, vw_diffusive_boundaryₙ, zF, label="Base closure")
    lines!(axvw, vw_residualₙ, zF, label="Residual")
    lines!(axvw, vw_totalₙ, zF, label="NDE")

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

    axislegend(axu, position=:rb)
    axislegend(axuw, position=:rb)
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
              OptimizationOptimisers.ADAM(1e-4)]

maxiters = [1000, 1000, 1000, 1000]
# optimizers = [OptimizationOptimisers.ADAM()]
# maxiters = [10]

for (epoch, (optimizer, maxiter)) in enumerate(zip(optimizers, maxiters))
    res, loss, sols, fluxes, losses, diffusivities, sols_noNN, fluxes_noNN, diffusivities_noNN = train_NDE(train_data, train_data_plot, NNs, ps_training, ps_baseclosure, st_NN, rng, maxiter=maxiter, solver=VCABM3(), optimizer=optimizer, epoch=epoch)
    u = res.u
    jldsave("$(FILE_DIR)/training_results_$(epoch).jld2"; res, u, loss, sols, fluxes, losses, NNs, st_NN, diffusivities)
    plot_loss(losses, FILE_DIR, epoch=epoch)
    for i in eachindex(field_datasets)
        animate_data(train_data_plot, train_data.scaling, sols, fluxes, diffusivities, sols_noNN, fluxes_noNN, diffusivities_noNN, i, FILE_DIR, epoch=epoch)
    end
    ps_training .= u
end

# rm.(glob("$(FILE_DIR)/intermediate_training_results_*.jld2"))