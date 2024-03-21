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

function find_min(a...)
    return minimum(minimum.([a...]))
end
  
function find_max(a...)
    return maximum(maximum.([a...]))
end

FILE_DIR = "./training_output/1CNN_maxpool_stride5_128_swish_local_diffusivity_piecewise_linear_noclamp_VCABM3_reltol1e-5_ADAM1e-3_lossequal_test"
mkpath(FILE_DIR)
@info "$(FILE_DIR)"

LES_FILE_DIRS = [
    "./LES_training/linearTS_dTdz_0.0013_dSdz_-0.0014_QU_-0.0002_QT_3.0e-5_QS_-3.0e-5_T_4.3_S_33.5_f_-0.00012_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.0002_QT_0.0003_QS_-3.0e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.0002_QT_0.0003_QS_-3.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_-0.025_dSdz_-0.0045_QU_-0.0002_QT_-0.0003_QS_-3.0e-5_T_-3.6_S_33.9_f_-0.000125_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
]

BASECLOSURE_FILE_DIR = "./training_output/local_diffusivity_piecewise_linear_noclamp_lossequal/training_results_2.jld2"

field_datasets = [FieldDataset(FILE_DIR, backend=OnDisk()) for FILE_DIR in LES_FILE_DIRS]

ps_baseclosure = jldopen(BASECLOSURE_FILE_DIR, "r")["u"]

full_timeframes = [25:length(data["ubar"].times) for data in field_datasets]
timeframes = [25:5:length(data["ubar"].times) for data in field_datasets]
train_data = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, timeframes)
coarse_size = 32

train_data_plot = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, full_timeframes)

rng = Random.default_rng(123)

NN = Chain(Conv(Tuple(5), 10 => 16, swish),
           MaxPool(Tuple(2), stride=2),
           Conv(Tuple(5), 16 => 32, swish),
           MaxPool(Tuple(2), stride=2),
           FlattenLayer(),
           Dense(32*5, 128, swish),
           Dense(128, 124)
)

ps, st = Lux.setup(rng, NN)

ps = ps |> ComponentArray .|> Float64

ps .*= 0

NNs = (; NDE=NN)
ps_training = ComponentArray(NDE=ps)
st_NN = (; NDE=st)

function train_NDE(train_data, train_data_plot, NNs, ps_training, ps_baseclosure, st_NN, rng; 
                   coarse_size=32, dev=cpu_device(), maxiter=10, optimizer=OptimizationOptimisers.ADAM(0.001), solver=ROCK2(), Ri_clamp_lims=(-Inf, Inf))
    train_data = train_data |> dev
    x₀s = [vcat(data.profile.u.scaled[:, 1], data.profile.v.scaled[:, 1], data.profile.T.scaled[:, 1], data.profile.S.scaled[:, 1]) for data in train_data.data] |> dev
    eos = TEOS10EquationOfState()

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
    wTs_top_scaled = [data.flux.wT.surface.scaled for data in train_data.data] |> dev
    wSs_top_scaled = [data.flux.wS.surface.scaled for data in train_data.data] |> dev

    f_features = construct_gaussian_fourier_features(fs_scaled, coarse_size, rng) |> dev
    uw_features = construct_gaussian_fourier_features(uws_top_scaled, coarse_size, rng) |> dev
    vw_features = construct_gaussian_fourier_features(vws_top_scaled, coarse_size, rng) |> dev
    wT_features = construct_gaussian_fourier_features(wTs_top_scaled, coarse_size, rng) |> dev
    wS_features = construct_gaussian_fourier_features(wSs_top_scaled, coarse_size, rng) |> dev

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
                                 wT = (scaled = (top=data.flux.wT.surface.scaled, bottom=data.flux.wT.bottom.scaled),
                                       unscaled = (top=data.flux.wT.surface.unscaled, bottom=data.flux.wT.bottom.unscaled)),
                                 wS = (scaled = (top=data.flux.wS.surface.scaled, bottom=data.flux.wS.bottom.scaled),
                                       unscaled = (top=data.flux.wS.surface.unscaled, bottom=data.flux.wS.bottom.unscaled)),                                    
                            scaling = train_data.scaling,
                            feature = (f=f_feature, uw=uw_feature, vw=vw_feature, wT=wT_feature, wS=wS_feature),
                            ) for (data, plot_data, f_feature, uw_feature, vw_feature, wT_feature, wS_feature) in zip(train_data.data, train_data_plot.data, f_features, uw_features, vw_features, wT_features, wS_features)] |> dev

    function predict_residual_flux(u_hat, v_hat, T_hat, S_hat, ρ_hat, p, params, st)
        x′ = reshape(hcat(u_hat, v_hat, T_hat, S_hat, ρ_hat, params.feature.uw, params.feature.vw, params.feature.wT, params.feature.wS, params.feature.f), coarse_size, 10, 1)
        
        interior_size = coarse_size-1
        NN_pred = first(NNs.NDE(x′, p.NDE, st.NDE))

        uw = vcat(0, NN_pred[1:interior_size, 1], 0)
        vw = vcat(0, NN_pred[interior_size+1:2*interior_size, 1], 0)
        wT = vcat(0, NN_pred[2*interior_size+1:3*interior_size, 1], 0)
        wS = vcat(0, NN_pred[3*interior_size+1:4*interior_size, 1], 0)

        return uw, vw, wT, wS
    end

    function predict_residual_flux_dimensional(u_hat, v_hat, T_hat, S_hat, ρ_hat, p, params, st)
        uw_hat, vw_hat, wT_hat, wS_hat = predict_residual_flux(u_hat, v_hat, T_hat, S_hat, ρ_hat, p, params, st)
        uw = inv(params.scaling.uw).(uw_hat)
        vw = inv(params.scaling.vw).(vw_hat)
        wT = inv(params.scaling.wT).(wT_hat)
        wS = inv(params.scaling.wS).(wS_hat)

        uw = uw .- uw[1]
        vw = vw .- vw[1]
        wT = wT .- wT[1]
        wS = wS .- wS[1]

        return uw, vw, wT, wS
    end

    function predict_boundary_flux(params)
        uw = vcat(fill(params.uw.scaled.bottom, coarse_size), params.uw.scaled.top)
        vw = vcat(fill(params.vw.scaled.bottom, coarse_size), params.vw.scaled.top)
        wT = vcat(fill(params.wT.scaled.bottom, coarse_size), params.wT.scaled.top)
        wS = vcat(fill(params.wS.scaled.bottom, coarse_size), params.wS.scaled.top)

        return uw, vw, wT, wS
    end

    function predict_diffusivities(Ris, p)
        νs = local_Ri_ν_piecewise_linear.(Ris, ps_baseclosure.ν₁, ps_baseclosure.m)
        κs = local_Ri_κ_piecewise_linear.(νs, ps_baseclosure.Pr)
        return νs, κs
    end

    function predict_diffusive_flux(u, v, ρ, u_hat, v_hat, T_hat, S_hat, p, params, st)
        Ris = calculate_Ri(u, v, ρ, params.Dᶠ, params.g, eos.reference_density, clamp_lims=Ri_clamp_lims)
        νs, κs = predict_diffusivities(Ris, p)

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

    function predict_diffusive_boundary_flux_dimensional(u, v, ρ, u_hat, v_hat, T_hat, S_hat, p, params, st)
        _uw_diffusive, _vw_diffusive, _wT_diffusive, _wS_diffusive = predict_diffusive_flux(u, v, ρ, u_hat, v_hat, T_hat, S_hat, p, params, st)
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

    get_u(x::Vector, params) = x[1:params.coarse_size]
    get_v(x::Vector, params) = x[params.coarse_size+1:2*params.coarse_size]
    get_T(x::Vector, params) = x[2*params.coarse_size+1:3*params.coarse_size]
    get_S(x::Vector, params) = x[3*params.coarse_size+1:4*params.coarse_size]

    get_u(x::Matrix, params) = @view x[1:params.coarse_size, :]
    get_v(x::Matrix, params) = @view x[params.coarse_size+1:2*params.coarse_size, :]
    get_T(x::Matrix, params) = @view x[2*params.coarse_size+1:3*params.coarse_size, :]
    get_S(x::Matrix, params) = @view x[3*params.coarse_size+1:4*params.coarse_size, :]

    function get_scaled_profiles(x, params)
        u_hat = get_u(x, params)
        v_hat = get_v(x, params)
        T_hat = get_T(x, params)
        S_hat = get_S(x, params)

        return u_hat, v_hat, T_hat, S_hat
    end

    function unscale_profiles(u_hat, v_hat, T_hat, S_hat, params)
        u = inv(params.scaling.u).(u_hat)
        v = inv(params.scaling.v).(v_hat)
        T = inv(params.scaling.T).(T_hat)
        S = inv(params.scaling.S).(S_hat)

        return u, v, T, S
    end

    function calculate_unscaled_density(T, S)
        return TEOS10.ρ.(T, S, 0, Ref(eos))
    end

    function calculate_scaled_profile_gradients(u, v, T, S, ρ, params)
        Dᶠ = params.Dᶠ

        ∂u∂z_hat = params.scaling.∂u∂z.(Dᶠ * u)
        ∂v∂z_hat = params.scaling.∂v∂z.(Dᶠ * v)
        ∂T∂z_hat = params.scaling.∂T∂z.(Dᶠ * T)
        ∂S∂z_hat = params.scaling.∂S∂z.(Dᶠ * S)
        ∂ρ∂z_hat = params.scaling.∂ρ∂z.(Dᶠ * ρ)

        return ∂u∂z_hat, ∂v∂z_hat, ∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat
    end

    function NDE(x, p, t, params, st)
        f = params.f
        Dᶜ_hat = params.Dᶜ_hat
        scaling = params.scaling
        τ, H = params.τ, params.H

        u_hat, v_hat, T_hat, S_hat = get_scaled_profiles(x, params)
        u, v, T, S = unscale_profiles(u_hat, v_hat, T_hat, S_hat, params)
        
        ρ = calculate_unscaled_density(T, S)
        ρ_hat = params.scaling.ρ.(ρ)

        uw_residual, vw_residual, wT_residual, wS_residual = predict_residual_flux(u_hat, v_hat, T_hat, S_hat, ρ_hat, p, params, st)
        uw_boundary, vw_boundary, wT_boundary, wS_boundary = predict_boundary_flux(params)
        uw_diffusive, vw_diffusive, wT_diffusive, wS_diffusive = predict_diffusive_flux(u, v, ρ, u_hat, v_hat, T_hat, S_hat, p, params, st)

        du = -τ / H^2 .* (Dᶜ_hat * uw_diffusive) .- τ / H * scaling.uw.σ / scaling.u.σ .* (Dᶜ_hat * (uw_boundary .+ uw_residual)) .+ f * τ ./ scaling.u.σ .* v
        dv = -τ / H^2 .* (Dᶜ_hat * vw_diffusive) .- τ / H * scaling.vw.σ / scaling.v.σ .* (Dᶜ_hat * (vw_boundary .+ vw_residual)) .- f * τ ./ scaling.v.σ .* u
        dT = -τ / H^2 .* (Dᶜ_hat * wT_diffusive) .- τ / H * scaling.wT.σ / scaling.T.σ .* (Dᶜ_hat * (wT_boundary .+ wT_residual))
        dS = -τ / H^2 .* (Dᶜ_hat * wS_diffusive) .- τ / H * scaling.wS.σ / scaling.S.σ .* (Dᶜ_hat * (wS_boundary .+ wS_residual))

        return vcat(du, dv, dT, dS)
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
        u_hats = [get_u(pred, param) for (pred, param) in zip(preds, params)]
        v_hats = [get_v(pred, param) for (pred, param) in zip(preds, params)]
        T_hats = [get_T(pred, param) for (pred, param) in zip(preds, params)]
        S_hats = [get_S(pred, param) for (pred, param) in zip(preds, params)]
        
        us = [inv(param.scaling.u).(u_hat) for (u_hat, param) in zip(u_hats, params)]
        vs = [inv(param.scaling.v).(v_hat) for (v_hat, param) in zip(v_hats, params)]
        Ts = [inv(param.scaling.T).(T_hat) for (T_hat, param) in zip(T_hats, params)]
        Ss = [inv(param.scaling.S).(S_hat) for (S_hat, param) in zip(S_hats, params)]

        ρs = [calculate_unscaled_density(T, S) for (T, S) in zip(Ts, Ss)]
        ρ_hats = [param.scaling.ρ.(ρ) for (ρ, param) in zip(ρs, params)]

        ∂u∂z_hats = [hcat([param.scaling.∂u∂z.(param.Dᶠ * @view(u[:, i])) for i in axes(u, 2)]...) for (param, u) in zip(params, us)]
        ∂v∂z_hats = [hcat([param.scaling.∂v∂z.(param.Dᶠ * @view(v[:, i])) for i in axes(v, 2)]...) for (param, v) in zip(params, vs)]
        ∂T∂z_hats = [hcat([param.scaling.∂T∂z.(param.Dᶠ * @view(T[:, i])) for i in axes(T, 2)]...) for (param, T) in zip(params, Ts)]
        ∂S∂z_hats = [hcat([param.scaling.∂S∂z.(param.Dᶠ * @view(S[:, i])) for i in axes(S, 2)]...) for (param, S) in zip(params, Ss)]
        ∂ρ∂z_hats = [hcat([param.scaling.∂ρ∂z.(param.Dᶠ * @view(ρ[:, i])) for i in axes(ρ, 2)]...) for (param, ρ) in zip(params, ρs)]

        return u_hats, v_hats, T_hats, S_hats, ρ_hats, ∂u∂z_hats, ∂v∂z_hats, ∂T∂z_hats, ∂S∂z_hats, ∂ρ∂z_hats, preds
    end

    function predict_losses(u, v, T, S, ρ, ∂u∂z, ∂v∂z, ∂T∂z, ∂S∂z, ∂ρ∂z, loss_scaling=(; u=1, v=1, T=1, S=1, ρ=1, ∂u∂z=1, ∂v∂z=1, ∂T∂z=1, ∂S∂z=1, ∂ρ∂z=1))
        u_loss = loss_scaling.u * mean(mean.([(data.profile.u.scaled .- u).^2 for (data, u) in zip(train_data.data, u)]))
        v_loss = loss_scaling.v * mean(mean.([(data.profile.v.scaled .- v).^2 for (data, v) in zip(train_data.data, v)]))
        T_loss = loss_scaling.T * mean(mean.([(data.profile.T.scaled .- T).^2 for (data, T) in zip(train_data.data, T)]))
        S_loss = loss_scaling.S * mean(mean.([(data.profile.S.scaled .- S).^2 for (data, S) in zip(train_data.data, S)]))
        ρ_loss = loss_scaling.ρ * mean(mean.([(data.profile.ρ.scaled .- ρ).^2 for (data, ρ) in zip(train_data.data, ρ)]))

        ∂u∂z_loss = loss_scaling.∂u∂z * mean(mean.([(data.profile.∂u∂z.scaled .- ∂u∂z).^2 for (data, ∂u∂z) in zip(train_data.data, ∂u∂z)]))
        ∂v∂z_loss = loss_scaling.∂v∂z * mean(mean.([(data.profile.∂v∂z.scaled .- ∂v∂z).^2 for (data, ∂v∂z) in zip(train_data.data, ∂v∂z)]))
        ∂T∂z_loss = loss_scaling.∂T∂z * mean(mean.([(data.profile.∂T∂z.scaled .- ∂T∂z).^2 for (data, ∂T∂z) in zip(train_data.data, ∂T∂z)]))
        ∂S∂z_loss = loss_scaling.∂S∂z * mean(mean.([(data.profile.∂S∂z.scaled .- ∂S∂z).^2 for (data, ∂S∂z) in zip(train_data.data, ∂S∂z)]))
        ∂ρ∂z_loss = loss_scaling.∂ρ∂z * mean(mean.([(data.profile.∂ρ∂z.scaled .- ∂ρ∂z).^2 for (data, ∂ρ∂z) in zip(train_data.data, ∂ρ∂z)]))

        return (u=u_loss, v=v_loss, T=T_loss, S=S_loss, ρ=ρ_loss, ∂u∂z=∂u∂z_loss, ∂v∂z=∂v∂z_loss, ∂T∂z=∂T∂z_loss, ∂S∂z=∂S∂z_loss, ∂ρ∂z=∂ρ∂z_loss)
    end

    function compute_loss_prefactor(u_loss, v_loss, T_loss, S_loss, ρ_loss, ∂u∂z_loss, ∂v∂z_loss, ∂T∂z_loss, ∂S∂z_loss, ∂ρ∂z_loss)
        ρ_prefactor = 1
        T_prefactor = ρ_loss / T_loss
        S_prefactor = ρ_loss / S_loss
        u_prefactor = ρ_loss / u_loss
        v_prefactor = ρ_loss / v_loss

        ∂ρ∂z_prefactor = 1
        ∂T∂z_prefactor = ∂ρ∂z_loss / ∂T∂z_loss
        ∂S∂z_prefactor = ∂ρ∂z_loss / ∂S∂z_loss
        ∂u∂z_prefactor = ∂ρ∂z_loss / ∂u∂z_loss
        ∂v∂z_prefactor = ∂ρ∂z_loss / ∂v∂z_loss

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

    function compute_loss_prefactor(p)
        u, v, T, S, ρ, ∂u∂z, ∂v∂z, ∂T∂z, ∂S∂z, ∂ρ∂z, _ = predict_scaled_profiles(p)
        losses = predict_losses(u, v, T, S, ρ, ∂u∂z, ∂v∂z, ∂T∂z, ∂S∂z, ∂ρ∂z)
        return compute_loss_prefactor(losses...)
    end

    @info "Computing prefactor for losses"

    losses_prefactor = compute_loss_prefactor(ps_training)

    function loss_NDE(p)
        u, v, T, S, ρ, ∂u∂z, ∂v∂z, ∂T∂z, ∂S∂z, ∂ρ∂z, preds = predict_scaled_profiles(p)
        individual_loss = predict_losses(u, v, T, S, ρ, ∂u∂z, ∂v∂z, ∂T∂z, ∂S∂z, ∂ρ∂z, losses_prefactor)
        loss = sum(values(individual_loss))

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
            u_hat, v_hat, T_hat, S_hat = get_scaled_profiles(sol[:, t], params[i])
            u, v, T, S = unscale_profiles(u_hat, v_hat, T_hat, S_hat, params[i])

            ρ = calculate_unscaled_density(T, S)
            ρ_hat = params[i].scaling.ρ.(ρ)

            uw_diffusive_boundary, vw_diffusive_boundary, wT_diffusive_boundary, wS_diffusive_boundary = predict_diffusive_boundary_flux_dimensional(u, v, ρ, u_hat, v_hat, T_hat, S_hat, res.u, params[i], st_NN)
            uw_residual, vw_residual, wT_residual, wS_residual = predict_residual_flux_dimensional(u_hat, v_hat, T_hat, S_hat, ρ_hat, res.u, params[i], st_NN)

            Ris = calculate_Ri(u, v, ρ, params[i].Dᶠ, params[i].g, eos.reference_density, clamp_lims=Ri_clamp_lims)

            Ris_truth = calculate_Ri(train_data_plot.data[i].profile.u.unscaled[:, t], 
                                     train_data_plot.data[i].profile.v.unscaled[:, t],
                                     train_data_plot.data[i].profile.ρ.unscaled[:, t], 
                                     params[i].Dᶠ, params[i].g, eos.reference_density, clamp_lims=Ri_clamp_lims)

            νs, κs = predict_diffusivities(Ris, res.u)

            uw_residual_posttraining[i][:, t] .= uw_residual
            vw_residual_posttraining[i][:, t] .= vw_residual
            wT_residual_posttraining[i][:, t] .= wT_residual
            wS_residual_posttraining[i][:, t] .= wS_residual

            uw_diffusive_boundary_posttraining[i][:, t] .= uw_diffusive_boundary
            vw_diffusive_boundary_posttraining[i][:, t] .= vw_diffusive_boundary
            wT_diffusive_boundary_posttraining[i][:, t] .= wT_diffusive_boundary
            wS_diffusive_boundary_posttraining[i][:, t] .= wS_diffusive_boundary

            νs_posttraining[i][:, t] .= νs
            κs_posttraining[i][:, t] .= κs

            Ri_posttraining[i][:, t] .= Ris
            Ri_truth[i][:, t] .= Ris_truth
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

    losses = (total=losses, u=u_losses, v=v_losses, T=T_losses, S=S_losses, ρ=ρ_losses, ∂u∂z=∂u∂z_losses, ∂v∂z=∂v∂z_losses, ∂T∂z=∂T∂z_losses, ∂S∂z=∂S∂z_losses, ∂ρ∂z=∂ρ∂z_losses)

    return res, loss_NDE(res.u), sols_posttraining, flux_posttraining, losses, diffusivities_posttraining
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

    u_NDE = inv(train_data.scaling.u).(sols[index][1:coarse_size, :])
    v_NDE = inv(train_data.scaling.v).(sols[index][coarse_size+1:2*coarse_size, :])
    T_NDE = inv(train_data.scaling.T).(sols[index][2*coarse_size+1:3*coarse_size, :])
    S_NDE = inv(train_data.scaling.S).(sols[index][3*coarse_size+1:4*coarse_size, :])
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
    xlims!(axRi, Rilim)
    xlims!(axdiffusivity, diffusivitylim)

    CairoMakie.record(fig, "$(FILE_DIR)/training_$(index)_epoch$(epoch).mp4", 1:Nt, framerate=15) do nn
        n[] = nn
    end
end

epoch = 1

res, loss, sols, fluxes, losses, diffusivities = train_NDE(train_data, train_data_plot, NNs, ps_training, ps_baseclosure, st_NN, rng, maxiter=200, solver=VCABM3(), optimizer=OptimizationOptimisers.ADAM(1e-3))

u = res.u
jldsave("$(FILE_DIR)/training_results_$(epoch).jld2"; res, u, loss, sols, fluxes, losses, NNs, st_NN, diffusivities)
plot_loss(losses, FILE_DIR, epoch=epoch)
for i in eachindex(field_datasets)
    animate_data(train_data_plot, sols, fluxes, diffusivities, i, FILE_DIR, epoch=epoch)
end

epoch += 1

res, loss, sols, fluxes, losses, diffusivities = train_NDE(train_data, train_data_plot, NNs, res.u, ps_baseclosure, st_NN, rng, maxiter=200, solver=VCABM3(), optimizer=OptimizationOptimisers.ADAM(1e-3))
u = res.u
jldsave("$(FILE_DIR)/training_results_$(epoch).jld2"; res, u, loss, sols, fluxes, losses, NNs, st_NN, diffusivities)
plot_loss(losses, FILE_DIR, epoch=epoch)
for i in eachindex(field_datasets)
    animate_data(train_data_plot, sols, fluxes, diffusivities, i, FILE_DIR, epoch=epoch)
end

epoch += 1

res, loss, sols, fluxes, losses, diffusivities = train_NDE(train_data, train_data_plot, NNs, res.u, ps_baseclosure, st_NN, rng, maxiter=200, solver=VCABM3(), optimizer=OptimizationOptimisers.ADAM(5e-4))

u = res.u
jldsave("$(FILE_DIR)/training_results_$(epoch).jld2"; res, u, loss, sols, fluxes, losses, NNs, st_NN, diffusivities)
plot_loss(losses, FILE_DIR, epoch=epoch)
for i in eachindex(field_datasets)
    animate_data(train_data_plot, sols, fluxes, diffusivities, i, FILE_DIR, epoch=epoch)
end

epoch += 1

res, loss, sols, fluxes, losses, diffusivities = train_NDE(train_data, train_data_plot, NNs, res.u, ps_baseclosure, st_NN, rng, maxiter=100, solver=VCABM3(), Ri_clamp_lims=(-20, 20), optimizer=OptimizationOptimisers.ADAM(5e-4))

u = res.u
jldsave("$(FILE_DIR)/training_results_$(epoch).jld2"; res, u, loss, sols, fluxes, losses, NNs, st_NN, diffusivities)
plot_loss(losses, FILE_DIR, epoch=epoch)
for i in eachindex(field_datasets)
    animate_data(train_data_plot, sols, fluxes, diffusivities, i, FILE_DIR, epoch=epoch)
end


