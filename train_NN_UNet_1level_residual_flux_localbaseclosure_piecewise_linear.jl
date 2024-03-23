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

FILE_DIR = "./training_output/residual_flux_UNet_1level_128_swish_local_diffusivity_piecewise_linear_noclamp_ADAM1e-3_test"
mkpath(FILE_DIR)
@info "$(FILE_DIR)"

LES_FILE_DIRS = [
    "./LES_training/linearTS_dTdz_0.0013_dSdz_-0.0014_QU_-0.0002_QT_3.0e-5_QS_-3.0e-5_T_4.3_S_33.5_f_-0.00012_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.0002_QT_0.0003_QS_-3.0e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.0002_QT_0.0003_QS_-3.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_-0.025_dSdz_-0.0045_QU_-0.0002_QT_-0.0003_QS_-3.0e-5_T_-3.6_S_33.9_f_-0.000125_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
]

BASECLOSURE_FILE_DIR = "./training_output/local_diffusivity_piecewise_linear_noclamp_lossequal_reltol1e-5/training_results_4.jld2"

field_datasets = [FieldDataset(FILE_DIR, backend=OnDisk()) for FILE_DIR in LES_FILE_DIRS]

ps_baseclosure = jldopen(BASECLOSURE_FILE_DIR, "r")["u"]

full_timeframes = [25:length(data["ubar"].times) for data in field_datasets]
train_data = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, full_timeframes)
coarse_size = 32

train_data_plot = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, full_timeframes)

rng = Random.default_rng(123)

layer1 = Chain(Conv(Tuple(5), 10 => 8, swish, pad=SamePad()),
               Conv(Tuple(5), 8 => 8, swish, pad=SamePad()),)

layer2 = Chain(MaxPool(Tuple(2), stride=2), 
               Conv(Tuple(5), 8 => 16, swish, pad=SamePad()),
               Upsample(2),
               Conv(Tuple(5), 16 => 8, swish, pad=SamePad()))

layer3 = Chain(Conv(Tuple(5), 16 => 16, swish, pad=SamePad()),
               Conv(Tuple(5), 16 => 8, swish, pad=SamePad()),
               Conv(Tuple(1), 8 => 5, pad=SamePad()),
               FlattenLayer(),)

layer4 = Chain(Dense(32*5, 128, swish),
               Dense(128, 124))

function concat_two_layers(output, input)
    return cat(output, input, dims=2)
end

NN = Chain(layer1, SkipConnection(layer2, concat_two_layers), layer3, layer4)

ps, st = Lux.setup(rng, NN)

ps = ps |> ComponentArray .|> Float64

ps .= glorot_uniform(rng, Float64, length(ps))

# ps .*= 0

NNs = (; NDE=NN)
ps_training = ComponentArray(NDE=ps)
st_NN = (; NDE=st)

function train_NN(train_data, NNs, ps_training, ps_baseclosure, st_NN, rng; 
                   coarse_size=32, dev=cpu_device(), maxiter=10, optimizer=OptimizationOptimisers.ADAM(0.001), Ri_clamp_lims=(-Inf, Inf))
    train_data = train_data |> dev
    xs = [vcat(data.profile.u.scaled, data.profile.v.scaled, data.profile.T.scaled, data.profile.S.scaled) for data in train_data.data] |> dev
    fluxes_truth = [vcat(data.flux.uw.column.scaled[2:end-1, :], data.flux.vw.column.scaled[2:end-1, :], data.flux.wT.column.scaled[2:end-1, :], data.flux.wS.column.scaled[2:end-1, :]) for data in train_data.data] |> dev
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
                            ) for (data, f_feature, uw_feature, vw_feature, wT_feature, wS_feature) in zip(train_data.data, f_features, uw_features, vw_features, wT_features, wS_features)] |> dev

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

    function predict_diffusive_flux_nondimensional(u, v, ρ, u_hat, v_hat, T_hat, S_hat, p, params, st)
        _uw_diffusive, _vw_diffusive, _wT_diffusive, _wS_diffusive = predict_diffusive_flux(u, v, ρ, u_hat, v_hat, T_hat, S_hat, p, params, st)

        uw_diffusive = params.scaling.uw(params.scaling.u.σ / params.H .* _uw_diffusive)
        vw_diffusive = params.scaling.vw(params.scaling.v.σ / params.H .* _vw_diffusive)
        wT_diffusive = params.scaling.wT(params.scaling.T.σ / params.H .* _wT_diffusive)
        wS_diffusive = params.scaling.wS(params.scaling.S.σ / params.H .* _wS_diffusive)

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

    function total_flux(x, p, params, st)
        u_hat, v_hat, T_hat, S_hat = get_scaled_profiles(x, params)
        u, v, T, S = unscale_profiles(u_hat, v_hat, T_hat, S_hat, params)
        
        ρ = calculate_unscaled_density(T, S)
        ρ_hat = params.scaling.ρ.(ρ)

        uw_residual, vw_residual, wT_residual, wS_residual = predict_residual_flux(u_hat, v_hat, T_hat, S_hat, ρ_hat, p, params, st)
        uw_boundary, vw_boundary, wT_boundary, wS_boundary = predict_boundary_flux(params)
        uw_diffusive, vw_diffusive, wT_diffusive, wS_diffusive = predict_diffusive_flux_nondimensional(u, v, ρ, u_hat, v_hat, T_hat, S_hat, p, params, st)

        uw = uw_residual + uw_boundary + uw_diffusive
        vw = vw_residual + vw_boundary + vw_diffusive
        wT = wT_residual + wT_boundary + wT_diffusive
        wS = wS_residual + wS_boundary + wS_diffusive

        return vcat(uw[2:end-1], vw[2:end-1], wT[2:end-1], wS[2:end-1])
    end

    function predict_fluxes(p)
        preds = [hcat([total_flux(x[:, i], p, param, st_NN) for i in axes(x, 2)]...) for (x, param) in zip(xs, params)]

        return preds
    end

    function loss_NN(p)
        preds = predict_fluxes(p)
        loss = mean([mean((pred .- flux) .^ 2) for (pred, flux) in zip(preds, fluxes_truth)]) + 5e-3 * mean(p .^ 2)

        return loss, preds
    end

    iter = 0

    losses = zeros(maxiter+1)

    wall_clock = [time_ns()]

    callback = function (p, l, pred)
        @printf("%s, Δt %s, iter %d/%d, loss total %6.10e\n",
                Dates.now(), prettytime(1e-9 * (time_ns() - wall_clock[1])), iter, maxiter, l)
        losses[iter+1] = l

        iter += 1
        wall_clock[1] = time_ns()
        return false
    end

    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_NN(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, ps_training)

    @info "Training NN"
    res = Optimization.solve(optprob, optimizer, callback=callback, maxiters=maxiter)

    sols_posttraining = predict_fluxes(res.u)

    losses = (total=losses)

    return res, loss_NN(res.u), sols_posttraining, losses
end

#%%
function plot_loss(losses, FILE_DIR; epoch=1)
    colors = distinguishable_colors(10, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    
    fig = Figure(size=(1000, 600))
    axtotalloss = CairoMakie.Axis(fig[1, 1], title="Total Loss", xlabel="Iterations", ylabel="Loss", yscale=log10)
    axindividualloss = CairoMakie.Axis(fig[1, 2], title="Individual Loss", xlabel="Iterations", ylabel="Loss", yscale=log10)

    lines!(axtotalloss, losses.total, label="Total Loss", color=colors[1])

    # lines!(axindividualloss, losses.u, label="u", color=colors[1])
    # lines!(axindividualloss, losses.v, label="v", color=colors[2])
    # lines!(axindividualloss, losses.T, label="T", color=colors[3])
    # lines!(axindividualloss, losses.S, label="S", color=colors[4])
    # lines!(axindividualloss, losses.ρ, label="ρ", color=colors[5])
    # lines!(axindividualloss, losses.∂u∂z, label="∂u∂z", color=colors[6])
    # lines!(axindividualloss, losses.∂v∂z, label="∂v∂z", color=colors[7])
    # lines!(axindividualloss, losses.∂T∂z, label="∂T∂z", color=colors[8])
    # lines!(axindividualloss, losses.∂S∂z, label="∂S∂z", color=colors[9])
    # lines!(axindividualloss, losses.∂ρ∂z, label="∂ρ∂z", color=colors[10])

    axislegend(axindividualloss, position=:rt)
    save("$(FILE_DIR)/losses_epoch$(epoch).png", fig, px_per_unit=8)
end

#%%
# sols = train_NN(train_data, train_data_plot, NNs, ps_training, ps_baseclosure, st_NN, rng, maxiter=10, optimizer=OptimizationOptimisers.ADAM(1e-4))

epoch=1
res, loss, sols, losses = train_NN(train_data, NNs, ps_training, ps_baseclosure, st_NN, rng, maxiter=1000, optimizer=OptimizationOptimisers.ADAM(1e-3))

u = res.u
jldsave("$(FILE_DIR)/training_results_$(epoch).jld2"; res, u, loss, sols, losses, NNs, st_NN)

epoch += 1
res, loss, sols, losses = train_NN(train_data, NNs, res.u, ps_baseclosure, st_NN, rng, maxiter=1000, optimizer=OptimizationOptimisers.ADAM(5e-4))
jldsave("$(FILE_DIR)/training_results_$(epoch).jld2"; res, u, loss, sols, losses, NNs, st_NN)

epoch += 1
res, loss, sols, losses = train_NN(train_data, NNs, res.u, ps_baseclosure, st_NN, rng, maxiter=1000, optimizer=OptimizationOptimisers.ADAM(2e-4))
jldsave("$(FILE_DIR)/training_results_$(epoch).jld2"; res, u, loss, sols, losses, NNs, st_NN)

epoch += 1
res, loss, sols, losses = train_NN(train_data, NNs, res.u, ps_baseclosure, st_NN, rng, maxiter=1000, optimizer=OptimizationOptimisers.ADAM(1e-4))
jldsave("$(FILE_DIR)/training_results_$(epoch).jld2"; res, u, loss, sols, losses, NNs, st_NN)

#%%
# fig = Figure()
# ax = CairoMakie.Axis(fig[1, 1])
# lines!(ax, sols[3][63:93, 53], train_data.data[3].metadata["zF"][2:end-1], label="NN")
# lines!(ax, train_data.data[3].flux.wT.column.scaled[:, 53], train_data.data[1].metadata["zF"], label="Truth")
# axislegend(ax, position=:lb)
# display(fig)
# #%%