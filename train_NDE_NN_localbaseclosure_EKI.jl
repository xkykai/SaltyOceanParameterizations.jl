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
using ModelingToolkit
using LinearAlgebra
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses

function find_min(a...)
    return minimum(minimum.([a...]))
end
  
function find_max(a...)
    return maximum(maximum.([a...]))
end

FILE_DIR = "./training_output/NN_local_diffusivity_NDE_gradient_relu_noclamp_ROCK4_EKI_fast_test"
mkpath(FILE_DIR)

LES_FILE_DIRS = [
    "./LES_training/linearTS_dTdz_0.0013_dSdz_-0.0014_QU_-0.0002_QT_3.0e-5_QS_-3.0e-5_T_4.3_S_33.5_f_-0.00012_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.0002_QT_0.0003_QS_-3.0e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.0002_QT_0.0003_QS_-3.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_-0.025_dSdz_-0.0045_QU_-0.0002_QT_-0.0003_QS_-3.0e-5_T_-3.6_S_33.9_f_-0.000125_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
]

BASECLOSURE_FILE_DIR = "./training_output/local_diffusivity_NDE_gradient_relu_noclamp/training_results_2.jld2"

field_datasets = [FieldDataset(FILE_DIR, backend=OnDisk()) for FILE_DIR in LES_FILE_DIRS]

file_baseclosure = jldopen(BASECLOSURE_FILE_DIR, "r")
baseclosure_NN = file_baseclosure["NN"]
st_baseclosure = file_baseclosure["st_NN"]
ps_baseclosure = file_baseclosure["res"].u
close(file_baseclosure)

full_timeframes = [1:length(data["ubar"].times) for data in field_datasets]
timeframes = [5:5:length(data["ubar"].times) for data in field_datasets]
train_data = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, timeframes)
coarse_size = 32

train_data_plot = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, full_timeframes)

dev = cpu_device()

rng = Random.default_rng(123)

uw_NN = Chain(Dense(164, 64, leakyrelu), Dense(64, 31))
vw_NN = Chain(Dense(164, 64, leakyrelu), Dense(64, 31))
wT_NN = Chain(Dense(164, 64, leakyrelu), Dense(64, 31))
wS_NN = Chain(Dense(164, 64, leakyrelu), Dense(64, 31))

# uw_NN = Chain(Dense(164, 4, leakyrelu), Dense(4, 31))
# vw_NN = Chain(Dense(164, 4, leakyrelu), Dense(4, 31))
# wT_NN = Chain(Dense(164, 4, leakyrelu), Dense(4, 31))
# wS_NN = Chain(Dense(164, 4, leakyrelu), Dense(4, 31))

ps_uw, st_uw = Lux.setup(rng, uw_NN)
ps_vw, st_vw = Lux.setup(rng, vw_NN)
ps_wT, st_wT = Lux.setup(rng, wT_NN)
ps_wS, st_wS = Lux.setup(rng, wS_NN)

ps_uw = ps_uw |> ComponentArray .|> Float64
ps_vw = ps_vw |> ComponentArray .|> Float64
ps_wT = ps_wT |> ComponentArray .|> Float64
ps_wS = ps_wS |> ComponentArray .|> Float64

ps_uw .*= 1e-6
ps_vw .*= 1e-6
ps_wT .*= 1e-6
ps_wS .*= 1e-6

st_uw = st_uw
st_vw = st_vw
st_wT = st_wT
st_wS = st_wS

NNs = (uw=uw_NN, vw=vw_NN, wT=wT_NN, wS=wS_NN, baseclosure=baseclosure_NN)
ps_NN = ComponentArray(uw=ps_uw, vw=ps_vw, wT=ps_wT, wS=ps_wS, baseclosure=ps_baseclosure)
ax_ps_NN = getaxes(ps_NN)
st_NN = (uw=st_uw, vw=st_vw, wT=st_wT, wS=st_wS, baseclosure=st_baseclosure)

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
                                    scaling = merge(train_data.scaling, (; diffusivity=DiffusivityScaling()))
            ) for data in train_data.data] |> dev

x₀s = [vcat(data.profile.u.scaled[:, 1], data.profile.v.scaled[:, 1], data.profile.T.scaled[:, 1], data.profile.S.scaled[:, 1]) for data in train_data.data] |> dev
eos = TEOS10EquationOfState()

function NDE!(dx, x, p, t, params, st)
    f = params.f
    Dᶜ_hat = params.Dᶜ_hat
    Dᶠ_hat = params.Dᶠ_hat
    scaling = params.scaling
    τ, H = params.τ, params.H

    u_hat = x[1:params.coarse_size]
    v_hat = x[params.coarse_size+1:2*params.coarse_size]
    T_hat = x[2*params.coarse_size+1:3*params.coarse_size]
    S_hat = x[3*params.coarse_size+1:4*params.coarse_size]

    u = inv(params.scaling.u).(u_hat)
    v = inv(params.scaling.v).(v_hat)
    T = inv(params.scaling.T).(T_hat)
    S = inv(params.scaling.S).(S_hat)
    
    ρ = TEOS10.ρ.(T, S, 0, Ref(eos))
    ρ_hat = params.scaling.ρ.(ρ)

    du = @view dx[1:params.coarse_size]
    dv = @view dx[params.coarse_size+1:2*params.coarse_size]
    dT = @view dx[2*params.coarse_size+1:3*params.coarse_size]
    dS = @view dx[3*params.coarse_size+1:4*params.coarse_size]

    x′ = vcat(u_hat, v_hat, T_hat, S_hat, ρ_hat, params.uw.scaled.top, params.vw.scaled.top, params.wT.scaled.top, params.wS.scaled.top)
    uw_residual = vcat(0, first(NNs.uw(x′, p.uw, st.uw)), 0)
    vw_residual = vcat(0, first(NNs.vw(x′, p.vw, st.vw)), 0)
    wT_residual = vcat(0, first(NNs.wT(x′, p.wT, st.wT)), 0)
    wS_residual = vcat(0, first(NNs.wS(x′, p.wS, st.wS)), 0)

    Ris = calculate_Ri(u, v, ρ, params.Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf))
    diffusivities = [params.scaling.diffusivity(first(NNs.baseclosure([Ri], p.baseclosure, st.baseclosure))) for Ri in Ris]

    νs = [diffusivity[1] for diffusivity in diffusivities]
    κs = [diffusivity[2] for diffusivity in diffusivities]

    ∂u∂z_hat = Dᶠ_hat * u_hat
    ∂v∂z_hat = Dᶠ_hat * v_hat
    ∂T∂z_hat = Dᶠ_hat * T_hat
    ∂S∂z_hat = Dᶠ_hat * S_hat

    uw_diffusive = -νs .* ∂u∂z_hat
    vw_diffusive = -νs .* ∂v∂z_hat
    wT_diffusive = -κs .* ∂T∂z_hat
    wS_diffusive = -κs .* ∂S∂z_hat

    uw_boundary = vcat(fill(params.uw.scaled.bottom, coarse_size), params.uw.scaled.top)
    vw_boundary = vcat(fill(params.vw.scaled.bottom, coarse_size), params.vw.scaled.top)
    wT_boundary = vcat(fill(params.wT.scaled.bottom, coarse_size), params.wT.scaled.top)
    wS_boundary = vcat(fill(params.wS.scaled.bottom, coarse_size), params.wS.scaled.top)

    du .= -τ / H^2 .* (Dᶜ_hat * uw_diffusive) .- τ / H * scaling.uw.σ / scaling.u.σ .* (Dᶜ_hat * (uw_boundary .+ uw_residual)) .+ f * τ ./ scaling.u.σ .* v
    dv .= -τ / H^2 .* (Dᶜ_hat * vw_diffusive) .- τ / H * scaling.vw.σ / scaling.v.σ .* (Dᶜ_hat * (vw_boundary .+ vw_residual)) .- f * τ ./ scaling.v.σ .* u
    dT .= -τ / H^2 .* (Dᶜ_hat * wT_diffusive) .- τ / H * scaling.wT.σ / scaling.T.σ .* (Dᶜ_hat * (wT_boundary .+ wT_residual))
    dS .= -τ / H^2 .* (Dᶜ_hat * wS_diffusive) .- τ / H * scaling.wS.σ / scaling.S.σ .* (Dᶜ_hat * (wS_boundary .+ wS_residual))

    return dx
end

# probs = [ODEProblem((dx, x, p′, t) -> NDE!(dx, x, p′, t, param, st_NN), x₀, (param.scaled_time[1], param.scaled_time[end]), ps_NN) for (x₀, param) in zip(x₀s, params)]
# sols = [Array(solve(prob, ROCK4(), saveat=param.scaled_time, reltol=1e-3)) for (param, prob) in zip(params, probs)]

_params = [(                   f = data.metadata["coriolis_parameter"],
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
                                 Δ = (data.metadata["zC"][2] - data.metadata["zC"][1]),
                             Δ_hat = (data.metadata["zC"][2] - data.metadata["zC"][1]) / data.metadata["original_grid"].Lz,
                                uw = (scaled = (top=data.flux.uw.surface.scaled, bottom=data.flux.uw.bottom.scaled),
                                    unscaled = (top=data.flux.uw.surface.unscaled, bottom=data.flux.uw.bottom.unscaled)),
                                vw = (scaled = (top=data.flux.vw.surface.scaled, bottom=data.flux.vw.bottom.scaled),
                                    unscaled = (top=data.flux.vw.surface.unscaled, bottom=data.flux.vw.bottom.unscaled)),
                                wT = (scaled = (top=data.flux.wT.surface.scaled, bottom=data.flux.wT.bottom.scaled),
                                    unscaled = (top=data.flux.wT.surface.unscaled, bottom=data.flux.wT.bottom.unscaled)),
                                wS = (scaled = (top=data.flux.wS.surface.scaled, bottom=data.flux.wS.bottom.scaled),
                                    unscaled = (top=data.flux.wS.surface.unscaled, bottom=data.flux.wS.bottom.unscaled)),           
                                     scaling = merge(train_data.scaling, (; diffusivity=DiffusivityScaling())),
                                       cache = (u=zeros(coarse_size), v=zeros(coarse_size), T=zeros(coarse_size), S=zeros(coarse_size), ρ=zeros(coarse_size), ρ_hat=zeros(coarse_size), 
                                                # x′=ComponentArray(u=zeros(coarse_size), v=zeros(coarse_size), T=zeros(coarse_size), S=zeros(coarse_size), ρ=zeros(coarse_size), 
                                                #                   uw=zeros(1), vw=zeros(1), wT=zeros(1), wS=zeros(1)),
                                                x′=zeros(5*coarse_size+4),
                                                Ri=zeros(coarse_size+1), ν=zeros(coarse_size+1), κ=zeros(coarse_size+1),
                                                ∂b∂z=zeros(coarse_size+1), ∂u∂z=zeros(coarse_size+1), ∂v∂z=zeros(coarse_size+1),
                                                uw_diffusive=zeros(coarse_size+1), vw_diffusive=zeros(coarse_size+1), wT_diffusive=zeros(coarse_size+1), wS_diffusive=zeros(coarse_size+1),
                                                uw_residual=zeros(coarse_size+1), vw_residual=zeros(coarse_size+1), wT_residual=zeros(coarse_size+1), wS_residual=zeros(coarse_size+1),
                                                u_RHS_diffusive=zeros(coarse_size), v_RHS_diffusive=zeros(coarse_size), T_RHS_diffusive=zeros(coarse_size), S_RHS_diffusive=zeros(coarse_size),
                                                u_RHS_residual=zeros(coarse_size), v_RHS_residual=zeros(coarse_size), T_RHS_residual=zeros(coarse_size), S_RHS_residual=zeros(coarse_size),)
            ) for data in train_data.data] |> dev

function NDE_opt(x, p, t, params, NNs, st)
    eos = TEOS10EquationOfState()
    coarse_size = params.coarse_size
    f = params.f
    Δ, Δ_hat = params.Δ, params.Δ_hat
    scaling = params.scaling
    τ, H = params.τ, params.H
    u, v, T, S, ρ, ρ_hat, x′ = params.cache.u, params.cache.v, params.cache.T, params.cache.S, params.cache.ρ, params.cache.ρ_hat, params.cache.x′
    Ri, ν, κ = params.cache.Ri, params.cache.ν, params.cache.κ
    ∂b∂z, ∂u∂z, ∂v∂z, g, ρ₀ = params.cache.∂b∂z, params.cache.∂u∂z, params.cache.∂v∂z, params.g, eos.reference_density
    uw_diffusive, vw_diffusive, wT_diffusive, wS_diffusive = params.cache.uw_diffusive, params.cache.vw_diffusive, params.cache.wT_diffusive, params.cache.wS_diffusive
    uw_residual, vw_residual, wT_residual, wS_residual = params.cache.uw_residual, params.cache.vw_residual, params.cache.wT_residual, params.cache.wS_residual
    u_RHS_diffusive, v_RHS_diffusive, T_RHS_diffusive, S_RHS_diffusive = params.cache.u_RHS_diffusive, params.cache.v_RHS_diffusive, params.cache.T_RHS_diffusive, params.cache.S_RHS_diffusive
    u_RHS_residual, v_RHS_residual, T_RHS_residual, S_RHS_residual = params.cache.u_RHS_residual, params.cache.v_RHS_residual, params.cache.T_RHS_residual, params.cache.S_RHS_residual

    u_hat = @view x[1:coarse_size]
    v_hat = @view x[coarse_size+1:2*coarse_size]
    T_hat = @view x[2*coarse_size+1:3*coarse_size]
    S_hat = @view x[3*coarse_size+1:4*coarse_size]

    u .= inv(params.scaling.u).(u_hat)
    v .= inv(params.scaling.v).(v_hat)
    T .= inv(params.scaling.T).(T_hat)
    S .= inv(params.scaling.S).(S_hat)

    ρ .= TEOS10.ρ.(T, S, 0, Ref(eos))
    ρ_hat .= params.scaling.ρ.(ρ)

    uw_residual_nottop = @view uw_residual[1:end-1]
    vw_residual_nottop = @view vw_residual[1:end-1]
    wT_residual_nottop = @view wT_residual[1:end-1]
    wS_residual_nottop = @view wS_residual[1:end-1]

    uw_residual_interior = @view uw_residual[2:end-1]
    vw_residual_interior = @view vw_residual[2:end-1]
    wT_residual_interior = @view wT_residual[2:end-1]
    wS_residual_interior = @view wS_residual[2:end-1]

    uw_residual[end] = params.uw.scaled.top
    vw_residual[end] = params.vw.scaled.top
    wT_residual[end] = params.wT.scaled.top
    wS_residual[end] = params.wS.scaled.top

    uw_residual_nottop .= params.uw.scaled.bottom
    vw_residual_nottop .= params.vw.scaled.bottom
    wT_residual_nottop .= params.wT.scaled.bottom
    wS_residual_nottop .= params.wS.scaled.bottom

    x′u = @view x′[1:coarse_size]
    x′v = @view x′[coarse_size+1:2*coarse_size]
    x′T = @view x′[2*coarse_size+1:3*coarse_size]
    x′S = @view x′[3*coarse_size+1:4*coarse_size]
    x′ρ = @view x′[4*coarse_size+1:5*coarse_size]

    x′u .= u_hat
    x′v .= v_hat
    x′T .= T_hat
    x′S .= S_hat
    x′ρ .= ρ_hat

    x′[5*coarse_size+1] = params.uw.scaled.top
    x′[5*coarse_size+2] = params.vw.scaled.top
    x′[5*coarse_size+3] = params.wT.scaled.top
    x′[5*coarse_size+4] = params.wS.scaled.top

    uw_residual_interior .+= first(NNs.uw(x′, p.uw, st.uw))
    vw_residual_interior .+= first(NNs.vw(x′, p.vw, st.vw))
    wT_residual_interior .+= first(NNs.wT(x′, p.wT, st.wT))
    wS_residual_interior .+= first(NNs.wS(x′, p.wS, st.wS))

    Dᶜ!(u_RHS_residual, uw_residual, Δ_hat)
    Dᶜ!(v_RHS_residual, vw_residual, Δ_hat)
    Dᶜ!(T_RHS_residual, wT_residual, Δ_hat)
    Dᶜ!(S_RHS_residual, wS_residual, Δ_hat)

    calculate_Ri!(Ri, u, v, ρ, ∂b∂z, ∂u∂z, ∂v∂z, g, ρ₀, Δ; clamp_lims=(-Inf, Inf))

    for i in eachindex(ν)
        ν[i], κ[i] = params.scaling.diffusivity(first(NNs.baseclosure(@view(Ri[i:i]), p.baseclosure, st.baseclosure)))
    end

    Dᶠ!(uw_diffusive, u_hat, Δ_hat)
    Dᶠ!(vw_diffusive, v_hat, Δ_hat)
    Dᶠ!(wT_diffusive, T_hat, Δ_hat)
    Dᶠ!(wS_diffusive, S_hat, Δ_hat)

    uw_diffusive .*= -ν
    vw_diffusive .*= -ν
    wT_diffusive .*= -κ
    wS_diffusive .*= -κ

    Dᶜ!(u_RHS_diffusive, uw_diffusive, Δ_hat)
    Dᶜ!(v_RHS_diffusive, vw_diffusive, Δ_hat)
    Dᶜ!(T_RHS_diffusive, wT_diffusive, Δ_hat)
    Dᶜ!(S_RHS_diffusive, wS_diffusive, Δ_hat)

    du = -τ / H^2 .* u_RHS_diffusive - τ / H * scaling.uw.σ / scaling.u.σ .* u_RHS_residual + f * τ / scaling.u.σ .* v
    dv = -τ / H^2 .* v_RHS_diffusive - τ / H * scaling.vw.σ / scaling.v.σ .* v_RHS_residual - f * τ / scaling.v.σ .* u
    dT = -τ / H^2 .* T_RHS_diffusive - τ / H * scaling.wT.σ / scaling.T.σ .* T_RHS_residual
    dS = -τ / H^2 .* S_RHS_diffusive - τ / H * scaling.wS.σ / scaling.S.σ .* S_RHS_residual

    return vcat(du, dv, dT, dS)
end

function NDE_opt!(dx, x, p, t, params, NNs, st)
    eos = TEOS10EquationOfState()
    coarse_size = params.coarse_size
    f = params.f
    Δ, Δ_hat = params.Δ, params.Δ_hat
    scaling = params.scaling
    τ, H = params.τ, params.H
    u, v, T, S, ρ, ρ_hat, x′ = params.cache.u, params.cache.v, params.cache.T, params.cache.S, params.cache.ρ, params.cache.ρ_hat, params.cache.x′
    Ri, ν, κ = params.cache.Ri, params.cache.ν, params.cache.κ
    ∂b∂z, ∂u∂z, ∂v∂z, g, ρ₀ = params.cache.∂b∂z, params.cache.∂u∂z, params.cache.∂v∂z, params.g, eos.reference_density
    uw_diffusive, vw_diffusive, wT_diffusive, wS_diffusive = params.cache.uw_diffusive, params.cache.vw_diffusive, params.cache.wT_diffusive, params.cache.wS_diffusive
    uw_residual, vw_residual, wT_residual, wS_residual = params.cache.uw_residual, params.cache.vw_residual, params.cache.wT_residual, params.cache.wS_residual
    u_RHS_diffusive, v_RHS_diffusive, T_RHS_diffusive, S_RHS_diffusive = params.cache.u_RHS_diffusive, params.cache.v_RHS_diffusive, params.cache.T_RHS_diffusive, params.cache.S_RHS_diffusive
    u_RHS_residual, v_RHS_residual, T_RHS_residual, S_RHS_residual = params.cache.u_RHS_residual, params.cache.v_RHS_residual, params.cache.T_RHS_residual, params.cache.S_RHS_residual

    u_hat = @view x[1:coarse_size]
    v_hat = @view x[coarse_size+1:2*coarse_size]
    T_hat = @view x[2*coarse_size+1:3*coarse_size]
    S_hat = @view x[3*coarse_size+1:4*coarse_size]

    u .= inv(params.scaling.u).(u_hat)
    v .= inv(params.scaling.v).(v_hat)
    T .= inv(params.scaling.T).(T_hat)
    S .= inv(params.scaling.S).(S_hat)

    ρ .= TEOS10.ρ.(T, S, 0, Ref(eos))
    ρ_hat .= params.scaling.ρ.(ρ)

    uw_residual_nottop = @view uw_residual[1:end-1]
    vw_residual_nottop = @view vw_residual[1:end-1]
    wT_residual_nottop = @view wT_residual[1:end-1]
    wS_residual_nottop = @view wS_residual[1:end-1]

    uw_residual_interior = @view uw_residual[2:end-1]
    vw_residual_interior = @view vw_residual[2:end-1]
    wT_residual_interior = @view wT_residual[2:end-1]
    wS_residual_interior = @view wS_residual[2:end-1]

    uw_residual[end] = params.uw.scaled.top
    vw_residual[end] = params.vw.scaled.top
    wT_residual[end] = params.wT.scaled.top
    wS_residual[end] = params.wS.scaled.top

    uw_residual_nottop .= params.uw.scaled.bottom
    vw_residual_nottop .= params.vw.scaled.bottom
    wT_residual_nottop .= params.wT.scaled.bottom
    wS_residual_nottop .= params.wS.scaled.bottom

    x′u = @view x′[1:coarse_size]
    x′v = @view x′[coarse_size+1:2*coarse_size]
    x′T = @view x′[2*coarse_size+1:3*coarse_size]
    x′S = @view x′[3*coarse_size+1:4*coarse_size]
    x′ρ = @view x′[4*coarse_size+1:5*coarse_size]

    x′u .= u_hat
    x′v .= v_hat
    x′T .= T_hat
    x′S .= S_hat
    x′ρ .= ρ_hat

    x′[5*coarse_size+1] = params.uw.scaled.top
    x′[5*coarse_size+2] = params.vw.scaled.top
    x′[5*coarse_size+3] = params.wT.scaled.top
    x′[5*coarse_size+4] = params.wS.scaled.top

    uw_residual_interior .+= first(NNs.uw(x′, p.uw, st.uw))
    vw_residual_interior .+= first(NNs.vw(x′, p.vw, st.vw))
    wT_residual_interior .+= first(NNs.wT(x′, p.wT, st.wT))
    wS_residual_interior .+= first(NNs.wS(x′, p.wS, st.wS))

    Dᶜ!(u_RHS_residual, uw_residual, Δ_hat)
    Dᶜ!(v_RHS_residual, vw_residual, Δ_hat)
    Dᶜ!(T_RHS_residual, wT_residual, Δ_hat)
    Dᶜ!(S_RHS_residual, wS_residual, Δ_hat)

    calculate_Ri!(Ri, u, v, ρ, ∂b∂z, ∂u∂z, ∂v∂z, g, ρ₀, Δ; clamp_lims=(-Inf, Inf))

    for i in eachindex(ν)
        ν[i], κ[i] = params.scaling.diffusivity(first(NNs.baseclosure(@view(Ri[i:i]), p.baseclosure, st.baseclosure)))
    end

    Dᶠ!(uw_diffusive, u_hat, Δ_hat)
    Dᶠ!(vw_diffusive, v_hat, Δ_hat)
    Dᶠ!(wT_diffusive, T_hat, Δ_hat)
    Dᶠ!(wS_diffusive, S_hat, Δ_hat)

    uw_diffusive .*= -ν
    vw_diffusive .*= -ν
    wT_diffusive .*= -κ
    wS_diffusive .*= -κ

    Dᶜ!(u_RHS_diffusive, uw_diffusive, Δ_hat)
    Dᶜ!(v_RHS_diffusive, vw_diffusive, Δ_hat)
    Dᶜ!(T_RHS_diffusive, wT_diffusive, Δ_hat)
    Dᶜ!(S_RHS_diffusive, wS_diffusive, Δ_hat)

    du = @view dx[1:coarse_size]
    dv = @view dx[coarse_size+1:2*coarse_size]
    dT = @view dx[2*coarse_size+1:3*coarse_size]
    dS = @view dx[3*coarse_size+1:4*coarse_size]

    @. du = -τ / H^2 * u_RHS_diffusive - τ / H * scaling.uw.σ / scaling.u.σ * u_RHS_residual + f * τ / scaling.u.σ * v
    @. dv = -τ / H^2 * v_RHS_diffusive - τ / H * scaling.vw.σ / scaling.v.σ * v_RHS_residual - f * τ / scaling.v.σ * u
    @. dT = -τ / H^2 * T_RHS_diffusive - τ / H * scaling.wT.σ / scaling.T.σ * T_RHS_residual
    @. dS = -τ / H^2 * S_RHS_diffusive - τ / H * scaling.wS.σ / scaling.S.σ * S_RHS_residual
end

# _x₀s = [ComponentArray(u=data.profile.u.scaled[:, 1], v=data.profile.v.scaled[:, 1], T=data.profile.T.scaled[:, 1], S=data.profile.S.scaled[:, 1]) for data in train_data.data]
_x₀s = [vcat(data.profile.u.scaled[:, 1], data.profile.v.scaled[:, 1], data.profile.T.scaled[:, 1], data.profile.S.scaled[:, 1]) for data in train_data.data]

# dx = deepcopy(_x₀s[1])
# x = deepcopy(_x₀s[1])
# NDE_opt!(dx, x, ps_NN, 0, _params[1], NNs, st_NN)

# _probs = [ODEProblem((x, p′, t) -> NDE_opt(x, p′, t, param, NNs, st_NN), x₀, (param.scaled_time[1], param.scaled_time[end]), ps_NN) for (x₀, param) in zip(_x₀s, _params)]
# _sols = [Array(solve(prob, ROCK2(), saveat=param.scaled_time, reltol=1e-3)) for (param, prob) in zip(_params, _probs)]

# prob1 = ODEProblem((x, p′, t) -> NDE_opt(x, p′, t, _params[1], NNs, st_NN), _x₀s[1], (_params[1].scaled_time[1], _params[1].scaled_time[end]), ps_NN)
# prob2 = ODEProblem((dx, x, p′, t) -> NDE_opt!(dx, x, p′, t, _params[1], NNs, st_NN), _x₀s[1], (_params[1].scaled_time[1], _params[1].scaled_time[end]), ps_NN)
# Array(solve(prob1, ROCK2(), saveat=_params[1].scaled_time, reltol=1e-3))
# Array(solve(prob2, ROCK2(), saveat=_params[1].scaled_time, reltol=1e-3))

# _x₀s_plot = [ComponentArray(u=data.profile.u.scaled[:, 1], v=data.profile.v.scaled[:, 1], T=data.profile.T.scaled[:, 1], S=data.profile.S.scaled[:, 1]) for data in train_data_plot.data]
# _probs_plots = [ODEProblem((x, p′, t) -> NDE_opt(x, p′, t, param, NNs, st_NN), x₀, (param.scaled_original_time[1], param.scaled_original_time[end]), ps_NN) for (x₀, param) in zip(_x₀s, _params)]
# _sols_plot = [solve(prob, ROCK2(), saveat=param.scaled_original_time, reltol=1e-3) for (param, prob) in zip(_params, _probs_plots)]

function animate_data(train_data, sols, index, FILE_DIR; coarse_size=32, epoch=1)
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

    n = Observable(1)
    zC = train_data.data[index].metadata["zC"]
    zF = train_data.data[index].metadata["zF"]

    u_NDE = inv(train_data.scaling.u).(sols[index][1:coarse_size, :])
    v_NDE = inv(train_data.scaling.v).(sols[index][coarse_size+1:2*coarse_size, :])
    T_NDE = inv(train_data.scaling.T).(sols[index][2*coarse_size+1:3*coarse_size, :])
    S_NDE = inv(train_data.scaling.S).(sols[index][3*coarse_size+1:4*coarse_size, :])
    ρ_NDE = TEOS10.ρ.(T_NDE, S_NDE, 0, Ref(TEOS10EquationOfState()))

    ulim = (find_min(u_NDE, train_data.data[index].profile.u.unscaled), find_max(u_NDE, train_data.data[index].profile.u.unscaled))
    vlim = (find_min(v_NDE, train_data.data[index].profile.v.unscaled), find_max(v_NDE, train_data.data[index].profile.v.unscaled))
    Tlim = (find_min(T_NDE, train_data.data[index].profile.T.unscaled), find_max(T_NDE, train_data.data[index].profile.T.unscaled))
    Slim = (find_min(S_NDE, train_data.data[index].profile.S.unscaled), find_max(S_NDE, train_data.data[index].profile.S.unscaled))
    ρlim = (find_min(ρ_NDE, train_data.data[index].profile.ρ.unscaled), find_max(ρ_NDE, train_data.data[index].profile.ρ.unscaled))

    uwlim = (find_min(train_data.data[index].flux.uw.column.unscaled), 
             find_max(train_data.data[index].flux.uw.column.unscaled))
    vwlim = (find_min(train_data.data[index].flux.vw.column.unscaled),
             find_max(train_data.data[index].flux.vw.column.unscaled))
    wTlim = (find_min(train_data.data[index].flux.wT.column.unscaled),
             find_max(train_data.data[index].flux.wT.column.unscaled))
    wSlim = (find_min(train_data.data[index].flux.wS.column.unscaled),
             find_max(train_data.data[index].flux.wS.column.unscaled))

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

    lines!(axvw, vw_truthₙ, zF, label="Truth")

    lines!(axwT, wT_truthₙ, zF, label="Truth")

    lines!(axwS, wS_truthₙ, zF, label="Truth")

    axislegend(axu, position=:rb)
    axislegend(axuw, position=:rb)
    
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
    # display(fig)

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

# for index in 1:1
#     animate_data(train_data_plot, _sols_plot, index, FILE_DIR; coarse_size=32, epoch=1)
# end

priors_uw = [constrained_gaussian("uw $i", 0, 1e-6, -Inf, Inf) for i in eachindex(ps_NN.uw)]
priors_vw = [constrained_gaussian("vw $i", 0, 1e-6, -Inf, Inf) for i in eachindex(ps_NN.vw)]
priors_wT = [constrained_gaussian("wT $i", 0, 1e-6, -Inf, Inf) for i in eachindex(ps_NN.wT)]
priors_wS = [constrained_gaussian("wS $i", 0, 1e-6, -Inf, Inf) for i in eachindex(ps_NN.wS)]
priors_baseline = [constrained_gaussian("baseline $i", p, 1e-6, -Inf, Inf) for (i, p) in enumerate(ps_NN.baseclosure)]

priors = combine_distributions(vcat(priors_uw, priors_vw, priors_wT, priors_wS, priors_baseline))

y = vec(vcat([vcat(data.profile.u.scaled, data.profile.v.scaled, data.profile.T.scaled, data.profile.S.scaled) for data in train_data.data]...))

N_ensemble = 720
N_iterations = 10
Γ = 1e-6 * I

initial_ensemble = EKP.construct_initial_ensemble(rng, priors, N_ensemble)
ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng)

params_ekp = [_params for _ in 1:Threads.nthreads()]

for i in 1:N_iterations
    ps_eki = get_ϕ_final(priors, ensemble_kalman_process)
    G_ens = zeros(length(y), N_ensemble)
    @info "$(Dates.now()), iteration $i/$N_iterations"
    Threads.@threads for j in 1:N_ensemble
        threadid = Threads.threadid()
        ps_particle = ComponentArray(ps_eki[:, j], ax_ps_NN)
        probs = [ODEProblem((x, p′, t) -> NDE_opt(x, p′, t, param, NNs, st_NN), x₀, (param.scaled_time[1], param.scaled_time[end]), ps_particle) for (x₀, param) in zip(_x₀s, params_ekp[threadid])]
        sols = [Array(solve(prob, ROCK2(), saveat=param.scaled_time, reltol=1e-3)) for (param, prob) in zip(params_ekp[threadid], probs)]
        G_ens[:, j] .= vec(vcat(sols...))
    end
    @info "$(Dates.now()), updating ensemble"
    EKP.update_ensemble!(ensemble_kalman_process, G_ens)
    final_ensemble = get_ϕ_final(priors, ensemble_kalman_process)
    jldsave("$(FILE_DIR)/training_results_$(i).jld2"; final_ensemble, ax_ps_NN, NNs, st_NN)
end

#%%
#=
function train_NDE(train_data, params, NNs, ps_NN, st_NN; coarse_size=32, dev=cpu_device(), solver=ROCK2(), Ri_clamp_lims=(-Inf, Inf))
    train_data = train_data |> dev
    x₀s = [vcat(data.profile.u.scaled[:, 1], data.profile.v.scaled[:, 1], data.profile.T.scaled[:, 1], data.profile.S.scaled[:, 1]) for data in train_data.data] |> dev
    eos = TEOS10EquationOfState()
    
    function predict_residual_flux(u_hat, v_hat, T_hat, S_hat, ρ_hat, p, params, st)
        x′ = vcat(u_hat, v_hat, T_hat, S_hat, ρ_hat, params.uw.scaled.top, params.vw.scaled.top, params.wT.scaled.top, params.wS.scaled.top)

        uw = vcat(0, first(NNs.uw(x′, p.uw, st.uw)), 0)
        vw = vcat(0, first(NNs.vw(x′, p.vw, st.vw)), 0)
        wT = vcat(0, first(NNs.wT(x′, p.wT, st.wT)), 0)
        wS = vcat(0, first(NNs.wS(x′, p.wS, st.wS)), 0)

        return uw, vw, wT, wS
    end

    function predict_residual_flux_dimensional(u_hat, v_hat, T_hat, S_hat, ρ_hat, p, params, st)
        uw_hat, vw_hat, wT_hat, wS_hat = predict_residual_flux(u_hat, v_hat, T_hat, S_hat, ρ_hat, p, params, st)
        uw = inv(params.scaling.uw).(uw_hat)
        vw = inv(params.scaling.vw).(vw_hat)
        wT = inv(params.scaling.wT).(wT_hat)
        wS = inv(params.scaling.wS).(wS_hat)

        uw .-= uw[1]
        vw .-= vw[1]
        wT .-= wT[1]
        wS .-= wS[1]

        return uw, vw, wT, wS
    end

    function predict_boundary_flux(params)
        uw = vcat(fill(params.uw.scaled.bottom, coarse_size), params.uw.scaled.top)
        vw = vcat(fill(params.vw.scaled.bottom, coarse_size), params.vw.scaled.top)
        wT = vcat(fill(params.wT.scaled.bottom, coarse_size), params.wT.scaled.top)
        wS = vcat(fill(params.wS.scaled.bottom, coarse_size), params.wS.scaled.top)

        return uw, vw, wT, wS
    end

    function predict_diffusivities(Ris, p, params, st)
        diffusivities = [params.scaling.diffusivity(first(NNs.baseclosure([Ri], p.baseclosure, st.baseclosure))) for Ri in Ris]

        νs = [diffusivity[1] for diffusivity in diffusivities]
        κs = [diffusivity[2] for diffusivity in diffusivities]

        return νs, κs
    end

    function predict_diffusive_flux(u, v, ρ, u_hat, v_hat, T_hat, S_hat, p, params, st)
        Ris = calculate_Ri(u, v, ρ, params.Dᶠ, params.g, eos.reference_density, clamp_lims=Ri_clamp_lims)
        νs, κs = predict_diffusivities(Ris, p, params, st)

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

        # du = @view dx[1:params.coarse_size]
        # dv = @view dx[params.coarse_size+1:2*params.coarse_size]
        # dT = @view dx[2*params.coarse_size+1:3*params.coarse_size]
        # dS = @view dx[3*params.coarse_size+1:4*params.coarse_size]

        du = -τ / H^2 .* (Dᶜ_hat * uw_diffusive) .- τ / H * scaling.uw.σ / scaling.u.σ .* (Dᶜ_hat * (uw_boundary .+ uw_residual)) .+ f * τ ./ scaling.u.σ .* v
        dv = -τ / H^2 .* (Dᶜ_hat * vw_diffusive) .- τ / H * scaling.vw.σ / scaling.v.σ .* (Dᶜ_hat * (vw_boundary .+ vw_residual)) .- f * τ ./ scaling.v.σ .* u
        dT = -τ / H^2 .* (Dᶜ_hat * wT_diffusive) .- τ / H * scaling.wT.σ / scaling.T.σ .* (Dᶜ_hat * (wT_boundary .+ wT_residual))
        dS = -τ / H^2 .* (Dᶜ_hat * wS_diffusive) .- τ / H * scaling.wS.σ / scaling.S.σ .* (Dᶜ_hat * (wS_boundary .+ wS_residual))

        return vcat(du, dv, dT, dS)
    end

    function predict_NDE(p)
        probs = [ODEProblem((x, p′, t) -> NDE(x, p′, t, param, st_NN), x₀, (param.scaled_time[1], param.scaled_time[end]), p) for (x₀, param) in zip(x₀s, params)]
        sols = [Array(solve(prob, solver, saveat=param.scaled_time, reltol=1e-3)) for (param, prob) in zip(params, probs)]
        return sols
    end

    predict_NDE(ps_NN)
end

sols_nonmutating = train_NDE(train_data, params, NNs, ps_NN, st_NN; coarse_size=32, dev=cpu_device(), solver=ROCK2(), Ri_clamp_lims=(-Inf, Inf))

lines(sols_nonmutating[1][128,:])

a = 3
#=
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

    # display(fig)

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

res, loss, sols, fluxes, losses, diffusivities = train_NDE(train_data, train_data_plot, NNs, ps_NN, st_NN, maxiter=50, solver=ROCK4(), sensealg=GaussAdjoint(autojacvec=ZygoteVJP()))

jldsave("$(FILE_DIR)/training_results_$(epoch).jld2"; res, loss, sols, fluxes, losses, NNs, st_NN, diffusivities)
plot_loss(losses, FILE_DIR, epoch=epoch)
for i in eachindex(field_datasets)
    animate_data(train_data_plot, sols, fluxes, diffusivities, i, FILE_DIR, epoch=epoch)
end

epoch += 1

res, loss, sols, fluxes, losses, diffusivities = train_NDE(train_data, train_data_plot, NNs, res.u, st_NN, maxiter=50, solver=ROCK4(), sensealg=GaussAdjoint(autojacvec=ZygoteVJP()))

jldsave("$(FILE_DIR)/training_results_$(epoch).jld2"; res, loss, sols, fluxes, losses, NNs, st_NN, diffusivities)
plot_loss(losses, FILE_DIR, epoch=epoch)
for i in eachindex(field_datasets)
    animate_data(train_data_plot, sols, fluxes, diffusivities, i, FILE_DIR, epoch=epoch)
end

epoch += 1

res, loss, sols, fluxes, losses, diffusivities = train_NDE(train_data, train_data_plot, NNs, res.u, st_NN, maxiter=50, solver=ROCK4(), sensealg=GaussAdjoint(autojacvec=ZygoteVJP()))

jldsave("$(FILE_DIR)/training_results_$(epoch).jld2"; res, loss, sols, fluxes, losses, NNs, st_NN, diffusivities)
plot_loss(losses, FILE_DIR, epoch=epoch)
for i in eachindex(field_datasets)
    animate_data(train_data_plot, sols, fluxes, diffusivities, i, FILE_DIR, epoch=epoch)
end

epoch += 1

res, loss, sols, fluxes, losses, diffusivities = train_NDE(train_data, train_data_plot, NNs, res.u, st_NN, maxiter=50, solver=ROCK4())

jldsave("$(FILE_DIR)/training_results_$(epoch).jld2"; res, loss, sols, fluxes, losses, NNs, st_NN, diffusivities)
plot_loss(losses, FILE_DIR, epoch=epoch)
for i in eachindex(field_datasets)
    animate_data(train_data_plot, sols, fluxes, diffusivities, i, FILE_DIR, epoch=epoch)
end


=#