using Distributed
addprocs(80)
@everywhere begin
    using SaltyOceanParameterizations, SaltyOceanParameterizations.DataWrangling
    using Oceananigans, SeawaterPolynomials.TEOS10
    using ComponentArrays, Lux, OrdinaryDiffEq, Optimization, Random, LuxCUDA
    using Statistics
    using Printf
    using Dates
    using JLD2
    using SciMLBase
    using LinearAlgebra
    using SharedArrays
    import SeawaterPolynomials.TEOS10: s, ΔS, Sₐᵤ

    s(Sᴬ) = Sᴬ + ΔS >= 0 ? √((Sᴬ + ΔS) / Sₐᵤ) : NaN
end
using CairoMakie
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses

function find_min(a...)
    return minimum(minimum.([a...]))
end
  
function find_max(a...)
    return maximum(maximum.([a...]))
end

FILE_DIR = "./training_output/NN_small_local_diffusivity_NDE_gradient_relu_noclamp_ROCK4_EKI_fast_test_diffeqensemble"
mkpath(FILE_DIR)

LES_FILE_DIRS = [
    "./LES_training/linearTS_dTdz_0.0013_dSdz_-0.0014_QU_-0.0002_QT_3.0e-5_QS_-3.0e-5_T_4.3_S_33.5_f_-0.00012_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.0002_QT_0.0003_QS_-3.0e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.0002_QT_0.0003_QS_-3.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_-0.025_dSdz_-0.0045_QU_-0.0002_QT_-0.0003_QS_-3.0e-5_T_-3.6_S_33.9_f_-0.000125_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
]

BASECLOSURE_FILE_DIR = "./training_output/local_diffusivity_NDE_gradient_relu_noclamp/training_results_2.jld2"
PS_BASECLOSURE_FILE_DIR = "./training_output/local_diffusivity_NDE_gradient_relu_noclamp/ps_baseclosure.jld2"

field_datasets = [FieldDataset(FILE_DIR, backend=OnDisk()) for FILE_DIR in LES_FILE_DIRS]

file_baseclosure = jldopen(BASECLOSURE_FILE_DIR, "r")
baseclosure_NN = file_baseclosure["NN"]
st_baseclosure = file_baseclosure["st_NN"]
close(file_baseclosure)

file_ps_baseclosure = jldopen(PS_BASECLOSURE_FILE_DIR, "r")
ps_baseclosure = file_ps_baseclosure["ps_baseclosure"]
close(file_ps_baseclosure)

full_timeframes = [1:length(data["ubar"].times) for data in field_datasets]
timeframes = [5:5:length(data["ubar"].times) for data in field_datasets]
train_data = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, timeframes)
coarse_size = 32

rng = Random.MersenneTwister(123)

uw_NN = Chain(Dense(165, 32, leakyrelu), Dense(32, 31))
vw_NN = Chain(Dense(165, 32, leakyrelu), Dense(32, 31))
wT_NN = Chain(Dense(165, 32, leakyrelu), Dense(32, 31))
wS_NN = Chain(Dense(165, 32, leakyrelu), Dense(32, 31))

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
@everywhere NNs = $NNs
ps_NN = ComponentArray(uw=ps_uw, vw=ps_vw, wT=ps_wT, wS=ps_wS, baseclosure=ps_baseclosure)
N_parameters = length(ps_NN)

ax_ps_NN = getaxes(ps_NN)
@everywhere ax_ps_NN = $ax_ps_NN
st_NN = (uw=st_uw, vw=st_vw, wT=st_wT, wS=st_wS, baseclosure=st_baseclosure)
@everywhere st_NN = $st_NN

data_params = [(                   f = data.metadata["coriolis_parameter"],
                        f_scaled = data.coriolis.scaled,
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
                                                x′=zeros(5*coarse_size+5),
                                                Ri=zeros(coarse_size+1), ν=zeros(coarse_size+1), κ=zeros(coarse_size+1),
                                                ∂b∂z=zeros(coarse_size+1), ∂u∂z=zeros(coarse_size+1), ∂v∂z=zeros(coarse_size+1),
                                                uw_diffusive=zeros(coarse_size+1), vw_diffusive=zeros(coarse_size+1), wT_diffusive=zeros(coarse_size+1), wS_diffusive=zeros(coarse_size+1),
                                                uw_residual=zeros(coarse_size+1), vw_residual=zeros(coarse_size+1), wT_residual=zeros(coarse_size+1), wS_residual=zeros(coarse_size+1),
                                                u_RHS_diffusive=zeros(coarse_size), v_RHS_diffusive=zeros(coarse_size), T_RHS_diffusive=zeros(coarse_size), S_RHS_diffusive=zeros(coarse_size),
                                                u_RHS_residual=zeros(coarse_size), v_RHS_residual=zeros(coarse_size), T_RHS_residual=zeros(coarse_size), S_RHS_residual=zeros(coarse_size),)
            ) for data in train_data.data]

@everywhere data_params = $data_params

x₀s = [vcat(data.profile.u.scaled[:, 1], data.profile.v.scaled[:, 1], data.profile.T.scaled[:, 1], data.profile.S.scaled[:, 1]) for data in train_data.data]
@everywhere x₀s = $x₀s
@everywhere n_simulations = length(x₀s)

@everywhere function NDE(x, p, t, params, NNs, st)
    eos = TEOS10EquationOfState()
    coarse_size = params.coarse_size
    f, f_scaled = params.f, params.f_scaled
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
    x′[5*coarse_size+5] = f_scaled

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

priors_uw = [constrained_gaussian("uw $i", 0, 1e-3, -Inf, Inf) for i in eachindex(ps_NN.uw)]
priors_vw = [constrained_gaussian("vw $i", 0, 1e-3, -Inf, Inf) for i in eachindex(ps_NN.vw)]
priors_wT = [constrained_gaussian("wT $i", 0, 1e-3, -Inf, Inf) for i in eachindex(ps_NN.wT)]
priors_wS = [constrained_gaussian("wS $i", 0, 1e-3, -Inf, Inf) for i in eachindex(ps_NN.wS)]
priors_baseline = [constrained_gaussian("baseline $i", p, 1e-3, -Inf, Inf) for (i, p) in enumerate(ps_NN.baseclosure)]

priors = combine_distributions(vcat(priors_uw, priors_vw, priors_wT, priors_wS, priors_baseline))

function compute_losses(sim, i, train_data; coarse_size=32, n_simulations=4, losses_prefactor=(u=1, v=1, T=1, S=1, ρ=1, ∂u∂z=1, ∂v∂z=1, ∂T∂z=1, ∂S∂z=1, ∂ρ∂z=1))
    if sim[i].retcode == ReturnCode.Success
        sim_index = mod1(i, n_simulations)
        sol = sim[i].u
        Δ = train_data.data[sim_index].metadata["zC"][2] - train_data.data[sim_index].metadata["zC"][1]

        u = hcat([sol[j][1:coarse_size] for j in eachindex(sol)]...)
        v = hcat([sol[j][coarse_size+1:2*coarse_size] for j in eachindex(sol)]...)
        T = hcat([sol[j][2*coarse_size+1:3*coarse_size] for j in eachindex(sol)]...)
        S = hcat([sol[j][3*coarse_size+1:4*coarse_size] for j in eachindex(sol)]...)
        ρ = train_data.scaling.ρ.(TEOS10.ρ.(inv(train_data.scaling.T).(T), inv(train_data.scaling.S).(S), 0, Ref(TEOS10EquationOfState())))

        ∂u∂z_hats = zeros(size(u, 1)+1, size(u, 2))
        ∂v∂z_hats = zeros(size(v, 1)+1, size(v, 2))
        ∂T∂z_hats = zeros(size(T, 1)+1, size(T, 2))
        ∂S∂z_hats = zeros(size(S, 1)+1, size(S, 2))
        ∂ρ∂z_hats = zeros(size(ρ, 1)+1, size(ρ, 2))

        for j in axes(∂u∂z_hats, 2)
            ∂u∂z_col =  @view ∂u∂z_hats[:, j]
            ∂v∂z_col =  @view ∂v∂z_hats[:, j]
            ∂T∂z_col =  @view ∂T∂z_hats[:, j]
            ∂S∂z_col =  @view ∂S∂z_hats[:, j]
            ∂ρ∂z_col =  @view ∂ρ∂z_hats[:, j]

            u_col = @view u[:, j]
            v_col = @view v[:, j]
            T_col = @view T[:, j]
            S_col = @view S[:, j]
            ρ_col = @view ρ[:, j]

            Dᶠ!(∂u∂z_col, inv(train_data.scaling.u).(u_col), Δ)
            Dᶠ!(∂v∂z_col, inv(train_data.scaling.v).(v_col), Δ)
            Dᶠ!(∂T∂z_col, inv(train_data.scaling.T).(T_col), Δ)
            Dᶠ!(∂S∂z_col, inv(train_data.scaling.S).(S_col), Δ)
            Dᶠ!(∂ρ∂z_col, inv(train_data.scaling.ρ).(ρ_col), Δ)

            ∂u∂z_col .= train_data.scaling.∂u∂z.(∂u∂z_col)
            ∂v∂z_col .= train_data.scaling.∂v∂z.(∂v∂z_col)
            ∂T∂z_col .= train_data.scaling.∂T∂z.(∂T∂z_col)
            ∂S∂z_col .= train_data.scaling.∂S∂z.(∂S∂z_col)
            ∂ρ∂z_col .= train_data.scaling.∂ρ∂z.(∂ρ∂z_col)
        end

        u_truth = train_data.data[sim_index].profile.u.scaled
        v_truth = train_data.data[sim_index].profile.v.scaled
        T_truth = train_data.data[sim_index].profile.T.scaled
        S_truth = train_data.data[sim_index].profile.S.scaled
        ρ_truth = train_data.data[sim_index].profile.ρ.scaled

        ∂u∂z_truth = train_data.data[sim_index].profile.∂u∂z.scaled
        ∂v∂z_truth = train_data.data[sim_index].profile.∂v∂z.scaled
        ∂T∂z_truth = train_data.data[sim_index].profile.∂T∂z.scaled
        ∂S∂z_truth = train_data.data[sim_index].profile.∂S∂z.scaled
        ∂ρ∂z_truth = train_data.data[sim_index].profile.∂ρ∂z.scaled

        u_loss = losses_prefactor.u * sum((u .- u_truth) .^ 2)
        v_loss = losses_prefactor.v * sum((v .- v_truth) .^ 2)
        T_loss = losses_prefactor.T * sum((T .- T_truth) .^ 2)
        S_loss = losses_prefactor.S * sum((S .- S_truth) .^ 2)
        ρ_loss = losses_prefactor.ρ * sum((ρ .- ρ_truth) .^ 2)

        ∂u∂z_loss = losses_prefactor.∂u∂z * sum((∂u∂z_hats .- ∂u∂z_truth) .^ 2)
        ∂v∂z_loss = losses_prefactor.∂v∂z * sum((∂v∂z_hats .- ∂v∂z_truth) .^ 2)
        ∂T∂z_loss = losses_prefactor.∂T∂z * sum((∂T∂z_hats .- ∂T∂z_truth) .^ 2)
        ∂S∂z_loss = losses_prefactor.∂S∂z * sum((∂S∂z_hats .- ∂S∂z_truth) .^ 2)
        ∂ρ∂z_loss = losses_prefactor.∂ρ∂z * sum((∂ρ∂z_hats .- ∂ρ∂z_truth) .^ 2)
    else
        u_loss = NaN
        v_loss = NaN
        T_loss = NaN
        S_loss = NaN
        ρ_loss = NaN

        ∂u∂z_loss = NaN
        ∂v∂z_loss = NaN
        ∂T∂z_loss = NaN
        ∂S∂z_loss = NaN
        ∂ρ∂z_loss = NaN
    end
    return (u=u_loss, v=v_loss, T=T_loss, S=S_loss, ρ=ρ_loss, ∂u∂z=∂u∂z_loss, ∂v∂z=∂v∂z_loss, ∂T∂z=∂T∂z_loss, ∂S∂z=∂S∂z_loss, ∂ρ∂z=∂ρ∂z_loss)
end

function compute_loss(sim, i, train_data; coarse_size=32, n_simulations=4, losses_prefactor=(u=1, v=1, T=1, S=1, ρ=1, ∂u∂z=1, ∂v∂z=1, ∂T∂z=1, ∂S∂z=1, ∂ρ∂z=1))
    return sum(values(compute_losses(sim, i, train_data; coarse_size=coarse_size, n_simulations=n_simulations, losses_prefactor=losses_prefactor)))
end

function compute_loss_prefactor(u_loss, v_loss, T_loss, S_loss, ρ_loss, ∂u∂z_loss, ∂v∂z_loss, ∂T∂z_loss, ∂S∂z_loss, ∂ρ∂z_loss)
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

    return (u=u_prefactor, v=v_prefactor, T=T_prefactor, S=S_prefactor, ρ=ρ_prefactor, ∂u∂z=∂u∂z_prefactor, ∂v∂z=∂v∂z_prefactor, ∂T∂z=∂T∂z_prefactor, ∂S∂z=∂S∂z_prefactor, ∂ρ∂z=∂ρ∂z_prefactor)
end

function compute_loss_prefactor(u_loss, v_loss, T_loss, S_loss, ρ_loss)
    ρ_prefactor = 1
    T_prefactor = ρ_loss / T_loss
    S_prefactor = ρ_loss / S_loss
    u_prefactor = ρ_loss / u_loss * (0.05/0.3)
    v_prefactor = ρ_loss / v_loss * (0.05/0.3)

    return (u=u_prefactor, v=v_prefactor, T=T_prefactor, S=S_prefactor, ρ=ρ_prefactor)
end

target = [0.]

N_ensemble = 4000
N_iterations = 100
Γ = 1e-6 * I

ps_eki = SharedArray{Float64}(N_parameters, N_ensemble)

for proc in procs(ps_eki)
    @fetchfrom proc identity(ps_eki)
end

@info "Constructing initial ensemble"
ps_eki .= EKP.construct_initial_ensemble(rng, priors, N_ensemble)

ensemble_kalman_process = EKP.EnsembleKalmanProcess(ps_eki, target, Γ, Inversion(); 
                                                    rng = rng, 
                                                    failure_handler_method = SampleSuccGauss(), 
                                                    scheduler = DataMisfitController(on_terminate="continue"))
# )

total_ensemble = N_ensemble * n_simulations

prob_base = ODEProblem((x, p′, t) -> NDE(x, p′, t, data_params[1], NNs, st_NN), x₀s[1], (data_params[1].scaled_time[1], data_params[1].scaled_time[end]), ps_NN)

@everywhere function prob_func(prob, i, repeat)
    sim_index = mod1(i, n_simulations)
    particle_index = Int(ceil(i / n_simulations))
    # @info "$(Dates.now()), i = $i, sim_index = $sim_index, particle_index = $particle_index"

    ps_particle = ComponentArray(ps_eki[:, particle_index], ax_ps_NN)
    x₀ = x₀s[sim_index]
    params = data_params[sim_index]
    remake(prob, f=(x, p′, t) -> NDE(x, p′, t, params, NNs, st_NN), u0=x₀, p=ps_particle)
end

@info "First solve to obtain losses prefactor"
ensemble_prob = EnsembleProblem(prob_base, prob_func=prob_func, safetycopy=false)
sim = solve(ensemble_prob, VCABM3(), EnsembleDistributed(), saveat=data_params[1].scaled_time, reltol=1e-3, trajectories=total_ensemble, maxiters=1e5)
sim_losses = [compute_losses(sim, i, train_data, coarse_size=32, n_simulations=length(train_data.data)) for i in eachindex(sim)]
losses_prefactor = compute_loss_prefactor(mean([losses.u for losses in sim_losses]), 
                                          mean([losses.v for losses in sim_losses]), 
                                          mean([losses.T for losses in sim_losses]), 
                                          mean([losses.S for losses in sim_losses]), 
                                          mean([losses.ρ for losses in sim_losses]),
                                          mean([losses.∂u∂z for losses in sim_losses]),
                                          mean([losses.∂v∂z for losses in sim_losses]),
                                          mean([losses.∂T∂z for losses in sim_losses]),
                                          mean([losses.∂S∂z for losses in sim_losses]),
                                          mean([losses.∂ρ∂z for losses in sim_losses]))

for i in 1:N_iterations
    @info "$(Dates.now()), iteration $i/$N_iterations"
    @info "$(Dates.now()), obtaining weights"

    ps_eki .= get_ϕ_final(priors, ensemble_kalman_process)

    @info "$(Dates.now()), Begin solving ensemble problem"
    sim = solve(ensemble_prob, VCABM3(), EnsembleDistributed(), saveat=data_params[1].scaled_time, reltol=1e-3, trajectories=total_ensemble, maxiters=1e5)

    @info "$(Dates.now()), computing loss"
    sim_loss = [compute_loss(sim, i, train_data, coarse_size=32, n_simulations=length(train_data.data), losses_prefactor=losses_prefactor) for i in eachindex(sim)]
    
    loss = hcat([mean(sim_loss[i*n_simulations+1:i*n_simulations+n_simulations]) for i in 0:N_ensemble-1]...)
    @info "$(Dates.now()), mean loss: $(mean(sim_loss))"
    # @info "loss = $loss"
    @info "$(Dates.now()), updating ensemble"
    EKP.update_ensemble!(ensemble_kalman_process, loss)

    @info "$(Dates.now()), obtaining posterior"
    final_ensemble = get_ϕ_final(priors, ensemble_kalman_process)
    jldsave("$(FILE_DIR)/training_results_$(i).jld2"; final_ensemble, ax_ps_NN, NNs, st_NN)
end