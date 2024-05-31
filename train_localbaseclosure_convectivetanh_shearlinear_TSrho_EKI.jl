using LinearAlgebra
using Lux, ComponentArrays, Random
using Printf
using SaltyOceanParameterizations
using SaltyOceanParameterizations.DataWrangling
using SaltyOceanParameterizations: calculate_Ri, local_Ri_ν_convectivetanh_shearlinear, local_Ri_κ_convectivetanh_shearlinear
using Oceananigans
using JLD2
using SeawaterPolynomials.TEOS10
using CairoMakie
using Printf
import Dates
using Statistics
using Colors
using ArgParse
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses
using SeawaterPolynomials

function parse_commandline()
    s = ArgParseSettings()
  
    @add_arg_table! s begin
      "--S_scaling"
        help = "Scaling factor for S"
        arg_type = Float64
        default = 1.0
      "--momentum_ratio"
        help = "Momentum ratio"
        arg_type = Float64
        default = 1.0
    end
    return parse_args(s)
end

args = parse_commandline()

LES_FILE_DIRS = ["./LES2/$(file)/instantaneous_timeseries.jld2" for file in LES_suite["train21PWFC"]]
const S_scaling = args["S_scaling"]
const momentum_ratio = args["momentum_ratio"]
FILE_DIR = "./training_output/$(length(LES_FILE_DIRS))simPWFC_mom_$(momentum_ratio)_localbaseclosure_convectivetanh_shearlinear_EKI"
mkpath(FILE_DIR)

field_datasets = [FieldDataset(FILE_DIR, backend=OnDisk()) for FILE_DIR in LES_FILE_DIRS]

timeframes = [25:10:length(data["ubar"].times) for data in field_datasets]
full_timeframes = [25:length(data["ubar"].times) for data in field_datasets]
coarse_size = 32
train_data = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, timeframes, coarse_size)
scaling = train_data.scaling

truths = [(; u=data.profile.u.scaled, v=data.profile.v.scaled, T=data.profile.T.scaled, S=data.profile.S.scaled, ρ=data.profile.ρ.scaled, 
             ∂u∂z=data.profile.∂u∂z.scaled, ∂v∂z=data.profile.∂v∂z.scaled, ∂T∂z=data.profile.∂T∂z.scaled, ∂S∂z=data.profile.∂S∂z.scaled, ∂ρ∂z=data.profile.∂ρ∂z.scaled) for data in train_data.data]

x₀s = [(; u=data.profile.u.scaled[:, 1], v=data.profile.v.scaled[:, 1], T=data.profile.T.scaled[:, 1], S=data.profile.S.scaled[:, 1]) for data in train_data.data]

train_data_plot = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, full_timeframes, coarse_size)

params = ODEParams(train_data)
params_plot = ODEParams(train_data_plot, scaling)

caches = [(boundary=(uw=zeros(coarse_size+1), vw=zeros(coarse_size+1), wT=zeros(coarse_size+1), wS=zeros(coarse_size+1)),
           diffusivities=(ν=zeros(coarse_size+1), κ=zeros(coarse_size+1)),
           Ri=zeros(coarse_size+1))
            for _ in params]

rng = Random.default_rng(123)

ps = ComponentArray(ν_conv=1.295, ν_shear=7.932e-02, m=-1.757e-01, Pr=1.193, ΔRi=0.0108)

function predict_boundary_flux(params)
    uw = vcat(fill(params.uw.scaled.bottom, params.coarse_size), params.uw.scaled.top)
    vw = vcat(fill(params.vw.scaled.bottom, params.coarse_size), params.vw.scaled.top)
    wT = vcat(fill(params.wT.scaled.bottom, params.coarse_size), params.wT.scaled.top)
    wS = vcat(fill(params.wS.scaled.bottom, params.coarse_size), params.wS.scaled.top)

    return uw, vw, wT, wS
end

function predict_boundary_flux!(uw, vw, wT, wS, params)
    uw[1:end-1] .= params.uw.scaled.bottom
    vw[1:end-1] .= params.vw.scaled.bottom
    wT[1:end-1] .= params.wT.scaled.bottom
    wS[1:end-1] .= params.wS.scaled.bottom

    uw[end] = params.uw.scaled.top
    vw[end] = params.vw.scaled.top
    wT[end] = params.wT.scaled.top
    wS[end] = params.wS.scaled.top

    return nothing
end

predict_boundary_flux!(caches[1].boundary..., params[1])

function predict_diffusivities(Ris, ps_baseclosure)
    νs = local_Ri_ν_convectivetanh_shearlinear.(Ris, ps_baseclosure.ν_conv, ps_baseclosure.ν_shear, ps_baseclosure.m, ps_baseclosure.ΔRi)
    κs = local_Ri_κ_convectivetanh_shearlinear.(νs, ps_baseclosure.Pr)
    return νs, κs
end

function predict_diffusivities!(νs, κs, Ris, ps_baseclosure)
    νs .= local_Ri_ν_convectivetanh_shearlinear.(Ris, ps_baseclosure.ν_conv, ps_baseclosure.ν_shear, ps_baseclosure.m, ps_baseclosure.ΔRi)
    κs .= local_Ri_κ_convectivetanh_shearlinear.(νs, ps_baseclosure.Pr)
    return nothing
end

predict_diffusivities!(caches[1].diffusivities..., rand(coarse_size+1), ps)

function solve_NDE(ps, params, x₀, timestep_multiple=10)
    eos = TEOS10EquationOfState()
    coarse_size = params.coarse_size
    Δt = (params.scaled_time[2] - params.scaled_time[1]) / timestep_multiple
    Nt_solve = (params.N_timesteps - 1) * timestep_multiple + 1
    Dᶜ_hat = params.Dᶜ_hat
    Dᶠ_hat = params.Dᶠ_hat
    Dᶠ = params.Dᶠ

    scaling = params.scaling
    τ, H = params.τ, params.H
    f = params.f

    u_hat = deepcopy(x₀.u)
    v_hat = deepcopy(x₀.v)
    T_hat = deepcopy(x₀.T)
    S_hat = deepcopy(x₀.S)
    ρ_hat = zeros(coarse_size)

    u = zeros(coarse_size)
    v = zeros(coarse_size)
    T = zeros(coarse_size)
    S = zeros(coarse_size)
    ρ = zeros(coarse_size)
    
    u_RHS = zeros(coarse_size)
    v_RHS = zeros(coarse_size)
    T_RHS = zeros(coarse_size)
    S_RHS = zeros(coarse_size)

    sol_u = zeros(coarse_size, Nt_solve)
    sol_v = zeros(coarse_size, Nt_solve)
    sol_T = zeros(coarse_size, Nt_solve)
    sol_S = zeros(coarse_size, Nt_solve)
    sol_ρ = zeros(coarse_size, Nt_solve)

    uw_boundary = zeros(coarse_size+1)
    vw_boundary = zeros(coarse_size+1)
    wT_boundary = zeros(coarse_size+1)
    wS_boundary = zeros(coarse_size+1)

    νs = zeros(coarse_size+1)
    κs = zeros(coarse_size+1)

    Ris = zeros(coarse_size+1)

    sol_u[:, 1] .= u_hat
    sol_v[:, 1] .= v_hat
    sol_T[:, 1] .= T_hat
    sol_S[:, 1] .= S_hat

    ν_LHS = Tridiagonal(zeros(coarse_size, coarse_size))
    κ_LHS = Tridiagonal(zeros(coarse_size, coarse_size))

    for i in 2:Nt_solve
        u .= inv(scaling.u).(u_hat)
        v .= inv(scaling.v).(v_hat)
        T .= inv(scaling.T).(T_hat)
        S .= inv(scaling.S).(S_hat)

        ρ .= TEOS10.ρ.(T, S, 0, Ref(eos))
        ρ_hat .= scaling.ρ.(ρ)
        sol_ρ[:, i-1] .= ρ_hat

        Ris .= calculate_Ri(u, v, ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf))
        predict_diffusivities!(νs, κs, Ris, ps)

        Dν = Dᶜ_hat * (-νs .* Dᶠ_hat)
        Dκ = Dᶜ_hat * (-κs .* Dᶠ_hat)

        predict_boundary_flux!(uw_boundary, vw_boundary, wT_boundary, wS_boundary, params)

        ν_LHS .= Tridiagonal(-τ / H^2 .* Dν)
        κ_LHS .= Tridiagonal(-τ / H^2 .* Dκ)

        u_RHS .= - τ / H * scaling.uw.σ / scaling.u.σ .* (Dᶜ_hat * (uw_boundary)) .+ f * τ ./ scaling.u.σ .* v
        v_RHS .= - τ / H * scaling.vw.σ / scaling.v.σ .* (Dᶜ_hat * (vw_boundary)) .- f * τ ./ scaling.v.σ .* u
        T_RHS .= - τ / H * scaling.wT.σ / scaling.T.σ .* (Dᶜ_hat * (wT_boundary))
        S_RHS .= - τ / H * scaling.wS.σ / scaling.S.σ .* (Dᶜ_hat * (wS_boundary))

        u_hat .= (I - Δt .* ν_LHS) \ (u_hat .+ Δt .* u_RHS)
        v_hat .= (I - Δt .* ν_LHS) \ (v_hat .+ Δt .* v_RHS)
        T_hat .= (I - Δt .* κ_LHS) \ (T_hat .+ Δt .* T_RHS)
        S_hat .= (I - Δt .* κ_LHS) \ (S_hat .+ Δt .* S_RHS)

        sol_u[:, i] .= u_hat
        sol_v[:, i] .= v_hat
        sol_T[:, i] .= T_hat
        sol_S[:, i] .= S_hat
    end

    sol_ρ[:, end] .= scaling.ρ.(TEOS10.ρ.(inv(scaling.T).(T_hat), inv(scaling.S).(S_hat), 0, Ref(eos)))

    return (; u=sol_u[:, 1:timestep_multiple:end], v=sol_v[:, 1:timestep_multiple:end], T=sol_T[:, 1:timestep_multiple:end], S=sol_S[:, 1:timestep_multiple:end], ρ=sol_ρ[:, 1:timestep_multiple:end])
end

# sol_u, sol_v, sol_T, sol_S, sol_ρ = solve_NDE(ps, params[6], x₀s[6], 10)
# sol_u, sol_v, sol_T, sol_S, sol_ρ = solve_NDE(ps, params[1], x₀s[1], 10)
# sol_u, sol_v, sol_T, sol_S, sol_ρ = solve_NDE(ps, params[4], x₀s[4])

# #%%
# truth = truths[4]
# fig = Figure(size=(900, 600))
# axu = CairoMakie.Axis(fig[1, 1], xlabel="T", ylabel="z")
# axv = CairoMakie.Axis(fig[1, 2], xlabel="T", ylabel="z")
# axT = CairoMakie.Axis(fig[2, 1], xlabel="T", ylabel="z")
# axS = CairoMakie.Axis(fig[2, 2], xlabel="S", ylabel="z")
# axρ = CairoMakie.Axis(fig[2, 3], xlabel="ρ", ylabel="z")

# lines!(axu, sol_u[:, 1], params[1].zC, label="initial")
# lines!(axu, sol_u[:, end], params[1].zC, label="final")
# lines!(axu, truth.u[:, end], train_data.data[1].metadata["zC"], label="truth")

# lines!(axv, sol_v[:, 1], params[1].zC, label="initial")
# lines!(axv, sol_v[:, end], params[1].zC, label="final")
# lines!(axv, truth.v[:, end], train_data.data[1].metadata["zC"], label="truth")

# lines!(axT, sol_T[:, 1], params[1].zC, label="initial")
# lines!(axT, sol_T[:, end], params[1].zC, label="final")
# lines!(axT, truth.T[:, end], train_data.data[1].metadata["zC"], label="truth")

# lines!(axS, sol_S[:, 1], params[1].zC, label="initial")
# lines!(axS, sol_S[:, end], params[1].zC, label="final")
# lines!(axS, truth.S[:, end], train_data.data[1].metadata["zC"], label="truth")

# lines!(axρ, sol_ρ[:, 1], params[1].zC, label="initial")
# lines!(axρ, sol_ρ[:, end], params[1].zC, label="final")
# lines!(axρ, truth.ρ[:, end], train_data.data[1].metadata["zC"], label="truth")

# Legend(fig[1, 3], axT, orientation=:vertical, tellwidth=false)
# display(fig)
#%%
function individual_loss(ps, truth, params, x₀)
    Dᶠ = params.Dᶠ
    scaling = params.scaling
    sol_u, sol_v, sol_T, sol_S, sol_ρ = solve_NDE(ps, params, x₀)

    u_loss = mean((sol_u .- truth.u).^2)
    v_loss = mean((sol_v .- truth.v).^2)
    T_loss = mean((sol_T .- truth.T).^2)
    S_loss = mean((sol_S .- truth.S).^2)
    ρ_loss = mean((sol_ρ .- truth.ρ).^2)

    u = inv(scaling.u).(sol_u)
    v = inv(scaling.v).(sol_v)
    T = inv(scaling.T).(sol_T)
    S = inv(scaling.S).(sol_S)
    ρ = inv(scaling.ρ).(sol_ρ)

    ∂u∂z = scaling.∂u∂z.(Dᶠ * u)
    ∂v∂z = scaling.∂v∂z.(Dᶠ * v)
    ∂T∂z = scaling.∂T∂z.(Dᶠ * T)
    ∂S∂z = scaling.∂S∂z.(Dᶠ * S)
    ∂ρ∂z = scaling.∂ρ∂z.(Dᶠ * ρ)

    ∂u∂z_loss = mean((∂u∂z .- truth.∂u∂z).^2)
    ∂v∂z_loss = mean((∂v∂z .- truth.∂v∂z).^2)
    ∂T∂z_loss = mean((∂T∂z .- truth.∂T∂z)[1:end-3, :].^2)
    ∂S∂z_loss = mean((∂S∂z .- truth.∂S∂z)[1:end-3, :].^2)
    ∂ρ∂z_loss = mean((∂ρ∂z .- truth.∂ρ∂z)[1:end-3, :].^2)

    return (; u=u_loss, v=v_loss, T=T_loss, S=S_loss, ρ=ρ_loss, ∂u∂z=∂u∂z_loss, ∂v∂z=∂v∂z_loss, ∂T∂z=∂T∂z_loss, ∂S∂z=∂S∂z_loss, ∂ρ∂z=∂ρ∂z_loss)
end

function loss(ps, truth, params, x₀, losses_prefactor=(; u=1, v=1, T=1, S=1, ρ=1, ∂u∂z=1, ∂v∂z=1, ∂T∂z=1, ∂S∂z=1, ∂ρ∂z=1))
    losses = individual_loss(ps, truth, params, x₀)
    return sum(values(losses) .* values(losses_prefactor))
end

function compute_density_contribution(data)
    eos = TEOS10EquationOfState()
    ρ = data.profile.ρ.unscaled[:, 1]
    T = data.profile.T.unscaled[:, 1]
    S = data.profile.S.unscaled[:, 1]

    Δρ = maximum(ρ) - minimum(ρ)
    ΔT = maximum(T) - minimum(T)
    ΔS = maximum(S) - minimum(S)

    α = mean(SeawaterPolynomials.thermal_expansion.(T, S, 0, Ref(eos)))
    β = mean(SeawaterPolynomials.haline_contraction.(T, S, 0, Ref(eos)))
    ρ₀ = eos.reference_density

    T_contribution = α * ΔT * ρ₀
    S_contribution = β * ΔS * ρ₀

    return (; T=T_contribution, S=S_contribution, ρ=Δρ)
end

function compute_loss_prefactor_density_contribution(individual_loss, contribution, S_scaling=1.0, momentum_ratio=0.25)
    u_loss, v_loss, T_loss, S_loss, ρ_loss, ∂u∂z_loss, ∂v∂z_loss, ∂T∂z_loss, ∂S∂z_loss, ∂ρ∂z_loss = values(individual_loss)
    
    total_contribution = contribution.T + contribution.S
    T_prefactor = total_contribution / contribution.T
    S_prefactor = total_contribution / contribution.S

    TS_loss = T_prefactor * T_loss + S_prefactor * S_loss

    ρ_prefactor = TS_loss / ρ_loss * 0.1 / 0.9

    uv_loss = u_loss + v_loss

    if uv_loss > eps(eltype(uv_loss))
        uv_prefactor = TS_loss / uv_loss * momentum_ratio
    else
        uv_prefactor = TS_loss * 1000
    end

    ∂T∂z_prefactor = T_prefactor
    ∂S∂z_prefactor = S_prefactor

    ∂TS∂z_loss = ∂T∂z_loss + ∂S∂z_loss

    ∂ρ∂z_prefactor = ∂TS∂z_loss / ∂ρ∂z_loss * 0.1 / 0.9

    ∂uv∂z_loss = ∂u∂z_loss + ∂v∂z_loss

    if ∂uv∂z_loss > eps(eltype(∂uv∂z_loss))
        ∂uv∂z_prefactor = ∂TS∂z_loss / ∂uv∂z_loss * momentum_ratio
    else
        ∂uv∂z_prefactor = ∂TS∂z_loss * 1000
    end

    profile_loss = uv_prefactor * u_loss + uv_prefactor * v_loss + T_prefactor * T_loss + S_prefactor * S_loss + ρ_prefactor * ρ_loss
    gradient_loss = ∂uv∂z_prefactor * ∂u∂z_loss + ∂uv∂z_prefactor * ∂v∂z_loss + ∂T∂z_prefactor * ∂T∂z_loss + ∂S∂z_prefactor * ∂S∂z_loss + ∂ρ∂z_prefactor * ∂ρ∂z_loss

    gradient_prefactor = profile_loss / gradient_loss

    ∂ρ∂z_prefactor *= gradient_prefactor
    ∂T∂z_prefactor *= gradient_prefactor
    ∂S∂z_prefactor *= gradient_prefactor
    ∂uv∂z_prefactor *= gradient_prefactor

    S_prefactor *= S_scaling
    ∂S∂z_prefactor *= S_scaling

    return (u=uv_prefactor, v=uv_prefactor, T=T_prefactor, S=S_prefactor, ρ=ρ_prefactor, ∂u∂z=∂uv∂z_prefactor, ∂v∂z=∂uv∂z_prefactor, ∂T∂z=∂T∂z_prefactor, ∂S∂z=∂S∂z_prefactor, ∂ρ∂z=∂ρ∂z_prefactor)
end

function loss_multipleics(ps, truths, params, x₀s, losses_prefactors)
    losses = [loss(ps, truth, param, x₀, loss_prefactor) for (truth, x₀, param, loss_prefactor) in zip(truths, x₀s, params, losses_prefactors)]
    return mean(losses)
end

function predict_diffusive_flux(Ris, u_hat, v_hat, T_hat, S_hat, ps_baseclosure, params)
    νs, κs = predict_diffusivities(Ris, ps_baseclosure)

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

function predict_diffusive_boundary_flux_dimensional(Ris, u_hat, v_hat, T_hat, S_hat, ps_baseclosure, params)
    _uw_diffusive, _vw_diffusive, _wT_diffusive, _wS_diffusive = predict_diffusive_flux(Ris, u_hat, v_hat, T_hat, S_hat, ps_baseclosure, params)
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

function diagnose_fields(ps, params, x₀, train_data_plot, timestep_multiple=2)
    sols = solve_NDE(ps, params, x₀, timestep_multiple)

    coarse_size = params.coarse_size
    Dᶠ = params.Dᶠ
    scaling = params.scaling

    us = inv(scaling.u).(sols.u)
    vs = inv(scaling.v).(sols.v)
    Ts = inv(scaling.T).(sols.T)
    Ss = inv(scaling.S).(sols.S)
    ρs = inv(scaling.ρ).(sols.ρ)

    eos = TEOS10EquationOfState()
    Ris_truth = hcat([calculate_Ri(u, v, ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf)) for (u, v, ρ) in zip(eachcol(train_data_plot.profile.u.unscaled), eachcol(train_data_plot.profile.v.unscaled), eachcol(train_data_plot.profile.ρ.unscaled))]...)
    Ris = hcat([calculate_Ri(u, v, ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf)) for (u, v, ρ) in zip(eachcol(us), eachcol(vs), eachcol(ρs))]...)
    
    νs, κs = predict_diffusivities(Ris, ps)

    uw_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))
    vw_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))
    wT_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))
    wS_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))

    for i in 1:size(wT_diffusive_boundarys, 2)
        uw_diffusive_boundarys[:, i], vw_diffusive_boundarys[:, i], wT_diffusive_boundarys[:, i], wS_diffusive_boundarys[:, i] = predict_diffusive_boundary_flux_dimensional(Ris[:, i], sols.u[:, i], sols.v[:, i], sols.T[:, i], sols.S[:, i], ps, params)        
    end

    uw_totals = uw_diffusive_boundarys
    vw_totals = vw_diffusive_boundarys
    wT_totals = wT_diffusive_boundarys
    wS_totals = wS_diffusive_boundarys

    fluxes = (; uw = (; total=uw_totals),
                vw = (; total=vw_totals),
                wT = (; total=wT_totals), 
                wS = (; total=wS_totals))

    diffusivities = (; ν=νs, κ=κs, Ri=Ris, Ri_truth=Ris_truth)

    sols_dimensional = (; u=us, v=vs, T=Ts, S=Ss, ρ=ρs)
    return (; sols_dimensional, fluxes, diffusivities)
end

function animate_data(train_data, diffusivities, sols_noNN, fluxes_noNN, diffusivities_noNN, index, FILE_DIR; coarse_size=32, epoch=1)
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
    axRi = CairoMakie.Axis(fig[1, 6], title="Ri", xlabel="Ri", ylabel="z (m)")
    axdiffusivity = CairoMakie.Axis(fig[2, 5], title="Diffusivity", xlabel="Diffusivity (m² s⁻¹)", ylabel="z (m)", xscale=log10)

    n = Observable(1)
    zC = train_data.metadata["zC"]
    zF = train_data.metadata["zF"]

    u_noNN = sols_noNN.u
    v_noNN = sols_noNN.v
    T_noNN = sols_noNN.T
    S_noNN = sols_noNN.S
    ρ_noNN = sols_noNN.ρ

    uw_noNN = fluxes_noNN.uw.total
    vw_noNN = fluxes_noNN.vw.total
    wT_noNN = fluxes_noNN.wT.total
    wS_noNN = fluxes_noNN.wS.total

    ulim = (find_min(train_data.profile.u.unscaled, u_noNN), find_max(train_data.profile.u.unscaled, u_noNN))
    vlim = (find_min(train_data.profile.v.unscaled, v_noNN), find_max(train_data.profile.v.unscaled, v_noNN))
    Tlim = (find_min(train_data.profile.T.unscaled, T_noNN), find_max(train_data.profile.T.unscaled, T_noNN))
    Slim = (find_min(train_data.profile.S.unscaled, S_noNN), find_max(train_data.profile.S.unscaled, S_noNN))
    ρlim = (find_min(train_data.profile.ρ.unscaled, ρ_noNN), find_max(train_data.profile.ρ.unscaled, ρ_noNN))

    uwlim = (find_min(train_data.flux.uw.column.unscaled, uw_noNN),
             find_max(train_data.flux.uw.column.unscaled, uw_noNN))
    vwlim = (find_min(train_data.flux.vw.column.unscaled, vw_noNN),
             find_max(train_data.flux.vw.column.unscaled, vw_noNN))
    wTlim = (find_min(train_data.flux.wT.column.unscaled, wT_noNN),
             find_max(train_data.flux.wT.column.unscaled, wT_noNN))
    wSlim = (find_min(train_data.flux.wS.column.unscaled, wS_noNN),
             find_max(train_data.flux.wS.column.unscaled, wS_noNN))

    Rilim = (find_min(diffusivities.Ri_truth, diffusivities_noNN.Ri,), 
             find_max(diffusivities.Ri_truth, diffusivities_noNN.Ri,),)

    diffusivitylim = (find_min(diffusivities_noNN.ν, diffusivities_noNN.κ), 
                      find_max(diffusivities_noNN.ν, diffusivities_noNN.κ),)

    u_truthₙ = @lift train_data.profile.u.unscaled[:, $n]
    v_truthₙ = @lift train_data.profile.v.unscaled[:, $n]
    T_truthₙ = @lift train_data.profile.T.unscaled[:, $n]
    S_truthₙ = @lift train_data.profile.S.unscaled[:, $n]
    ρ_truthₙ = @lift train_data.profile.ρ.unscaled[:, $n]

    uw_truthₙ = @lift train_data.flux.uw.column.unscaled[:, $n]
    vw_truthₙ = @lift train_data.flux.vw.column.unscaled[:, $n]
    wT_truthₙ = @lift train_data.flux.wT.column.unscaled[:, $n]
    wS_truthₙ = @lift train_data.flux.wS.column.unscaled[:, $n]

    u_noNNₙ = @lift u_noNN[:, $n]
    v_noNNₙ = @lift v_noNN[:, $n]
    T_noNNₙ = @lift T_noNN[:, $n]
    S_noNNₙ = @lift S_noNN[:, $n]
    ρ_noNNₙ = @lift ρ_noNN[:, $n]

    uw_noNNₙ = @lift uw_noNN[:, $n]
    vw_noNNₙ = @lift vw_noNN[:, $n]
    wT_noNNₙ = @lift wT_noNN[:, $n]
    wS_noNNₙ = @lift wS_noNN[:, $n]

    Ri_truthₙ = @lift diffusivities.Ri_truth[:, $n]
    Ri_noNNₙ = @lift diffusivities_noNN.Ri[:, $n]

    ν_noNNₙ = @lift diffusivities_noNN.ν[:, $n]
    κ_noNNₙ = @lift diffusivities_noNN.κ[:, $n]

    Qᵀ = train_data.metadata["temperature_flux"]
    Qˢ = train_data.metadata["salinity_flux"]
    f = train_data.metadata["coriolis_parameter"]
    times = train_data.times
    Nt = length(times)

    time_str = @lift "Qᵀ = $(Qᵀ) m s⁻¹ °C, Qˢ = $(Qˢ) m s⁻¹ g kg⁻¹, f = $(f) s⁻¹, Time = $(round(times[$n]/24/60^2, digits=3)) days"

    lines!(axu, u_truthₙ, zC, label="Truth")
    lines!(axu, u_noNNₙ, zC, label="Base closure only")

    lines!(axv, v_truthₙ, zC, label="Truth")
    lines!(axv, v_noNNₙ, zC, label="Base closure only")

    lines!(axT, T_truthₙ, zC, label="Truth")
    lines!(axT, T_noNNₙ, zC, label="Base closure only")

    lines!(axS, S_truthₙ, zC, label="Truth")
    lines!(axS, S_noNNₙ, zC, label="Base closure only")

    lines!(axρ, ρ_truthₙ, zC, label="Truth")
    lines!(axρ, ρ_noNNₙ, zC, label="Base closure only")

    lines!(axuw, uw_truthₙ, zF, label="Truth")
    lines!(axuw, uw_noNNₙ, zF, label="Base closure only")

    lines!(axvw, vw_truthₙ, zF, label="Truth")
    lines!(axvw, vw_noNNₙ, zF, label="Base closure only")

    lines!(axwT, wT_truthₙ, zF, label="Truth")
    lines!(axwT, wT_noNNₙ, zF, label="Base closure only")

    lines!(axwS, wS_truthₙ, zF, label="Truth")
    lines!(axwS, wS_noNNₙ, zF, label="Base closure only")

    lines!(axRi, Ri_truthₙ, zF, label="Truth")
    lines!(axRi, Ri_noNNₙ, zF, label="Base closure only")

    lines!(axdiffusivity, ν_noNNₙ, zF, label="ν, Base closure only")
    lines!(axdiffusivity, κ_noNNₙ, zF, label="κ, Base closure only")

    axislegend(axu, position=:rb)
    # axislegend(axuw, position=:rb)
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

function plot_loss(losses, FILE_DIR; epoch=1)
    colors = distinguishable_colors(10, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    
    fig = Figure(size=(1000, 600))
    axtotalloss = CairoMakie.Axis(fig[1, 1], title="Total Loss", xlabel="Iterations", ylabel="Loss", yscale=log10)
    # axindividualloss = CairoMakie.Axis(fig[1, 2], title="Individual Loss", xlabel="Iterations", ylabel="Loss", yscale=log10)

    lines!(axtotalloss, losses.total, label="Total Loss", color=colors[1])

    # lines!(axindividualloss, losses.T, label="T", color=colors[3])
    # lines!(axindividualloss, losses.S, label="S", color=colors[4])
    # lines!(axindividualloss, losses.ρ, label="ρ", color=colors[5])
    # lines!(axindividualloss, losses.∂T∂z, label="∂T∂z", color=colors[8])
    # lines!(axindividualloss, losses.∂S∂z, label="∂S∂z", color=colors[9])
    # lines!(axindividualloss, losses.∂ρ∂z, label="∂ρ∂z", color=colors[10])

    axislegend(axtotalloss, position=:lb)
    # axislegend(axindividualloss, position=:rt)
    save("$(FILE_DIR)/losses_epoch$(epoch).png", fig, px_per_unit=8)
end

ps_prior = ComponentArray(ν_conv=1.295, ν_shear=7.932e-02, m=-1.757e-01, Pr=1.193, ΔRi=0.0108)

ind_losses = [individual_loss(ps_prior, truth, param, x₀) for (truth, x₀, param) in zip(truths, x₀s, params)]
loss_prefactors = compute_loss_prefactor_density_contribution.(ind_losses, compute_density_contribution.(train_data.data), S_scaling, momentum_ratio)

prior_loss = loss_multipleics(ps_prior, truths, params, x₀s, loss_prefactors)

prior_ν_conv = constrained_gaussian("ν_conv", ps_prior.ν_conv, 0.24, -Inf, Inf)
prior_ν_shear = constrained_gaussian("ν_shear", ps_prior.ν_shear, 1.5e-2, -Inf, Inf)
prior_m = constrained_gaussian("m", ps_prior.m, 3e-2, -Inf, Inf)
prior_Pr = constrained_gaussian("Pr", ps_prior.Pr, 0.2, -Inf, Inf)
prior_ΔRi = constrained_gaussian("ΔRi", ps_prior.ΔRi, 2e-3, -Inf, Inf)

priors = combine_distributions([prior_ν_conv, prior_ν_shear, prior_m, prior_Pr, prior_ΔRi])
target = [0.]

N_ensemble = 100
N_iterations = 2000
Γ = prior_loss / 1e6 * I

ps_eki = EKP.construct_initial_ensemble(rng, priors, N_ensemble)
ensemble_kalman_process = EKP.EnsembleKalmanProcess(ps_eki, target, Γ, Inversion(); 
                                                    rng = rng, 
                                                    failure_handler_method = SampleSuccGauss(), 
                                                    scheduler = DataMisfitController(on_terminate="continue"))
G_ens = zeros(1, N_ensemble)

losses = []

wall_clock = [time_ns()]
for i in 1:N_iterations
    global ps_eki .= get_ϕ_final(priors, ensemble_kalman_process)
    Threads.@threads for j in 1:N_ensemble
        ps_ensemble = ComponentArray(ν_conv=ps_eki[1, j], ν_shear=ps_eki[2, j], m=ps_eki[3, j], Pr=ps_eki[4, j], ΔRi=ps_eki[5, j])
        G_ens[j] = loss_multipleics(ps_ensemble, truths, params, x₀s, loss_prefactors)
    end
    EKP.update_ensemble!(ensemble_kalman_process, G_ens)
    @printf("%s, Δt %s, iter %d/%d, loss average %6.10e, ν_conv %6.5e, ν_shear %6.5e, m %6.5e, Pr %6.5e, ΔRi %6.5e\n",
            Dates.now(), prettytime(1e-9 * (time_ns() - wall_clock[1])), i, N_iterations, mean(G_ens), 
            mean(ps_eki[1, :]), mean(ps_eki[2, :]), mean(ps_eki[3, :]), mean(ps_eki[4, :]), mean(ps_eki[5, :]))
    push!(losses, mean(G_ens))
    wall_clock[1] = time_ns()
end

losses = Array{Float64}(losses)
losses = (; total=losses)
final_ensemble = get_ϕ_final(priors, ensemble_kalman_process)
ensemble_mean = vec(mean(final_ensemble, dims=2))
ensemble_losses = [loss_multipleics(ComponentArray(ν_conv=p[1], ν_shear=p[2], m=p[3], Pr=p[4], ΔRi=p[5]), 
                                    truths, params, x₀s, loss_prefactors) for p in eachcol(final_ensemble)]

ensemble_min = final_ensemble[:, argmin(ensemble_losses)]

ps_final_min = ComponentArray(; ν_conv=ensemble_min[1], ν_shear=ensemble_min[2], m=ensemble_min[3], Pr=ensemble_min[4], ΔRi=ensemble_min[5])

ensemble_mean_loss = loss_multipleics(ComponentArray(ν_conv=ensemble_mean[1], ν_shear=ensemble_mean[2], m=ensemble_mean[3], Pr=ensemble_mean[4], ΔRi=ensemble_mean[5]), 
                                      truths, params, x₀s, loss_prefactors)

ps_final_mean = ComponentArray(; ν_conv=ensemble_mean[1], ν_shear=ensemble_mean[2], m=ensemble_mean[3], Pr=ensemble_mean[4], ΔRi=ensemble_mean[5])

jldsave("$(FILE_DIR)/training_results_mean.jld2", u=ps_final_mean)
jldsave("$(FILE_DIR)/training_results_min.jld2", u=ps_final_min)

plot_loss(losses, FILE_DIR; epoch=1)
for i in eachindex(params)
    sols, fluxes, diffusivities = diagnose_fields(ps_final_mean, params_plot[i], x₀s[i], train_data_plot.data[i])
    animate_data(train_data_plot.data[i], diffusivities, sols, fluxes, diffusivities, i, FILE_DIR; epoch="1_mean")
end

for i in eachindex(params)
    sols, fluxes, diffusivities = diagnose_fields(ps_final_min, params_plot[i], x₀s[i], train_data_plot.data[i])
    animate_data(train_data_plot.data[i], diffusivities, sols, fluxes, diffusivities, i, FILE_DIR; epoch="1_min")
end