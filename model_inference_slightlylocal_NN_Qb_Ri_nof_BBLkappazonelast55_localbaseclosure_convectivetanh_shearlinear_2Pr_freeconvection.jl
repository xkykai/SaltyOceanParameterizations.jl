using LinearAlgebra
using Lux, ComponentArrays, Random
using Printf
using SaltyOceanParameterizations
using JLD2
using SeawaterPolynomials.TEOS10
using GLMakie
using Printf
import Dates
using Statistics
using Colors
using SeawaterPolynomials
using Oceananigans.Units
using Oceananigans.BuoyancyModels: g_Earth
using Oceananigans

FILE_DIR = "./training_output/NDE_FC_Qb_Ri_nof_BBLkappazonelast55_trainFC19new_scalingtrain59new_1layer_512_relu_10seed_2Pr"
@info FILE_DIR

BASECLOSURE_FILE_DIR = "./training_output/51simnew_6simstableRi_mom_1.0_localbaseclosure_convectivetanh_shearlinear_2Pr_unstableRi_EKI/training_results_mean.jld2"
ps_baseclosure = jldopen(BASECLOSURE_FILE_DIR, "r")["u"]

ps, scaling_params, sts, NNs = jldopen("$(FILE_DIR)/training_results_epoch12000_end285.jld2", "r") do file
    return file["u"], file["scaling"], file["sts"], file["model"]
end

const κ_conv = ps_baseclosure.ν_conv / ps_baseclosure.Pr_conv
const κ₀ = 1e-5 / ps_baseclosure.Pr_shear
const grid_point_below_kappa = 5
const grid_point_above_kappa = 5

scaling = construct_zeromeanunitvariance_scaling(scaling_params)

LES_FILE_DIRS = ["./LES2/$(file)/instantaneous_timeseries.jld2" for file in LES_suite["train59new"]]
field_datasets = [FieldDataset(FILE_DIR, backend=OnDisk()) for FILE_DIR in LES_FILE_DIRS]
full_timeframes = [25:285 for data in field_datasets]
train_data_plot = LESDatasets(field_datasets, scaling, full_timeframes, 32; abs_f=true)

truths = [(; u=data.profile.u.scaled, v=data.profile.v.scaled, T=data.profile.T.scaled, S=data.profile.S.scaled, ρ=data.profile.ρ.scaled, ∂u∂z=data.profile.∂u∂z.scaled, ∂v∂z=data.profile.∂v∂z.scaled, ∂T∂z=data.profile.∂T∂z.scaled, ∂S∂z=data.profile.∂S∂z.scaled, ∂ρ∂z=data.profile.∂ρ∂z.scaled) for data in train_data_plot.data]
params_plot = ODEParams(train_data_plot, scaling; abs_f=true)
x₀s = [(; u=data.profile.u.scaled[:, 1], v=data.profile.v.scaled[:, 1], T=data.profile.T.scaled[:, 1], S=data.profile.S.scaled[:, 1]) for data in train_data_plot.data]

# Nz = 32
# Lz = 256
# Δz = Lz / Nz
# zF = -Lz:Δz:0
# zC = -Lz+Δz/2:Δz:0-Δz/2

eos = TEOS10EquationOfState()

function predict_residual_flux(Ri, ∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat, κs, T_top, S_top, p, params, sts, NNs)
    eos = TEOS10EquationOfState()
    α = SeawaterPolynomials.thermal_expansion(T_top, S_top, 0, eos)
    β = SeawaterPolynomials.haline_contraction(T_top, S_top, 0, eos)
    wT = params.wT.unscaled.top
    wS = params.wS.unscaled.top
    coarse_size = params.coarse_size
    top_index = coarse_size + 1

    arctan_Ri = atan.(Ri)

    background_κ_index = findlast(κs[1:end-1] .≈ κ₀)
    nonbackground_κ_index = background_κ_index + 1
    last_index = ifelse(nonbackground_κ_index == top_index, top_index-1, min(background_κ_index + grid_point_above_kappa, coarse_size))
    first_index = ifelse(nonbackground_κ_index == top_index, top_index, max(background_κ_index - grid_point_below_kappa + 1, 2))

    wb_top_scaled = params.scaling.wb(params.g * (α * wT - β * wS))
    common_variables = wb_top_scaled

    wT = zeros(coarse_size+1)
    wS = zeros(coarse_size+1)

    for i in first_index:last_index
        x = vcat(arctan_Ri[i-1:i+1], ∂T∂z_hat[i-1:i+1], ∂S∂z_hat[i-1:i+1], ∂ρ∂z_hat[i-1:i+1], common_variables)
        wT[i] = first(NNs.wT(x, p.wT, sts.wT))[1]
        wS[i] = first(NNs.wS(x, p.wS, sts.wS))[1]
    end
    
    return wT, wS
end

function predict_diffusivities(Ris, ps_baseclosure)
    νs = local_Ri_ν_convectivetanh_shearlinear_2Pr.(Ris, ps_baseclosure.ν_conv, ps_baseclosure.ν_shear, ps_baseclosure.Riᶜ, ps_baseclosure.ΔRi)
    κs = local_Ri_κ_convectivetanh_shearlinear_2Pr.(Ris, ps_baseclosure.ν_conv, ps_baseclosure.ν_shear, ps_baseclosure.Riᶜ, ps_baseclosure.ΔRi, ps_baseclosure.Pr_conv, ps_baseclosure.Pr_shear)
    return νs, κs
end

function predict_diffusivities!(νs, κs, Ris, ps_baseclosure)
    νs .= local_Ri_ν_convectivetanh_shearlinear_2Pr.(Ris, ps_baseclosure.ν_conv, ps_baseclosure.ν_shear, ps_baseclosure.Riᶜ, ps_baseclosure.ΔRi)
    κs .= local_Ri_κ_convectivetanh_shearlinear_2Pr.(Ris, ps_baseclosure.ν_conv, ps_baseclosure.ν_shear, ps_baseclosure.Riᶜ, ps_baseclosure.ΔRi, ps_baseclosure.Pr_conv, ps_baseclosure.Pr_shear)
    return nothing
end

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

function predict_residual_flux_dimensional(Ri, ∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat, κs, T_top, S_top, p, params, sts, NNs)
    wT_hat, wS_hat = predict_residual_flux(Ri, ∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat, κs, T_top, S_top, p, params, sts, NNs)
    
    wT = inv(params.scaling.wT).(wT_hat)
    wS = inv(params.scaling.wS).(wS_hat)

    wT = wT .- wT[1]
    wS = wS .- wS[1]

    return wT, wS
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


function solve_NDE(ps, params, x₀, ps_baseclosure, sts, NNs, Nt, timestep_multiple=10)
    eos = TEOS10EquationOfState()
    coarse_size = params.coarse_size
    timestep = params.scaled_time[2] - params.scaled_time[1]
    Δt = timestep / timestep_multiple
    Nt_solve = (Nt - 1) * timestep_multiple + 1
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

    ∂u∂z_hat = zeros(coarse_size+1)
    ∂v∂z_hat = zeros(coarse_size+1)
    ∂T∂z_hat = zeros(coarse_size+1)
    ∂S∂z_hat = zeros(coarse_size+1)
    ∂ρ∂z_hat = zeros(coarse_size+1)

    u = zeros(coarse_size)
    v = zeros(coarse_size)
    T = zeros(coarse_size)
    S = zeros(coarse_size)
    ρ = zeros(coarse_size)
    
    u_RHS = zeros(coarse_size)
    v_RHS = zeros(coarse_size)
    T_RHS = zeros(coarse_size)
    S_RHS = zeros(coarse_size)

    wT_residual = zeros(coarse_size+1)
    wS_residual = zeros(coarse_size+1)

    uw_boundary = zeros(coarse_size+1)
    vw_boundary = zeros(coarse_size+1)
    wT_boundary = zeros(coarse_size+1)
    wS_boundary = zeros(coarse_size+1)

    νs = zeros(coarse_size+1)
    κs = zeros(coarse_size+1)

    Ris = zeros(coarse_size+1)
    
    sol_u = zeros(coarse_size, Nt_solve)
    sol_v = zeros(coarse_size, Nt_solve)
    sol_T = zeros(coarse_size, Nt_solve)
    sol_S = zeros(coarse_size, Nt_solve)
    sol_ρ = zeros(coarse_size, Nt_solve)

    sol_u[:, 1] .= u_hat
    sol_v[:, 1] .= v_hat
    sol_T[:, 1] .= T_hat
    sol_S[:, 1] .= S_hat

    LHS_uv = zeros(coarse_size, coarse_size)
    LHS_TS = zeros(coarse_size, coarse_size)

    for i in 2:Nt_solve
        u .= inv(scaling.u).(u_hat)
        v .= inv(scaling.v).(v_hat)
        T .= inv(scaling.T).(T_hat)
        S .= inv(scaling.S).(S_hat)

        ρ .= TEOS10.ρ.(T, S, 0, Ref(eos))
        ρ_hat .= scaling.ρ.(ρ)
        sol_ρ[:, i-1] .= ρ_hat

        ∂u∂z_hat .= scaling.∂u∂z.(Dᶠ * u)
        ∂v∂z_hat .= scaling.∂v∂z.(Dᶠ * v)
        ∂T∂z_hat .= scaling.∂T∂z.(Dᶠ * T)
        ∂S∂z_hat .= scaling.∂S∂z.(Dᶠ * S)
        ∂ρ∂z_hat .= scaling.∂ρ∂z.(Dᶠ * ρ)

        Ris .= calculate_Ri(u, v, ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf))
        predict_diffusivities!(νs, κs, Ris, ps_baseclosure)

        LHS_uv .= Tridiagonal(Dᶜ_hat * (-νs .* Dᶠ_hat))
        LHS_TS .= Tridiagonal(Dᶜ_hat * (-κs .* Dᶠ_hat))

        LHS_uv .*= -τ / H^2
        LHS_TS .*= -τ / H^2

        T_top = T[end]
        S_top = S[end]

        wT_residual, wS_residual = predict_residual_flux(Ris, ∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat, κs, T_top, S_top, ps, params, sts, NNs)
        predict_boundary_flux!(uw_boundary, vw_boundary, wT_boundary, wS_boundary, params)

        u_RHS .= - τ / H * scaling.uw.σ / scaling.u.σ .* (Dᶜ_hat * (uw_boundary)) .+ f * τ ./ scaling.u.σ .* v
        v_RHS .= - τ / H * scaling.vw.σ / scaling.v.σ .* (Dᶜ_hat * (vw_boundary)) .- f * τ ./ scaling.v.σ .* u
        T_RHS .= - τ / H * scaling.wT.σ / scaling.T.σ .* (Dᶜ_hat * (wT_boundary .+ wT_residual))
        S_RHS .= - τ / H * scaling.wS.σ / scaling.S.σ .* (Dᶜ_hat * (wS_boundary .+ wS_residual))

        u_hat .= (I - Δt .* LHS_uv) \ (u_hat .+ Δt .* u_RHS)
        v_hat .= (I - Δt .* LHS_uv) \ (v_hat .+ Δt .* v_RHS)
        T_hat .= (I - Δt .* LHS_TS) \ (T_hat .+ Δt .* T_RHS)
        S_hat .= (I - Δt .* LHS_TS) \ (S_hat .+ Δt .* S_RHS)

        sol_u[:, i] .= u_hat
        sol_v[:, i] .= v_hat
        sol_T[:, i] .= T_hat
        sol_S[:, i] .= S_hat
    end

    sol_ρ[:, end] .= scaling.ρ.(TEOS10.ρ.(inv(scaling.T).(T_hat), inv(scaling.S).(S_hat), 0, Ref(eos)))

    return (; u=sol_u[:, 1:timestep_multiple:end], v=sol_v[:, 1:timestep_multiple:end], T=sol_T[:, 1:timestep_multiple:end], S=sol_S[:, 1:timestep_multiple:end], ρ=sol_ρ[:, 1:timestep_multiple:end])
end

sol_u, sol_v, sol_T, sol_S, sol_ρ = solve_NDE(ps, params_plot[1], x₀s[1], ps_baseclosure, sts, NNs, length(full_timeframes[1]), 2)

function diagnose_fields(ps, params, x₀, ps_baseclosure, sts, NNs, train_data_plot, Nt, timestep_multiple=2)
    sols = solve_NDE(ps, params, x₀, ps_baseclosure, sts, NNs, Nt, timestep_multiple)

    ps_noNN = deepcopy(ps) .= 0
    sols_noNN = solve_NDE(ps_noNN, params, x₀, ps_baseclosure, sts, NNs, Nt, timestep_multiple)

    coarse_size = params.coarse_size
    Dᶠ = params.Dᶠ
    scaling = params.scaling

    us = inv(scaling.u).(sols.u)
    vs = inv(scaling.v).(sols.v)
    Ts = inv(scaling.T).(sols.T)
    Ss = inv(scaling.S).(sols.S)
    ρs = inv(scaling.ρ).(sols.ρ)

    T_tops = Ts[end, :]
    S_tops = Ss[end, :]

    us_noNN = inv(scaling.u).(sols_noNN.u)
    vs_noNN = inv(scaling.v).(sols_noNN.v)
    Ts_noNN = inv(scaling.T).(sols_noNN.T)
    Ss_noNN = inv(scaling.S).(sols_noNN.S)
    ρs_noNN = inv(scaling.ρ).(sols_noNN.ρ)
    
    ∂u∂z_hats = hcat([params.scaling.∂u∂z.(params.Dᶠ * u) for u in eachcol(us)]...)
    ∂v∂z_hats = hcat([params.scaling.∂v∂z.(params.Dᶠ * v) for v in eachcol(vs)]...)
    ∂T∂z_hats = hcat([params.scaling.∂T∂z.(params.Dᶠ * T) for T in eachcol(Ts)]...)
    ∂S∂z_hats = hcat([params.scaling.∂S∂z.(params.Dᶠ * S) for S in eachcol(Ss)]...)
    ∂ρ∂z_hats = hcat([params.scaling.∂ρ∂z.(params.Dᶠ * ρ) for ρ in eachcol(ρs)]...)
    rss_shear_hats = sqrt.(∂u∂z_hats .^ 2 .+ ∂v∂z_hats .^ 2)

    eos = TEOS10EquationOfState()
    Ris_truth = hcat([calculate_Ri(u, v, ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf)) 
                      for (u, v, ρ) in zip(eachcol(train_data_plot.profile.u.unscaled), eachcol(train_data_plot.profile.v.unscaled), eachcol(train_data_plot.profile.ρ.unscaled))]...)
    Ris = hcat([calculate_Ri(u, v, ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf)) 
                      for (u, v, ρ) in zip(eachcol(us), eachcol(vs), eachcol(ρs))]...)
    Ris_noNN = hcat([calculate_Ri(u, v, ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf)) 
                      for (u, v, ρ) in zip(eachcol(us_noNN), eachcol(vs_noNN), eachcol(ρs_noNN))]...)
    
    νs, κs = predict_diffusivities(Ris, ps_baseclosure)
    νs_noNN, κs_noNN = predict_diffusivities(Ris_noNN, ps_baseclosure)

    uw_residuals = zeros(coarse_size+1, size(Ts, 2))
    vw_residuals = zeros(coarse_size+1, size(Ts, 2))
    wT_residuals = zeros(coarse_size+1, size(Ts, 2))
    wS_residuals = zeros(coarse_size+1, size(Ts, 2))

    uw_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))
    vw_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))
    wT_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))
    wS_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))

    uw_diffusive_boundarys_noNN = zeros(coarse_size+1, size(Ts, 2))
    vw_diffusive_boundarys_noNN = zeros(coarse_size+1, size(Ts, 2))
    wT_diffusive_boundarys_noNN = zeros(coarse_size+1, size(Ts, 2))
    wS_diffusive_boundarys_noNN = zeros(coarse_size+1, size(Ts, 2))

    for i in 1:size(wT_residuals, 2)
        wT_residuals[:, i], wS_residuals[:, i] = predict_residual_flux_dimensional(Ris[:, i], ∂T∂z_hats[:, i], ∂S∂z_hats[:, i], ∂ρ∂z_hats[:, i], κs[:, i], T_tops[i], S_tops[i], ps, params, sts, NNs)
        uw_diffusive_boundarys[:, i], vw_diffusive_boundarys[:, i], wT_diffusive_boundarys[:, i], wS_diffusive_boundarys[:, i] = predict_diffusive_boundary_flux_dimensional(Ris[:, i], sols.u[:, i], sols.v[:, i], sols.T[:, i], sols.S[:, i], ps_baseclosure, params)        

        uw_diffusive_boundarys_noNN[:, i], vw_diffusive_boundarys_noNN[:, i], wT_diffusive_boundarys_noNN[:, i], wS_diffusive_boundarys_noNN[:, i] = predict_diffusive_boundary_flux_dimensional(Ris_truth[:, i], sols_noNN.u[:, i], sols_noNN.v[:, i], sols_noNN.T[:, i], sols_noNN.S[:, i], ps_baseclosure, params)
    end

    uw_totals = uw_residuals .+ uw_diffusive_boundarys
    vw_totals = vw_residuals .+ vw_diffusive_boundarys
    wT_totals = wT_residuals .+ wT_diffusive_boundarys
    wS_totals = wS_residuals .+ wS_diffusive_boundarys

    fluxes = (; uw = (; diffusive_boundary=uw_diffusive_boundarys, residual=uw_residuals, total=uw_totals),
                vw = (; diffusive_boundary=vw_diffusive_boundarys, residual=vw_residuals, total=vw_totals),
                wT = (; diffusive_boundary=wT_diffusive_boundarys, residual=wT_residuals, total=wT_totals), 
                wS = (; diffusive_boundary=wS_diffusive_boundarys, residual=wS_residuals, total=wS_totals))

    fluxes_noNN = (; uw = (; total=uw_diffusive_boundarys_noNN),
                     vw = (; total=vw_diffusive_boundarys_noNN),
                     wT = (; total=wT_diffusive_boundarys_noNN), 
                     wS = (; total=wS_diffusive_boundarys_noNN))

    diffusivities = (; ν=νs, κ=κs, Ri=Ris, Ri_truth=Ris_truth)

    diffusivities_noNN = (; ν=νs_noNN, κ=κs_noNN, Ri=Ris_noNN)

    sols_dimensional = (; u=us, v=vs, T=Ts, S=Ss, ρ=ρs)
    sols_dimensional_noNN = (; u=us_noNN, v=vs_noNN, T=Ts_noNN, S=Ss_noNN, ρ=ρs_noNN)
    return (; sols_dimensional, sols_dimensional_noNN, fluxes, fluxes_noNN, diffusivities, diffusivities_noNN)
end
#%%
sim_index = 13
full_sol = diagnose_fields(ps, params_plot[sim_index], x₀s[sim_index], ps_baseclosure, sts, NNs, train_data_plot.data[sim_index], length(full_timeframes[sim_index]), 5)

sols = full_sol.sols_dimensional
fluxes = full_sol.fluxes
diffusivities = full_sol.diffusivities
sols_noNN = full_sol.sols_dimensional_noNN
fluxes_noNN = full_sol.fluxes_noNN
diffusivities_noNN = full_sol.diffusivities_noNN
train_data = train_data_plot.data[sim_index]

fig = Figure(size=(1920, 1080))
axu = GLMakie.Axis(fig[1, 1], title="u", xlabel="u (m s⁻¹)", ylabel="z (m)")
axv = GLMakie.Axis(fig[1, 2], title="v", xlabel="v (m s⁻¹)", ylabel="z (m)")
axT = GLMakie.Axis(fig[1, 3], title="T", xlabel="T (°C)", ylabel="z (m)")
axS = GLMakie.Axis(fig[1, 4], title="S", xlabel="S (g kg⁻¹)", ylabel="z (m)")
axρ = GLMakie.Axis(fig[1, 5], title="ρ", xlabel="ρ (kg m⁻³)", ylabel="z (m)")
axuw = GLMakie.Axis(fig[2, 1], title="uw", xlabel="uw (m² s⁻²)", ylabel="z (m)")
axvw = GLMakie.Axis(fig[2, 2], title="vw", xlabel="vw (m² s⁻²)", ylabel="z (m)")
axwT = GLMakie.Axis(fig[2, 3], title="wT", xlabel="wT (m s⁻¹ °C)", ylabel="z (m)")
axwS = GLMakie.Axis(fig[2, 4], title="wS", xlabel="wS (m s⁻¹ g kg⁻¹)", ylabel="z (m)")
axRi = GLMakie.Axis(fig[1, 6], title="Ri", xlabel="arctan(Ri)", ylabel="z (m)")
axdiffusivity = GLMakie.Axis(fig[2, 5], title="Diffusivity", xlabel="Diffusivity (m² s⁻¹)", ylabel="z (m)")

slider = Slider(fig[3, :], range=1:length(full_timeframes[sim_index]))
n = slider.value
zC = train_data.metadata["zC"]
zF = train_data.metadata["zF"]

u_NDE = sols.u
v_NDE = sols.v
T_NDE = sols.T
S_NDE = sols.S
ρ_NDE = sols.ρ

uw_residual = fluxes.uw.residual
vw_residual = fluxes.vw.residual
wT_residual = fluxes.wT.residual
wS_residual = fluxes.wS.residual

uw_diffusive_boundary = fluxes.uw.diffusive_boundary
vw_diffusive_boundary = fluxes.vw.diffusive_boundary
wT_diffusive_boundary = fluxes.wT.diffusive_boundary
wS_diffusive_boundary = fluxes.wS.diffusive_boundary

uw_total = fluxes.uw.total
vw_total = fluxes.vw.total
wT_total = fluxes.wT.total
wS_total = fluxes.wS.total

u_noNN = sols_noNN.u
v_noNN = sols_noNN.v
T_noNN = sols_noNN.T
S_noNN = sols_noNN.S
ρ_noNN = sols_noNN.ρ

uw_noNN = fluxes_noNN.uw.total
vw_noNN = fluxes_noNN.vw.total
wT_noNN = fluxes_noNN.wT.total
wS_noNN = fluxes_noNN.wS.total

ulim = (find_min(u_NDE, train_data.profile.u.unscaled, u_noNN, -1e-5), find_max(u_NDE, train_data.profile.u.unscaled, u_noNN, 1e-5))
vlim = (find_min(v_NDE, train_data.profile.v.unscaled, v_noNN, -1e-5), find_max(v_NDE, train_data.profile.v.unscaled, v_noNN, 1e-5))
Tlim = (find_min(T_NDE, train_data.profile.T.unscaled, T_noNN), find_max(T_NDE, train_data.profile.T.unscaled, T_noNN))
Slim = (find_min(S_NDE, train_data.profile.S.unscaled, S_noNN), find_max(S_NDE, train_data.profile.S.unscaled, S_noNN))
ρlim = (find_min(ρ_NDE, train_data.profile.ρ.unscaled, ρ_noNN), find_max(ρ_NDE, train_data.profile.ρ.unscaled, ρ_noNN))

uwlim = (find_min(uw_residual, train_data.flux.uw.column.unscaled, -1e-7), find_max(uw_residual, train_data.flux.uw.column.unscaled, 1e-7))
vwlim = (find_min(vw_residual, train_data.flux.vw.column.unscaled, -1e-7), find_max(vw_residual, train_data.flux.vw.column.unscaled, 1e-7))
wTlim = (find_min(wT_residual, train_data.flux.wT.column.unscaled), find_max(wT_residual, train_data.flux.wT.column.unscaled))
wSlim = (find_min(wS_residual, train_data.flux.wS.column.unscaled), find_max(wS_residual, train_data.flux.wS.column.unscaled))

Rilim = (-π/2, π/2)

diffusivitylim = (find_min(diffusivities.ν, diffusivities.κ, diffusivities_noNN.ν, diffusivities_noNN.κ), 
                    find_max(diffusivities.ν, diffusivities.κ, diffusivities_noNN.ν, diffusivities_noNN.κ),)

u_truthₙ = @lift train_data.profile.u.unscaled[:, $n]
v_truthₙ = @lift train_data.profile.v.unscaled[:, $n]
T_truthₙ = @lift train_data.profile.T.unscaled[:, $n]
S_truthₙ = @lift train_data.profile.S.unscaled[:, $n]
ρ_truthₙ = @lift train_data.profile.ρ.unscaled[:, $n]

uw_truthₙ = @lift train_data.flux.uw.column.unscaled[:, $n]
vw_truthₙ = @lift train_data.flux.vw.column.unscaled[:, $n]
wT_truthₙ = @lift train_data.flux.wT.column.unscaled[:, $n]
wS_truthₙ = @lift train_data.flux.wS.column.unscaled[:, $n]

u_NDEₙ = @lift u_NDE[:, $n]
v_NDEₙ = @lift v_NDE[:, $n]
T_NDEₙ = @lift T_NDE[:, $n]
S_NDEₙ = @lift S_NDE[:, $n]
ρ_NDEₙ = @lift ρ_NDE[:, $n]

u_noNNₙ = @lift u_noNN[:, $n]
v_noNNₙ = @lift v_noNN[:, $n]
T_noNNₙ = @lift T_noNN[:, $n]
S_noNNₙ = @lift S_noNN[:, $n]
ρ_noNNₙ = @lift ρ_noNN[:, $n]

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

uw_noNNₙ = @lift uw_noNN[:, $n]
vw_noNNₙ = @lift vw_noNN[:, $n]
wT_noNNₙ = @lift wT_noNN[:, $n]
wS_noNNₙ = @lift wS_noNN[:, $n]

Ri_truthₙ = @lift atan.(diffusivities.Ri_truth[:, $n])
Riₙ = @lift atan.(diffusivities.Ri[:, $n])
Ri_noNNₙ = @lift atan.(diffusivities_noNN.Ri[:, $n])

νₙ = @lift diffusivities.ν[:, $n]
κₙ = @lift diffusivities.κ[:, $n]

ν_noNNₙ = @lift diffusivities_noNN.ν[:, $n]
κ_noNNₙ = @lift diffusivities_noNN.κ[:, $n]

Qᵁ = train_data.metadata["momentum_flux"]
Qᵀ = train_data.metadata["temperature_flux"]
Qˢ = train_data.metadata["salinity_flux"]
f = train_data.metadata["coriolis_parameter"]
times = train_data.times
Nt = length(times)

time_str = @lift "Qᵁ = $(Qᵁ) m² s⁻², Qᵀ = $(Qᵀ) m s⁻¹ °C, Qˢ = $(Qˢ) m s⁻¹ g kg⁻¹, f = $(f) s⁻¹, Time = $(round(times[$n]/24/60^2, digits=3)) days"

lines!(axu, u_truthₙ, zC, label="Truth", linewidth=4, alpha=0.5)
lines!(axu, u_noNNₙ, zC, label="Base closure only")
lines!(axu, u_NDEₙ, zC, label="NDE", color=:black)

lines!(axv, v_truthₙ, zC, label="Truth", linewidth=4, alpha=0.5)
lines!(axv, v_noNNₙ, zC, label="Base closure only")
lines!(axv, v_NDEₙ, zC, label="NDE", color=:black)

lines!(axT, T_truthₙ, zC, label="Truth", linewidth=4, alpha=0.5)
lines!(axT, T_noNNₙ, zC, label="Base closure only")
lines!(axT, T_NDEₙ, zC, label="NDE", color=:black)

lines!(axS, S_truthₙ, zC, label="Truth", linewidth=4, alpha=0.5)
lines!(axS, S_noNNₙ, zC, label="Base closure only")
lines!(axS, S_NDEₙ, zC, label="NDE", color=:black)

lines!(axρ, ρ_truthₙ, zC, label="Truth", linewidth=4, alpha=0.5)
lines!(axρ, ρ_noNNₙ, zC, label="Base closure only")
lines!(axρ, ρ_NDEₙ, zC, label="NDE", color=:black)

lines!(axuw, uw_truthₙ, zF, label="Truth", linewidth=4, alpha=0.5)
lines!(axuw, uw_noNNₙ, zF, label="Base closure only")
lines!(axuw, uw_diffusive_boundaryₙ, zF, label="Base closure")
lines!(axuw, uw_totalₙ, zF, label="NDE")
lines!(axuw, uw_residualₙ, zF, label="Residual", color=:black)

lines!(axvw, vw_truthₙ, zF, label="Truth", linewidth=4, alpha=0.5)
lines!(axvw, vw_noNNₙ, zF, label="Base closure only")
lines!(axvw, vw_diffusive_boundaryₙ, zF, label="Base closure")
lines!(axvw, vw_totalₙ, zF, label="NDE")
lines!(axvw, vw_residualₙ, zF, label="Residual", color=:black)

lines!(axwT, wT_truthₙ, zF, label="Truth", linewidth=4, alpha=0.5)
lines!(axwT, wT_noNNₙ, zF, label="Base closure only")
lines!(axwT, wT_diffusive_boundaryₙ, zF, label="Base closure")
lines!(axwT, wT_totalₙ, zF, label="NDE")
lines!(axwT, wT_residualₙ, zF, label="Residual", color=:black)

lines!(axwS, wS_truthₙ, zF, label="Truth", linewidth=4, alpha=0.5)
lines!(axwS, wS_noNNₙ, zF, label="Base closure only")
lines!(axwS, wS_diffusive_boundaryₙ, zF, label="Base closure")
lines!(axwS, wS_totalₙ, zF, label="NDE")
lines!(axwS, wS_residualₙ, zF, label="Residual", color=:black)

lines!(axRi, Ri_truthₙ, zF, label="Truth", linewidth=4, alpha=0.5)
lines!(axRi, Ri_noNNₙ, zF, label="Base closure only")
lines!(axRi, Riₙ, zF, label="NDE", color=:black)

lines!(axdiffusivity, ν_noNNₙ, zF, label="ν, Base closure only")
lines!(axdiffusivity, κ_noNNₙ, zF, label="κ, Base closure only")
lines!(axdiffusivity, νₙ, zF, label="ν, NDE")
lines!(axdiffusivity, κₙ, zF, label="κ, NDE")

axislegend(axu, position=:lb)
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
#%%