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

FILE_DIR = "./training_output/NDE_FC_Qb_nof_BBL_trainFC24new_scalingtrain54new_2layer_64_relu_2Pr"
@info FILE_DIR

BASECLOSURE_FILE_DIR = "./training_output/51simnew_6simstableRi_mom_1.0_localbaseclosure_convectivetanh_shearlinear_2Pr_unstableRi_EKI/training_results_mean.jld2"
ps_baseclosure = jldopen(BASECLOSURE_FILE_DIR, "r")["u"]

ps, scaling_params, sts, NNs = jldopen("$(FILE_DIR)/training_results_epoch8000_end215.jld2", "r") do file
    return file["u"], file["scaling"], file["sts"], file["model"]
end

scaling = construct_zeromeanunitvariance_scaling(scaling_params)

Nz = 32
Lz = 256
Δz = Lz / Nz
zF = -Lz:Δz:0
zC = -Lz+Δz/2:Δz:0-Δz/2

dTdz = 0.014
dSdz = 0.0021

T_surface = 20.0
S_surface = 36.6

wT_top = 3e-4
wS_top = 0

f₀ = 8e-5

Δt = 5minutes
τ = 3days

ts = 0:Δt:τ
eos = TEOS10EquationOfState()

T₀s = dTdz .* zC .+ T_surface
S₀s = dSdz .* zC .+ S_surface

x₀ = (; T=scaling.T.(T₀s), S=scaling.S.(S₀s))

params = (; f = f₀,
            f_scaled = scaling.f(f₀),
            τ = τ,
            scaled_time = ts ./ τ,
            zC = zC,
            H = Lz,
            g = g_Earth,
            coarse_size = Nz,
            Dᶜ = Dᶜ(Nz, Δz),
            Dᶠ = Dᶠ(Nz, Δz),
            Dᶜ_hat = Dᶜ(Nz, Δz) * Lz,
            Dᶠ_hat = Dᶠ(Nz, Δz) * Lz,
            uw = (scaled = (top=scaling.uw(0), bottom=scaling.uw(0)),
                  unscaled = (top=0, bottom=0)),
            vw = (scaled = (top=scaling.vw(0), bottom=scaling.vw(0)),
                    unscaled = (top=0, bottom=0)),
            wT = (scaled = (top=scaling.wT(wT_top), bottom=scaling.wT(0)),
                    unscaled = (top=wT_top, bottom=0)),
            wS = (scaled = (top=scaling.wS(wS_top), bottom=scaling.wS(0)),
                    unscaled = (top=wS_top, bottom=0)),
            scaling = scaling)

function predict_residual_flux(∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat, T_top, S_top, p, params, sts, NNs)
    eos = TEOS10EquationOfState()
    α = SeawaterPolynomials.thermal_expansion(T_top, S_top, 0, eos)
    β = SeawaterPolynomials.haline_contraction(T_top, S_top, 0, eos)
    wT = params.wT.unscaled.top
    wS = params.wS.unscaled.top

    ∂²ρ∂z² = params.Dᶜ_hat * ∂ρ∂z_hat
    MLD_index = argmax(abs.(∂²ρ∂z²[2:end-1])) + 1
    # BBL_index = max(MLD_index - 2, 4)
    BBL_index = ifelse(MLD_index - 3 < 4, ifelse(MLD_index == 2, params.coarse_size, 4), MLD_index - 3)

    wb_top_scaled = params.scaling.wb(params.g * (α * wT - β * wS))
    common_variables = wb_top_scaled

    wT = zeros(params.coarse_size+1)
    wS = zeros(params.coarse_size+1)

    for i in 3:params.coarse_size-1
        x = vcat(∂T∂z_hat[i-1:i+1], ∂S∂z_hat[i-1:i+1], ∂ρ∂z_hat[i-1:i+1], common_variables)
        wT[i] = ifelse(i > BBL_index, first(NNs.wT(x, p.wT, sts.wT))[1], 0)
        wS[i] = ifelse(i > BBL_index, first(NNs.wS(x, p.wS, sts.wS))[1], 0)
    end

    return wT, wS
end

function predict_boundary_flux(params)
    wT = vcat(fill(params.wT.scaled.bottom, params.coarse_size), params.wT.scaled.top)
    wS = vcat(fill(params.wS.scaled.bottom, params.coarse_size), params.wS.scaled.top)

    return wT, wS
end

function predict_boundary_flux!(wT, wS, params)
    wT[1:end-1] .= params.wT.scaled.bottom
    wS[1:end-1] .= params.wS.scaled.bottom

    wT[end] = params.wT.scaled.top
    wS[end] = params.wS.scaled.top

    return nothing
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

    T_hat = deepcopy(x₀.T)
    S_hat = deepcopy(x₀.S)
    ρ_hat = zeros(coarse_size)

    ∂T∂z_hat = zeros(coarse_size+1)
    ∂S∂z_hat = zeros(coarse_size+1)
    ∂ρ∂z_hat = zeros(coarse_size+1)

    T = zeros(coarse_size)
    S = zeros(coarse_size)
    ρ = zeros(coarse_size)
    
    T_RHS = zeros(coarse_size)
    S_RHS = zeros(coarse_size)

    wT_residual = zeros(coarse_size+1)
    wS_residual = zeros(coarse_size+1)

    wT_boundary = zeros(coarse_size+1)
    wS_boundary = zeros(coarse_size+1)

    νs = zeros(coarse_size+1)
    κs = zeros(coarse_size+1)

    Ris = zeros(coarse_size+1)
    
    sol_T = zeros(coarse_size, Nt_solve)
    sol_S = zeros(coarse_size, Nt_solve)
    sol_ρ = zeros(coarse_size, Nt_solve)

    sol_wT_residual = zeros(coarse_size+1, Nt_solve)
    sol_wS_residual = zeros(coarse_size+1, Nt_solve)

    sol_ν = zeros(coarse_size+1, Nt_solve)
    sol_κ = zeros(coarse_size+1, Nt_solve)

    sol_T[:, 1] .= T_hat
    sol_S[:, 1] .= S_hat

    LHS = zeros(coarse_size, coarse_size)

    for i in 2:Nt_solve
        T .= inv(scaling.T).(T_hat)
        S .= inv(scaling.S).(S_hat)

        ρ .= TEOS10.ρ.(T, S, 0, Ref(eos))
        ρ_hat .= scaling.ρ.(ρ)
        sol_ρ[:, i-1] .= ρ_hat

        ∂T∂z_hat .= scaling.∂T∂z.(Dᶠ * T)
        ∂S∂z_hat .= scaling.∂S∂z.(Dᶠ * S)
        ∂ρ∂z_hat .= scaling.∂ρ∂z.(Dᶠ * ρ)

        Ris .= calculate_Ri(zeros(coarse_size), zeros(coarse_size), ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf))
        predict_diffusivities!(νs, κs, Ris, ps_baseclosure)

        LHS .= Tridiagonal(Dᶜ_hat * (-κs .* Dᶠ_hat))
        # LHS .= Dᶜ_hat * (-κs .* Dᶠ_hat)
        LHS .*= -τ / H^2

        T_top = T[end]
        S_top = S[end]

        wT_residual, wS_residual = predict_residual_flux(∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat, T_top, S_top, ps, params, sts, NNs)
        predict_boundary_flux!(wT_boundary, wS_boundary, params)

        T_RHS .= - τ / H * scaling.wT.σ / scaling.T.σ .* (Dᶜ_hat * (wT_boundary .+ wT_residual))
        S_RHS .= - τ / H * scaling.wS.σ / scaling.S.σ .* (Dᶜ_hat * (wS_boundary .+ wS_residual))

        T_hat .= (I - Δt .* LHS) \ (T_hat .+ Δt .* T_RHS)
        S_hat .= (I - Δt .* LHS) \ (S_hat .+ Δt .* S_RHS)

        sol_T[:, i] .= T_hat
        sol_S[:, i] .= S_hat

        sol_wT_residual[:, i] .= wT_residual
        sol_wS_residual[:, i] .= wS_residual

        sol_ν[:, i] .= νs
        sol_κ[:, i] .= κs
    end

    sol_ρ[:, end] .= scaling.ρ.(TEOS10.ρ.(inv(scaling.T).(T_hat), inv(scaling.S).(S_hat), 0, Ref(eos)))

    T = inv(scaling.T).(sol_T[:, 1:timestep_multiple:end])
    S = inv(scaling.S).(sol_S[:, 1:timestep_multiple:end])
    ρ = inv(scaling.ρ).(sol_ρ[:, 1:timestep_multiple:end])

    wT_residual_unscaled = scaling.wT.σ .* sol_wT_residual[:, 1:timestep_multiple:end]
    wS_residual_unscaled = scaling.wS.σ .* sol_wS_residual[:, 1:timestep_multiple:end]

    return (; T, S, ρ, wT_residual_unscaled, wS_residual_unscaled, sol_ν, sol_κ)
end

sol_T, sol_S, sol_ρ, sol_wT_residual_unscaled, sol_wS_residual_unscaled, sol_ν, sol_κ = solve_NDE(ps, params, x₀, ps_baseclosure, sts, NNs, length(ts), 1)
#%%
jldopen("$(FILE_DIR)/model_inference_run_nof_BBL.jld2", "w") do file
    file["ps_baseclosure"] = ps_baseclosure
    file["scaling_params"] = scaling_params
    file["sts"] = sts
    file["model"] = NNs
    file["Nz"] = Nz
    file["Lz"] = Lz
    file["dTdz"] = dTdz
    file["dSdz"] = dSdz
    file["T_surface"] = T_surface
    file["S_surface"] = S_surface
    file["wT_top"] = wT_top
    file["wS_top"] = wS_top
    file["f₀"] = f₀
    file["Δt"] = Δt
    file["τ"] = τ
    file["T₀s"] = T₀s
    file["S₀s"] = S₀s
    file["params"] = params
    file["sol_T"] = sol_T
    file["sol_S"] = sol_S
    file["sol_ρ"] = sol_ρ
    file["sol_wT_residual_unscaled"] = sol_wT_residual_unscaled
    file["sol_wS_residual_unscaled"] = sol_wS_residual_unscaled
    file["sol_ν"] = sol_ν
    file["sol_κ"] = sol_κ
end

fig = Figure(size=(900, 600))
axT = GLMakie.Axis(fig[1, 1], xlabel="T", ylabel="z")
axS = GLMakie.Axis(fig[1, 2], xlabel="S", ylabel="z")
axρ = GLMakie.Axis(fig[1, 3], xlabel="ρ", ylabel="z")

lines!(axT, sol_T[:, 1], params.zC, label="initial")
lines!(axT, sol_T[:, end], params.zC, label="final")

lines!(axS, sol_S[:, 1], params.zC, label="initial")
lines!(axS, sol_S[:, end], params.zC, label="final")

lines!(axρ, sol_ρ[:, 1], params.zC, label="initial")
lines!(axρ, sol_ρ[:, end], params.zC, label="final")

axislegend(axT, orientation=:vertical, position=:rb)
display(fig)
#%%