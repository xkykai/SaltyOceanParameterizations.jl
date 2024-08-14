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

FILE_DIR = "./training_output/NDE_FC_Qb_18simnew_2layer_128_relu_2Pr/"
@info FILE_DIR

BASECLOSURE_FILE_DIR = "./training_output/51simnew_mom_1.0_localbaseclosure_convectivetanh_shearlinear_2Pr_EKI/training_results_mean.jld2"
ps_baseclosure = jldopen(BASECLOSURE_FILE_DIR, "r")["u"]

ps, scaling_params, sts, NNs = jldopen("$(FILE_DIR)/training_results_epoch20000_end285.jld2", "r") do file
    return file["u"], file["scaling"], file["sts"], file["model"]
end

scaling_u = (μ = 0., σ = 1.)
scaling_v = (μ = 0., σ = 1.)
scaling_∂u∂z = (μ = 0., σ = 1.)
scaling_∂v∂z = (μ = 0., σ = 1.)
scaling_uw = (μ = 0., σ = 1.)
scaling_vw = (μ = 0., σ = 1.)

scaling_params_modified = (; u=scaling_u, v=scaling_v, T=scaling_params.T, S=scaling_params.S, ρ=scaling_params.ρ, 
                             ∂u∂z=scaling_∂u∂z, ∂v∂z=scaling_∂v∂z, ∂T∂z=scaling_params.∂T∂z, ∂S∂z=scaling_params.∂S∂z, ∂ρ∂z=scaling_params.∂ρ∂z, 
                             f=scaling_params.f, 
                             uw=scaling_uw, vw=scaling_vw, wT=scaling_params.wT, wS=scaling_params.wS, wb=scaling_params.wb)

scaling = construct_zeromeanunitvariance_scaling(scaling_params_modified)

LES_FILE_DIRS = [
    "./LES2/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.00015_QT_0.00045_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES2/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0001_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES2/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0005_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES2/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.0005_QT_0.0001_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES2/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.0002_QT_0.0003_QS_-3.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES2/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0003_QS_-3.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES2/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.0005_QT_0.0003_QS_-3.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    ]

field_dataset = [FieldDataset(FILE_DIR, backend=OnDisk()) for FILE_DIR in LES_FILE_DIRS]

full_timeframes = [1:length(data["ubar"].times) for data in field_dataset]

coarse_size = 32

train_data_plot = LESDatasets(field_dataset, ZeroMeanUnitVarianceScaling, full_timeframes, coarse_size; abs_f=false)

Nz = coarse_size
Lz = 256
Δz = Lz / Nz
zF = -Lz:Δz:0
zC = -Lz+Δz/2:Δz:0-Δz/2

dTdz = 0.014
dSdz = 0.0021

T_surface = 18.0
S_surface = 36.6

uw_top = -5e-4
vw_top = 0
wT_top = 3e-4
wS_top = -3e-5

f₀ = 8e-5

truth_index = 7

Δt = 10minutes
τ = 2days

ts = 0:Δt:τ
eos = TEOS10EquationOfState()

u₀s = zeros(Nz)
v₀s = zeros(Nz)
T₀s = dTdz .* zC .+ T_surface
S₀s = dSdz .* zC .+ S_surface

x₀ = (; u=scaling.u.(u₀s), v=scaling.v.(v₀s), T=scaling.T.(T₀s), S=scaling.S.(S₀s))

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
            uw = (scaled = (top=scaling.uw(uw_top), bottom=scaling.uw(0)),
                  unscaled = (top=0, bottom=0)),
            vw = (scaled = (top=scaling.vw(vw_top), bottom=scaling.vw(0)),
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
    wT_top = params.wT.unscaled.top
    wS_top = params.wS.unscaled.top

    wb_top = params.g * (α * wT_top - β * wS_top)
    convecting = wb_top > 0

    if convecting
        wb_top_scaled = params.scaling.wb(wb_top)
        common_variables = vcat(wb_top_scaled, params.f_scaled)

        wT_top = zeros(params.coarse_size+1)
        wS_top = zeros(params.coarse_size+1)

        for i in 3:params.coarse_size-1
            wT_top[i] = first(NNs.wT(vcat(∂T∂z_hat[i-1:i+1], ∂S∂z_hat[i-1:i+1], ∂ρ∂z_hat[i-1:i+1], common_variables), p.wT, sts.wT))[1]
            wS_top[i] = first(NNs.wS(vcat(∂T∂z_hat[i-1:i+1], ∂S∂z_hat[i-1:i+1], ∂ρ∂z_hat[i-1:i+1], common_variables), p.wS, sts.wS))[1]
        end

        wT_top[1:3] .= wT_top[4]
        wS_top[1:3] .= wS_top[4]
    end
    
    return wT_top, wS_top
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

        wT_residual, wS_residual = predict_residual_flux(∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat, T_top, S_top, ps, params, sts, NNs)
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

sol_u, sol_v, sol_T, sol_S, sol_ρ = solve_NDE(ps, params, x₀, ps_baseclosure, sts, NNs, length(ts), 2)
#%%
# jldopen("$(FILE_DIR)/model_inference_run.jld2", "w") do file
#     file["ps_baseclosure"] = ps_baseclosure
#     file["scaling_params"] = scaling_params
#     file["sts"] = sts
#     file["model"] = NNs
#     file["Nz"] = Nz
#     file["Lz"] = Lz
#     file["dTdz"] = dTdz
#     file["dSdz"] = dSdz
#     file["T_surface"] = T_surface
#     file["S_surface"] = S_surface
#     file["wT_top"] = wT_top
#     file["wS_top"] = wS_top
#     file["f₀"] = f₀
#     file["Δt"] = Δt
#     file["τ"] = τ
#     file["T₀s"] = T₀s
#     file["S₀s"] = S₀s
#     file["params"] = params
#     file["sol_T"] = sol_T
#     file["sol_S"] = sol_S
#     file["sol_ρ"] = sol_ρ
# end
#%%
fig = Figure(size=(2000, 600))
axu = GLMakie.Axis(fig[1, 1], xlabel="u", ylabel="z")
axv = GLMakie.Axis(fig[1, 2], xlabel="v", ylabel="z")
axT = GLMakie.Axis(fig[1, 3], xlabel="T", ylabel="z")
axS = GLMakie.Axis(fig[1, 4], xlabel="S", ylabel="z")
axρ = GLMakie.Axis(fig[1, 5], xlabel="ρ", ylabel="z")

Nt = size(sol_T, 2)

slider = GLMakie.Slider(fig[2, :], range=1:Nt)
n = slider.value

uₙ = @lift inv(scaling.u).(sol_u[:, $n])
vₙ = @lift inv(scaling.v).(sol_v[:, $n])
Tₙ = @lift inv(scaling.T).(sol_T[:, $n])
Sₙ = @lift inv(scaling.S).(sol_S[:, $n])
ρₙ = @lift inv(scaling.ρ).(sol_ρ[:, $n])

u_truthₙ = @lift train_data_plot.data[truth_index].profile.u.unscaled[:, $n]
v_truthₙ = @lift train_data_plot.data[truth_index].profile.v.unscaled[:, $n]
T_truthₙ = @lift train_data_plot.data[truth_index].profile.T.unscaled[:, $n]
S_truthₙ = @lift train_data_plot.data[truth_index].profile.S.unscaled[:, $n]
ρ_truthₙ = @lift train_data_plot.data[truth_index].profile.ρ.unscaled[:, $n]

ulim = (find_min(sol_u, train_data_plot.data[truth_index].profile.u.unscaled, -1e-8), find_max(sol_u, train_data_plot.data[truth_index].profile.u.unscaled, 1e-8))
vlim = (find_min(sol_v, train_data_plot.data[truth_index].profile.v.unscaled, -1e-8), find_max(sol_v, train_data_plot.data[truth_index].profile.v.unscaled, 1e-8))

lines!(axu, uₙ, params.zC, label="NDE")
lines!(axv, vₙ, params.zC, label="NDE")
lines!(axT, Tₙ, params.zC, label="NDE")
lines!(axS, Sₙ, params.zC, label="NDE")
lines!(axρ, ρₙ, params.zC, label="NDE")

lines!(axu, u_truthₙ, params.zC, label="LES")
lines!(axv, v_truthₙ, params.zC, label="LES")
lines!(axT, T_truthₙ, params.zC, label="LES")
lines!(axS, S_truthₙ, params.zC, label="LES")
lines!(axρ, ρ_truthₙ, params.zC, label="LES")

axislegend(axu, position=:lb)

xlims!(axu, ulim)
xlims!(axv, vlim)

title = @lift "Qᵁ = $(uw_top) m² s⁻², Qᵀ = $(wT_top) °C m s⁻¹, Qˢ = $(wS_top) psu m s⁻¹, f = $(f₀) s⁻¹, $(round(ts[$n] / 24 / 60^2, digits=2)) days"

Label(fig[0, :], title, font=:bold, tellwidth=false)

# display(fig)

GLMakie.record(fig, "./Data/model_inference_NDE_FC_Qb_18simnew_2layer_128_relu_2Pr_QU_$(uw_top)_QT_$(wT_top)_QS_$(wS_top)_T_$(T_surface)_S_$(S_surface)_f_$(f₀).mp4", 1:Nt, framerate=20) do nn
    n[] = nn
end
#%%