using LinearAlgebra
using DiffEqBase
import SciMLBase
using Lux, ComponentArrays, Random
using Optimization
using Printf
using Enzyme
using SaltyOceanParameterizations
using SaltyOceanParameterizations.DataWrangling
using SaltyOceanParameterizations: calculate_Ri, local_Ri_ν_convectivetanh_shearlinear, local_Ri_κ_convectivetanh_shearlinear
using Oceananigans
using JLD2
using SeawaterPolynomials.TEOS10
using CairoMakie
using SparseArrays
using Optimisers
using Printf
import Dates

function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

LES_FILE_DIRS = [
    "./LES_training/linearTS_b_dTdz_0.0013_dSdz_-0.0014_QU_0.0_QB_8.0e-7_T_4.3_S_33.5_f_-0.00012_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_b_dTdz_0.013_dSdz_0.00075_QU_0.0_QB_8.0e-7_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
]

BASECLOSURE_FILE_DIR = "./training_output/local_diffusivity_convectivetanh_shearlinear_rho_SW_FC_WWSC_SWWC/training_results_5.jld2"

field_datasets = [FieldDataset(FILE_DIR, backend=OnDisk()) for FILE_DIR in LES_FILE_DIRS]

ps_baseclosure = jldopen(BASECLOSURE_FILE_DIR, "r")["u"]

timeframes = [25:10:length(data["ubar"].times) for data in field_datasets]
full_timeframes = [25:length(data["ubar"].times) for data in field_datasets]
train_data = LESDatasetsB(field_datasets, ZeroMeanUnitVarianceScaling, timeframes)
coarse_size = 32

train_data_plot = LESDatasetsB(field_datasets, ZeroMeanUnitVarianceScaling, full_timeframes)

params = [(                   f = data.coriolis.unscaled,
                     f_scaled = data.coriolis.scaled,
                            τ = data.times[end] - data.times[1],
                        N_timesteps = length(data.times),
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
                        ) for (data, plot_data) in zip(train_data.data, train_data_plot.data)]

rng = Random.default_rng(123)
NN = Chain(Dense(67, 256, leakyrelu), Dense(256, 31))

ps, st = Lux.setup(rng, NN)
ps = ps |> ComponentArray .|> Float64

ps .*= 1e-5

x₀s = [data.profile.ρ.scaled[:, 1] for data in train_data.data]

function predict_residual_flux(ρ_hat, ∂ρ∂z_hat, p, params, st, NN)
    x′ = vcat(ρ_hat, ∂ρ∂z_hat, params.wρ.scaled.top, params.f_scaled)
    
    NN_pred = NN(x′, p, st)[1]
    wρ = vcat(0., NN_pred, 0.)

    return wρ
end

predict_residual_flux(x₀s[1], rand(33), ps, params[1], st, NN)

function predict_boundary_flux(params)
    wρ = vcat(fill(params.wρ.scaled.bottom, params.coarse_size), params.wρ.scaled.top)

    return wρ
end

predict_boundary_flux(params[1])

function predict_diffusivities(Ris, ps_baseclosure)
    νs = local_Ri_ν_convectivetanh_shearlinear.(Ris, ps_baseclosure.ν_conv, ps_baseclosure.ν_shear, ps_baseclosure.m, ps_baseclosure.ΔRi)
    κs = local_Ri_κ_convectivetanh_shearlinear.(νs, ps_baseclosure.Pr)
    return νs, κs
end

eos = TEOS10EquationOfState()
Ris = calculate_Ri(zeros(coarse_size), zeros(coarse_size), train_data.data[1].profile.ρ.unscaled[:, 1], params[1].Dᶠ, params[1].g, eos.reference_density, clamp_lims=(-Inf, Inf))

predict_diffusivities(Ris, ps_baseclosure)

function solve_NDE(ps, params, x₀, ps_baseclosure, st, NN)
    eos = TEOS10EquationOfState()
    coarse_size = params.coarse_size
    timestep_multiple = 20
    Δt = (params.scaled_time[2] - params.scaled_time[1]) / timestep_multiple
    Nt_solve = (params.N_timesteps - 1) * timestep_multiple + 1
    Dᶜ_hat = params.Dᶜ_hat
    Dᶠ_hat = params.Dᶠ_hat
    Dᶠ = params.Dᶠ

    scaling = params.scaling
    τ, H = params.τ, params.H

    ρ_hat = deepcopy(x₀)
    
    sol = zeros(coarse_size, Nt_solve)
    sol[:, 1] .= x₀

    for i in 2:Nt_solve
        ρ = inv(scaling.ρ).(ρ_hat)
        ∂ρ∂z_hat = scaling.∂ρ∂z.(Dᶠ * ρ)

        Ris = calculate_Ri(zeros(coarse_size), zeros(coarse_size), ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf))
        _, κs = predict_diffusivities(Ris, ps_baseclosure)

        D = Dᶜ_hat * (-κs .* Dᶠ_hat)

        wρ_residual = predict_residual_flux(ρ_hat, ∂ρ∂z_hat, ps, params, st, NN)
        wρ_boundary = predict_boundary_flux(params)

        LHS = -τ / H^2 .* D

        RHS = - τ / H * scaling.wρ.σ / scaling.ρ.σ .* (Dᶜ_hat * (wρ_boundary .+ wρ_residual))

        sol[:, i] .= (I - Δt .* LHS) \ (ρ_hat .+ Δt .* RHS)

        ρ_hat .= sol[:, i]
    end

    return sol[:, 1:timestep_multiple:end]
end

sol = solve_NDE(ps, params[1], x₀s[1], ps_baseclosure, st, NN)

function loss(ps, truth, params, x₀, ps_baseclosure, st, NN)
    sol = solve_NDE(ps, params, x₀, ps_baseclosure, st, NN)
    return sum(abs2, sol .- truth)
end

loss(ps, train_data.data[1].profile.ρ.scaled, params[1], x₀s[1], ps_baseclosure, st, NN)

dps = deepcopy(ps) .= 0
autodiff(Enzyme.Reverse, 
         loss, 
         Active, 
         Duplicated(ps, dps), 
         Const(train_data.data[1].profile.ρ.scaled), 
         Const(params[1]), 
         Duplicated(x₀s[1], deepcopy(x₀s[1])), 
         Const(ps_baseclosure), 
         Const(st), 
         Const(NN))
#%%
rule = Optimisers.Adam()
opt_state = Optimisers.setup(rule, ps)
loss(ps, train_data.data[1].profile.ρ.scaled, params[1], x₀s[1], ps_baseclosure, st, NN)

opt_state, ps = Optimisers.update!(opt_state, ps, dps)

loss(ps, train_data.data[1].profile.ρ.scaled, params[1], x₀s[1], ps_baseclosure, st, NN)

function train_NDE(ps, params, ps_baseclosure, st, NN, train_data; n_epochs=2, n_batches=2, rule=Optimisers.Adam())
    opt_state = Optimisers.setup(rule, ps)
    dps = deepcopy(ps) .= 0
    wall_clock = [time_ns()]
    maxiter = n_epochs * n_batches
    iter = 1
    for epoch in 1:n_epochs
        for batch in 1:n_batches
            autodiff(Enzyme.Reverse, 
                     loss, 
                     Active, 
                     Duplicated(ps, dps), 
                     Const(train_data.data[batch].profile.ρ.scaled), 
                     Const(params[batch]), 
                     Duplicated(x₀s[batch], deepcopy(x₀s[batch])), 
                     Const(ps_baseclosure), 
                     Const(st), 
                     Const(NN))
            opt_state, ps = Optimisers.update!(opt_state, ps, dps)
            @printf("%s, Δt %s, iter %d/%d, loss total %6.10e, max NN weight %6.5e\n",
                     Dates.now(), prettytime(1e-9 * (time_ns() - wall_clock[1])), iter, maxiter,
                     loss(ps, train_data.data[batch].profile.ρ.scaled, params[batch], x₀s[batch], ps_baseclosure, st, NN), 
                     maximum(abs, ps))
            wall_clock = [time_ns()]
            iter += 1
            # @info loss(ps, train_data.data[batch].profile.ρ.scaled, params[batch], x₀s[batch], ps_baseclosure, st, NN)
            dps .= 0
        end
    end
    return ps
end

train_NDE(ps, params, ps_baseclosure, st, NN, train_data; n_epochs=20, n_batches=1, rule=Optimisers.Adam(1e-4))

#%%
fig = Figure()
ax = CairoMakie.Axis(fig[1, 1], xlabel="ρ", ylabel="z")
lines!(ax, sol[:, 1], params[1].zC, label="initial")
lines!(ax, sol[:, end], params[1].zC, label="final")
lines!(ax, train_data.data[1].profile.ρ.scaled[:, end], train_data.data[1].metadata["zC"], label="truth")
axislegend(ax)
display(fig)
#%%
