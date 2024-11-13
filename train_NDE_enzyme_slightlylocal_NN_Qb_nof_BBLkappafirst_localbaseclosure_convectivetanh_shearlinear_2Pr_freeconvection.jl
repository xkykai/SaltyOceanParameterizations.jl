using LinearAlgebra
using Lux, ComponentArrays, Random
using Printf
using Enzyme
using SaltyOceanParameterizations
using Oceananigans
using JLD2
using SeawaterPolynomials.TEOS10
using CairoMakie
using Optimisers
using Printf
import Dates
using Statistics
using Colors
using ArgParse
using SeawaterPolynomials

function parse_commandline()
    s = ArgParseSettings()
  
    @add_arg_table! s begin
      "--hidden_layer_size"
        help = "Size of hidden layer"
        arg_type = Int64
        default = 64
      "--hidden_layer"
        help = "Number of hidden layers"
        arg_type = Int64
        default = 2
      "--activation"
        help = "Activation function"
        arg_type = String
        default = "relu"
      "--S_scaling"
        help = "Scaling factor for S"
        arg_type = Float64
        default = 1.0
      "--point_below_kappa"
        help = "Number of Grid points below convective kappa to turn off NN fluxes"
        arg_type = Int64
        default = 5
      "--point_above_kappa"
        help = "Number of Grid points above background kappa to turn off NN fluxes"
        arg_type = Int64
        default = 5
      "--random_seed"
        help = "Random seed"
        arg_type = Int64
        default = 123
      "--NN_grid_points"
        help = "Number of grid points to use for NN inputs"
        arg_type = Int64
        default = 3
    end
    return parse_args(s)
end

args = parse_commandline()

const hidden_layer_size = args["hidden_layer_size"]
const N_hidden_layer = args["hidden_layer"]

if args["activation"] == "relu"
    const activation = relu
elseif args["activation"] == "leakyrelu"
    const activation = leakyrelu
elseif args["activation"] == "swish"
    const activation = swish
else
    error("Activation function not recognized")
end

const S_scaling = args["S_scaling"]
const grid_point_below_kappa = args["point_below_kappa"]
const grid_point_above_kappa = args["point_above_kappa"]
seed = args["random_seed"]

const NN_grid_points = args["NN_grid_points"]

LES_suite_name = "trainFC26new"
scaling_LES_suite_name = "train62newnohighrotation"
validation_LES_suite_name = "validate37new"

FILE_DIR = "./training_output/NDE$(NN_grid_points)_FC_Qb_nof_BBLkappafirst$(grid_point_below_kappa)$(grid_point_above_kappa)_$(LES_suite_name)_scaling$(scaling_LES_suite_name)_$(args["hidden_layer"])layer_$(args["hidden_layer_size"])_$(args["activation"])_$(seed)seed_2Pr_test"
mkpath(FILE_DIR)
@info FILE_DIR

BASECLOSURE_FILE_DIR = "./training_output/51simnew_6simstableRi_mom_1.0_localbaseclosure_convectivetanh_shearlinear_2Pr_unstableRi_EKI/training_results_mean.jld2"
ps_baseclosure = jldopen(BASECLOSURE_FILE_DIR, "r")["u"]

coarse_size = 32
const Riᶜ = ps_baseclosure.Riᶜ

# Just to obtain scaling from the scaling suite
LES_FILE_DIRS_scaling = ["./LES2/$(file)/instantaneous_timeseries.jld2" for file in LES_suite[scaling_LES_suite_name]]
field_datasets_scaling = [FieldDataset(FILE_DIR, backend=OnDisk()) for FILE_DIR in LES_FILE_DIRS_scaling]
full_timeframes = [25:length(data["ubar"].times) for data in field_datasets_scaling]
train_data_plot = LESDatasets(field_datasets_scaling, ZeroMeanUnitVarianceScaling, full_timeframes, coarse_size; abs_f=true)
scaling = train_data_plot.scaling

# Now loading actual training data
LES_FILE_DIRS = ["./LES2/$(file)/instantaneous_timeseries.jld2" for file in LES_suite[LES_suite_name]]
field_datasets = [FieldDataset(FILE_DIR, backend=OnDisk()) for FILE_DIR in LES_FILE_DIRS]
timeframes = [25:10:length(data["ubar"].times) for data in field_datasets]
train_data = LESDatasets(field_datasets, scaling, timeframes, coarse_size; abs_f=true)

LES_FILE_DIRS_validation = ["./LES2/$(file)/instantaneous_timeseries.jld2" for file in LES_suite[validation_LES_suite_name]]
field_datasets_validation = [FieldDataset(FILE_DIR, backend=OnDisk()) for FILE_DIR in LES_FILE_DIRS_validation]
timeframes_validation = [25:length(data["ubar"].times) for data in field_datasets_validation]
validation_data = LESDatasets(field_datasets_validation, scaling, timeframes_validation, coarse_size; abs_f=true)

truths = [(;    T = data.profile.T.scaled, 
                S = data.profile.S.scaled, 
                ρ = data.profile.ρ.scaled, 
             ∂T∂z = data.profile.∂T∂z.scaled, 
             ∂S∂z = data.profile.∂S∂z.scaled, 
             ∂ρ∂z = data.profile.∂ρ∂z.scaled) for data in train_data.data]

truths_validation = [(; u = data.profile.u.scaled, 
                        v = data.profile.v.scaled, 
                        T = data.profile.T.scaled, 
                        S = data.profile.S.scaled, 
                        ρ = data.profile.ρ.scaled, 
                     ∂u∂z = data.profile.∂u∂z.scaled,
                     ∂v∂z = data.profile.∂v∂z.scaled, 
                     ∂T∂z = data.profile.∂T∂z.scaled, 
                     ∂S∂z = data.profile.∂S∂z.scaled, 
                     ∂ρ∂z = data.profile.∂ρ∂z.scaled) for data in validation_data.data]

params = ODEParams(train_data, scaling; abs_f=true)
params_plot = ODEParams(train_data_plot, scaling; abs_f=true)
params_validation = ODEParams(validation_data, scaling; abs_f=true)

rng = Random.default_rng(seed)

#%%
const window_split_size = NN_grid_points ÷ 2

NN_layers = vcat(Dense(3NN_grid_points + 1, hidden_layer_size, activation), [Dense(hidden_layer_size, hidden_layer_size, activation) for _ in 1:N_hidden_layer-1]..., Dense(hidden_layer_size, 1))

wT_NN = Chain(NN_layers...)
wS_NN = Chain(NN_layers...)

ps_wT, st_wT = Lux.setup(rng, wT_NN)
ps_wS, st_wS = Lux.setup(rng, wS_NN)

ps_wT = ps_wT |> ComponentArray .|> Float64
ps_wS = ps_wS |> ComponentArray .|> Float64

ps_wT .= glorot_uniform(rng, Float64, length(ps_wT))
ps_wS .= glorot_uniform(rng, Float64, length(ps_wS))

# ps_wT .*= 1e-5
# ps_wS .*= 1e-5

x₀s = [(; T=data.profile.T.scaled[:, 1], S=data.profile.S.scaled[:, 1]) for data in train_data.data]
x₀s_plot = [(; u=data.profile.u.scaled[:, 1], v=data.profile.v.scaled[:, 1], T=data.profile.T.scaled[:, 1], S=data.profile.S.scaled[:, 1]) for data in train_data_plot.data]
x₀s_validation = [(; u=data.profile.u.scaled[:, 1], v=data.profile.v.scaled[:, 1], T=data.profile.T.scaled[:, 1], S=data.profile.S.scaled[:, 1]) for data in validation_data.data]

ps = ComponentArray(; wT=ps_wT, wS=ps_wS)
NNs = (wT=wT_NN, wS=wS_NN)
sts = (wT=st_wT, wS=st_wS)

scaling_params = write_scaling_params(scaling)

function predict_residual_flux(∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat, Ri, T_top, S_top, p, params, sts, NNs)
    eos = TEOS10EquationOfState()
    α = SeawaterPolynomials.thermal_expansion(T_top, S_top, 0, eos)
    β = SeawaterPolynomials.haline_contraction(T_top, S_top, 0, eos)
    wT = params.wT.unscaled.top
    wS = params.wS.unscaled.top
    coarse_size = params.coarse_size
    top_index = coarse_size + 1

    background_κ_index = findfirst(Ri[2:end] .< Riᶜ)
    nonbackground_κ_index = background_κ_index + 1
    last_index = coarse_size
    first_index = ifelse(nonbackground_κ_index == top_index, top_index, max(background_κ_index - grid_point_below_kappa + 1, 2))

    wb_top_scaled = params.scaling.wb(params.g * (α * wT - β * wS))
    common_variables = wb_top_scaled

    wT = zeros(coarse_size+1)
    wS = zeros(coarse_size+1)

    for i in first_index:last_index
        i_min = i - window_split_size
        i_max = i + window_split_size

        if i_max >= top_index
            n_repeat = i_max - coarse_size
            ∂T∂z_hat_i = vcat(∂T∂z_hat[i_min:coarse_size], repeat(∂T∂z_hat[coarse_size:coarse_size], n_repeat))
            ∂S∂z_hat_i = vcat(∂S∂z_hat[i_min:coarse_size], repeat(∂S∂z_hat[coarse_size:coarse_size], n_repeat))
            ∂ρ∂z_hat_i = vcat(∂ρ∂z_hat[i_min:coarse_size], repeat(∂ρ∂z_hat[coarse_size:coarse_size], n_repeat))
            x = vcat(∂T∂z_hat_i, ∂S∂z_hat_i, ∂ρ∂z_hat_i, common_variables)
        elseif i_min <= 1
            n_repeat = 2 - i_min
            ∂T∂z_hat_i = vcat(repeat(∂T∂z_hat[2:2], n_repeat), ∂T∂z_hat[2:i_max])
            ∂S∂z_hat_i = vcat(repeat(∂S∂z_hat[2:2], n_repeat), ∂S∂z_hat[2:i_max])
            ∂ρ∂z_hat_i = vcat(repeat(∂ρ∂z_hat[2:2], n_repeat), ∂ρ∂z_hat[2:i_max])
            x = vcat(∂T∂z_hat_i, ∂S∂z_hat_i, ∂ρ∂z_hat_i, common_variables)
        else
            x = vcat(∂T∂z_hat[i_min:i_max], ∂S∂z_hat[i_min:i_max], ∂ρ∂z_hat[i_min:i_max], common_variables)
        end

        wT[i] = first(NNs.wT(x, p.wT, sts.wT))[1]
        wS[i] = first(NNs.wS(x, p.wS, sts.wS))[1]
    end
    
    return wT, wS
end

function predict_boundary_flux(params)
    wT = vcat(fill(params.wT.scaled.bottom, params.coarse_size), params.wT.scaled.top)
    wS = vcat(fill(params.wS.scaled.bottom, params.coarse_size), params.wS.scaled.top)

    return wT, wS
end

function predict_boundary_flux_withmomentum(params)
    uw = vcat(fill(params.uw.scaled.bottom, params.coarse_size), params.uw.scaled.top)
    vw = vcat(fill(params.vw.scaled.bottom, params.coarse_size), params.vw.scaled.top)
    wT = vcat(fill(params.wT.scaled.bottom, params.coarse_size), params.wT.scaled.top)
    wS = vcat(fill(params.wS.scaled.bottom, params.coarse_size), params.wS.scaled.top)

    return uw, vw, wT, wS
end

function predict_boundary_flux!(wT, wS, params)
    wT[1:end-1] .= params.wT.scaled.bottom
    wS[1:end-1] .= params.wS.scaled.bottom

    wT[end] = params.wT.scaled.top
    wS[end] = params.wS.scaled.top

    return nothing
end

function predict_boundary_flux_withmomentum(params)
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

    νs = ones(coarse_size+1)
    κs = ones(coarse_size+1)

    Ris = zeros(coarse_size+1)
    
    sol_T = zeros(coarse_size, Nt_solve)
    sol_S = zeros(coarse_size, Nt_solve)
    sol_ρ = zeros(coarse_size, Nt_solve)

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

        wT_residual, wS_residual = predict_residual_flux(∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat, Ris, T_top, S_top, ps, params, sts, NNs)
        predict_boundary_flux!(wT_boundary, wS_boundary, params)

        T_RHS .= - τ / H * scaling.wT.σ / scaling.T.σ .* (Dᶜ_hat * (wT_boundary .+ wT_residual))
        S_RHS .= - τ / H * scaling.wS.σ / scaling.S.σ .* (Dᶜ_hat * (wS_boundary .+ wS_residual))

        T_hat .= (I - Δt .* LHS) \ (T_hat .+ Δt .* T_RHS)
        S_hat .= (I - Δt .* LHS) \ (S_hat .+ Δt .* S_RHS)

        sol_T[:, i] .= T_hat
        sol_S[:, i] .= S_hat
    end

    sol_ρ[:, end] .= scaling.ρ.(TEOS10.ρ.(inv(scaling.T).(T_hat), inv(scaling.S).(S_hat), 0, Ref(eos)))

    return (; T=sol_T[:, 1:timestep_multiple:end], S=sol_S[:, 1:timestep_multiple:end], ρ=sol_ρ[:, 1:timestep_multiple:end])
end

function solve_NDE_withmomentum(ps, params, x₀, ps_baseclosure, sts, NNs, Nt, timestep_multiple=2)
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

        wT_residual, wS_residual = predict_residual_flux(∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat, Ris, T_top, S_top, ps, params, sts, NNs)
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


#%%
sol_index = 1
truth = truths[sol_index]
sol_T, sol_S, sol_ρ = solve_NDE(ps, params[sol_index], x₀s[sol_index], ps_baseclosure, sts, NNs, length(25:10:45))
fig = Figure(size=(900, 600))
axT = CairoMakie.Axis(fig[1, 1], xlabel="T", ylabel="z")
axS = CairoMakie.Axis(fig[1, 2], xlabel="S", ylabel="z")
axρ = CairoMakie.Axis(fig[1, 3], xlabel="ρ", ylabel="z")

lines!(axT, sol_T[:, 1], params[1].zC, label="initial")
lines!(axT, sol_T[:, end], params[1].zC, label="final")
lines!(axT, truth.T[:, length(25:10:45)], train_data.data[1].metadata["zC"], label="truth")

lines!(axS, sol_S[:, 1], params[1].zC, label="initial")
lines!(axS, sol_S[:, end], params[1].zC, label="final")
lines!(axS, truth.S[:, length(25:10:45)], train_data.data[1].metadata["zC"], label="truth")

lines!(axρ, sol_ρ[:, 1], params[1].zC, label="initial")
lines!(axρ, sol_ρ[:, end], params[1].zC, label="final")
lines!(axρ, truth.ρ[:, length(25:10:45)], train_data.data[1].metadata["zC"], label="truth")

axislegend(axT, orientation=:vertical, position=:rb)
# save("$(FILE_DIR)/initial_final_truth.png", fig)
display(fig)
#%%
sol_index = 1
truth = truths_validation[sol_index]
sol_u, sol_v, sol_T, sol_S, sol_ρ = solve_NDE_withmomentum(ps, params_validation[sol_index], x₀s_validation[sol_index], ps_baseclosure, sts, NNs, length(25:285))

fig = Figure(size=(1800, 600))
axu = CairoMakie.Axis(fig[1, 1], xlabel="u", ylabel="z")
axv = CairoMakie.Axis(fig[1, 2], xlabel="v", ylabel="z")
axT = CairoMakie.Axis(fig[1, 3], xlabel="T", ylabel="z")
axS = CairoMakie.Axis(fig[1, 4], xlabel="S", ylabel="z")
axρ = CairoMakie.Axis(fig[1, 5], xlabel="ρ", ylabel="z")

lines!(axu, sol_u[:, 1], params[1].zC, label="initial")
lines!(axu, sol_u[:, end], params[1].zC, label="final")
lines!(axu, truth.u[:, length(25:285)], train_data.data[1].metadata["zC"], label="truth")

lines!(axv, sol_v[:, 1], params[1].zC, label="initial")
lines!(axv, sol_v[:, end], params[1].zC, label="final")
lines!(axv, truth.v[:, length(25:285)], train_data.data[1].metadata["zC"], label="truth")

lines!(axT, sol_T[:, 1], params[1].zC, label="initial")
lines!(axT, sol_T[:, end], params[1].zC, label="final")
lines!(axT, truth.T[:, length(25:285)], train_data.data[1].metadata["zC"], label="truth")

lines!(axS, sol_S[:, 1], params[1].zC, label="initial")
lines!(axS, sol_S[:, end], params[1].zC, label="final")
lines!(axS, truth.S[:, length(25:285)], train_data.data[1].metadata["zC"], label="truth")

lines!(axρ, sol_ρ[:, 1], params[1].zC, label="initial")
lines!(axρ, sol_ρ[:, end], params[1].zC, label="final")
lines!(axρ, truth.ρ[:, length(25:285)], train_data.data[1].metadata["zC"], label="truth")

axislegend(axT, orientation=:vertical, position=:rb)
# save("$(FILE_DIR)/NDE_Qb_$(sol_index)_sol.png", fig)
display(fig)
#%%
function individual_loss(ps, truth, params, x₀, ps_baseclosure, sts, NNs, Nt, tstart=1, timestep_multiple=10)
    Dᶠ = params.Dᶠ
    scaling = params.scaling
    sol_T, sol_S, sol_ρ = solve_NDE(ps, params, x₀, ps_baseclosure, sts, NNs, Nt, timestep_multiple)

    T_loss = mean((sol_T .- truth.T[:, tstart:tstart+Nt-1]).^2)
    S_loss = mean((sol_S .- truth.S[:, tstart:tstart+Nt-1]).^2)
    ρ_loss = mean((sol_ρ .- truth.ρ[:, tstart:tstart+Nt-1]).^2)

    T = inv(scaling.T).(sol_T)
    S = inv(scaling.S).(sol_S)
    ρ = inv(scaling.ρ).(sol_ρ)

    ∂T∂z = scaling.∂T∂z.(Dᶠ * T)
    ∂S∂z = scaling.∂S∂z.(Dᶠ * S)
    ∂ρ∂z = scaling.∂ρ∂z.(Dᶠ * ρ)

    ∂T∂z_loss = mean((∂T∂z[1:end-4,:] .- truth.∂T∂z[1:end-4, tstart:tstart+Nt-1]).^2)
    ∂S∂z_loss = mean((∂S∂z[1:end-4,:] .- truth.∂S∂z[1:end-4, tstart:tstart+Nt-1]).^2)
    ∂ρ∂z_loss = mean((∂ρ∂z[1:end-4,:] .- truth.∂ρ∂z[1:end-4, tstart:tstart+Nt-1]).^2)

    return (; T=T_loss, S=S_loss, ρ=ρ_loss, ∂T∂z=∂T∂z_loss, ∂S∂z=∂S∂z_loss, ∂ρ∂z=∂ρ∂z_loss)
end

function loss(ps, truth, params, x₀, ps_baseclosure, sts, NNs, Nt, tstart=1, timestep_multiple=10, losses_prefactor=(; T=1, S=1, ρ=1, ∂T∂z=1, ∂S∂z=1, ∂ρ∂z=1))
    losses = individual_loss(ps, truth, params, x₀, ps_baseclosure, sts, NNs, Nt, tstart, timestep_multiple)
    return sum(values(losses) .* values(losses_prefactor))
end

loss(ps, truths[1], params[1], x₀s[1], ps_baseclosure, sts, NNs, length(25:10:45))

# dps = deepcopy(ps) .= 0
# autodiff(Enzyme.ReverseWithPrimal, 
#          loss, 
#          Active, 
#          DuplicatedNoNeed(ps, dps), 
#          Const(truths[1]), 
#          Const(params[1]), 
#          DuplicatedNoNeed(x₀s[1], deepcopy(x₀s[1])), 
#          Const(ps_baseclosure), 
#          Const(sts), 
#          Const(NNs),
#          Const(length(25:10:45)))

function compute_loss_prefactor_density_contribution(individual_loss, contribution, S_scaling=1.0)
    T_loss, S_loss, ρ_loss, ∂T∂z_loss, ∂S∂z_loss, ∂ρ∂z_loss = values(individual_loss)

    T_contribution = max(contribution.T, 1e-2)
    S_contribution = max(contribution.S, 1e-2)

    total_contribution = T_contribution + S_contribution
    T_prefactor = total_contribution / T_contribution
    S_prefactor = total_contribution / S_contribution

    TS_loss = T_prefactor * T_loss + S_prefactor * S_loss

    ρ_prefactor = TS_loss / ρ_loss * 0.1 / 0.9
    ∂T∂z_prefactor = T_prefactor
    ∂S∂z_prefactor = S_prefactor

    ∂TS∂z_loss = ∂T∂z_loss + ∂S∂z_loss
    ∂ρ∂z_prefactor = ∂TS∂z_loss / ∂ρ∂z_loss * 0.1 / 0.9

    profile_loss = T_prefactor * T_loss + S_prefactor * S_loss + ρ_prefactor * ρ_loss
    gradient_loss = ∂T∂z_prefactor * ∂T∂z_loss + ∂S∂z_prefactor * ∂S∂z_loss + ∂ρ∂z_prefactor * ∂ρ∂z_loss

    gradient_prefactor = profile_loss / gradient_loss

    ∂ρ∂z_prefactor *= gradient_prefactor
    ∂T∂z_prefactor *= gradient_prefactor
    ∂S∂z_prefactor *= gradient_prefactor

    S_prefactor *= S_scaling
    ∂S∂z_prefactor *= S_scaling

    return (T=T_prefactor, S=S_prefactor, ρ=ρ_prefactor, ∂T∂z=∂T∂z_prefactor, ∂S∂z=∂S∂z_prefactor, ∂ρ∂z=∂ρ∂z_prefactor)
end

ind_losses = [individual_loss(ps, truth, param, x₀, ps_baseclosure, sts, NNs, length(timeframe)) for (truth, x₀, param, timeframe) in zip(truths, x₀s, params, timeframes)]

loss_prefactors = compute_loss_prefactor_density_contribution.(ind_losses, compute_density_contribution.(train_data.data), S_scaling)

function loss_multipleics(ps, truths, params, x₀s, ps_baseclosure, sts, NNs, losses_prefactor, Nt, tstart=1, timestep_multiple=10)
    losses = [loss(ps, truth, param, x₀, ps_baseclosure, sts, NNs, Nt, tstart, timestep_multiple, loss_prefactor) for (truth, x₀, param, loss_prefactor) in zip(truths, x₀s, params, losses_prefactor)]
    return mean(losses)
end

# loss_multipleics(ps, [truths[1]], [params[1]], [x₀s[1]], ps_baseclosure, sts, NNs, [loss_prefactors[1]], length(25:10:45))

# dps = deepcopy(ps) .= 0
# autodiff(Enzyme.ReverseWithPrimal, 
#          loss_multipleics, 
#          Active, 
#          DuplicatedNoNeed(ps, dps), 
#          DuplicatedNoNeed([truths[1]], deepcopy([truths[1]])), 
#          DuplicatedNoNeed([params[1]], deepcopy([params[1]])), 
#          DuplicatedNoNeed([x₀s[1]], deepcopy([x₀s[1]])), 
#          DuplicatedNoNeed(ps_baseclosure, deepcopy(ps_baseclosure)), 
#          Const(sts), 
#          Const(NNs), 
#          DuplicatedNoNeed([loss_prefactors[1]], deepcopy([loss_prefactors[1]])),
#          Const(length(25:10:45)))

# dps = deepcopy(ps) .= 0
# autodiff(Enzyme.ReverseWithPrimal, 
#          loss_multipleics, 
#          Active, 
#          DuplicatedNoNeed(ps, dps), 
#          DuplicatedNoNeed(truths[1:2], deepcopy(truths[1:2])), 
#          DuplicatedNoNeed(params[1:2], deepcopy(params[1:2])), 
#          DuplicatedNoNeed(x₀s[1:2], deepcopy(x₀s[1:2])), 
#          DuplicatedNoNeed(ps_baseclosure, deepcopy(ps_baseclosure)), 
#          Const(sts), 
#          Const(NNs), 
#          DuplicatedNoNeed(loss_prefactors[1:2], deepcopy(loss_prefactors[1:2])),
#          Const(length(25:10:45)))

function predict_residual_flux_dimensional(∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat, Ri, T_top, S_top, p, params, sts, NNs)
    wT_hat, wS_hat = predict_residual_flux(∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat, Ri, T_top, S_top, p, params, sts, NNs)
    
    wT = inv(params.scaling.wT).(wT_hat)
    wS = inv(params.scaling.wS).(wS_hat)

    wT = wT .- wT[1]
    wS = wS .- wS[1]

    return wT, wS
end

function predict_diffusive_flux(Ris, T_hat, S_hat, ps_baseclosure, params)
    _, κs = predict_diffusivities(Ris, ps_baseclosure)

    ∂T∂z_hat = params.Dᶠ_hat * T_hat
    ∂S∂z_hat = params.Dᶠ_hat * S_hat

    wT_diffusive = -κs .* ∂T∂z_hat
    wS_diffusive = -κs .* ∂S∂z_hat
    return wT_diffusive, wS_diffusive
end

function predict_diffusive_boundary_flux_dimensional(Ris, T_hat, S_hat, ps_baseclosure, params)
    _wT_diffusive, _wS_diffusive = predict_diffusive_flux(Ris, T_hat, S_hat, ps_baseclosure, params)
    _wT_boundary, _wS_boundary = predict_boundary_flux(params)

    wT_diffusive = params.scaling.T.σ / params.H .* _wT_diffusive
    wS_diffusive = params.scaling.S.σ / params.H .* _wS_diffusive

    wT_boundary = inv(params.scaling.wT).(_wT_boundary)
    wS_boundary = inv(params.scaling.wS).(_wS_boundary)

    wT = wT_diffusive .+ wT_boundary
    wS = wS_diffusive .+ wS_boundary

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
    _uw_boundary, _vw_boundary, _wT_boundary, _wS_boundary = predict_boundary_flux_withmomentum(params)

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

function diagnose_fields(ps, params, x₀, ps_baseclosure, sts, NNs, train_data_plot, Nt, timestep_multiple=2)
    sols = solve_NDE_withmomentum(ps, params, x₀, ps_baseclosure, sts, NNs, Nt, timestep_multiple)

    ps_noNN = deepcopy(ps) .= 0
    sols_noNN = solve_NDE_withmomentum(ps_noNN, params, x₀, ps_baseclosure, sts, NNs, Nt, timestep_multiple)

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
    
    ∂T∂z_hats = hcat([params.scaling.∂T∂z.(params.Dᶠ * T) for T in eachcol(Ts)]...)
    ∂S∂z_hats = hcat([params.scaling.∂S∂z.(params.Dᶠ * S) for S in eachcol(Ss)]...)
    ∂ρ∂z_hats = hcat([params.scaling.∂ρ∂z.(params.Dᶠ * ρ) for ρ in eachcol(ρs)]...)

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
        wT_residuals[:, i], wS_residuals[:, i] = predict_residual_flux_dimensional(∂T∂z_hats[:, i], ∂S∂z_hats[:, i], ∂ρ∂z_hats[:, i], Ris[:, i], T_tops[i], S_tops[i], ps, params, sts, NNs)
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

function animate_data(train_data, sols, fluxes, diffusivities, sols_noNN, fluxes_noNN, diffusivities_noNN, index, FILE_DIR, Nframes; suffix=1)
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
    axRi = CairoMakie.Axis(fig[1, 6], title="Ri", xlabel="arctan(Ri)", ylabel="z (m)")
    axdiffusivity = CairoMakie.Axis(fig[2, 5], title="Diffusivity", xlabel="Diffusivity (m² s⁻¹)", ylabel="z (m)")

    n = Observable(1)
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

    CairoMakie.record(fig, "$(FILE_DIR)/training_$(index)_$(suffix).mp4", 1:Nframes, framerate=10) do nn
        n[] = nn
    end
end

function plot_loss(losses, FILE_DIR; suffix=1)
    colors = distinguishable_colors(10, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    
    fig = Figure(size=(1000, 600))
    axtotalloss = CairoMakie.Axis(fig[1, 1], title="Total Loss", xlabel="Iterations", ylabel="Loss", yscale=log10)
    # axindividualloss = CairoMakie.Axis(fig[1, 2], title="Individual Loss", xlabel="Iterations", ylabel="Loss", yscale=log10)

    lines!(axtotalloss, losses.total, label="Total Loss", color=colors[1])
    lines!(axtotalloss, losses.total_validation, label="Total Validation Loss", color=colors[2])

    # lines!(axindividualloss, losses.T, label="T", color=colors[3])
    # lines!(axindividualloss, losses.S, label="S", color=colors[4])
    # lines!(axindividualloss, losses.ρ, label="ρ", color=colors[5])
    # lines!(axindividualloss, losses.∂T∂z, label="∂T∂z", color=colors[8])
    # lines!(axindividualloss, losses.∂S∂z, label="∂S∂z", color=colors[9])
    # lines!(axindividualloss, losses.∂ρ∂z, label="∂ρ∂z", color=colors[10])

    axislegend(axtotalloss, position=:lb)
    # axislegend(axindividualloss, position=:rt)
    save("$(FILE_DIR)/losses_$(suffix).png", fig, px_per_unit=8)
end

function train_NDE_multipleics(ps, params, ps_baseclosure, sts, NNs, truths, x₀s, truths_validation, x₀s_validation, params_validation, validation_data, train_data_plot, timeframes, S_scaling, scaling_params; sim_index=[1], epoch=1, maxiter=2, rule=Optimisers.Adam())
    opt_state = Optimisers.setup(rule, ps)
    opt_statemin = deepcopy(opt_state)
    opt_statemin_validation = deepcopy(opt_state)

    l_min = Inf
    l_min_validation = Inf

    iter_min = 1
    iter_min_validation = 1

    ps_min = deepcopy(ps)
    ps_min_validation = deepcopy(ps)
    
    dps = deepcopy(ps) .= 0
    wall_clock = [time_ns()]

    losses = zeros(maxiter)
    losses_validation = zeros(maxiter)

    mean_loss = 0
    mean_loss_validation = 0

    ind_losses = [individual_loss(ps, truth, param, x₀, ps_baseclosure, sts, NNs, length(timeframes)) for (truth, x₀, param) in zip(truths[sim_index], x₀s[sim_index], params[sim_index])]
    ind_losses_validation = [individual_loss(ps, truth, param, x₀, ps_baseclosure, sts, NNs, length(timeframes)) for (truth, x₀, param) in zip(truths_validation, x₀s_validation, params_validation)]

    loss_prefactors = compute_loss_prefactor_density_contribution.(ind_losses, compute_density_contribution.(train_data.data), S_scaling)
    loss_prefactors_validation = compute_loss_prefactor_density_contribution.(ind_losses_validation, compute_density_contribution.(validation_data.data), S_scaling)

    jldopen("$(FILE_DIR)/model_weights_round$(epoch)_end$(timeframes[end]).jld2", "w") do file
        file["0"] = ps
        file["scaling"] = scaling_params
        file["model"] = NNs
        file["sts"] = sts
    end

    jldopen("$(FILE_DIR)/model_loss_training_round$(epoch)_end$(timeframes[end]).jld2", "w")
    jldopen("$(FILE_DIR)/model_loss_validation_round$(epoch)_end$(timeframes[end]).jld2", "w")

    for iter in 1:maxiter
        _, l = autodiff(Enzyme.ReverseWithPrimal, 
                        loss_multipleics, 
                        Active, 
                        DuplicatedNoNeed(ps, dps), 
                        DuplicatedNoNeed(truths[sim_index], deepcopy(truths[sim_index])), 
                        DuplicatedNoNeed(params[sim_index], deepcopy(params[sim_index])), 
                        DuplicatedNoNeed(x₀s[sim_index], deepcopy(x₀s[sim_index])), 
                        DuplicatedNoNeed(ps_baseclosure, deepcopy(ps_baseclosure)), 
                        Const(sts), 
                        Const(NNs), 
                        DuplicatedNoNeed(loss_prefactors[sim_index], deepcopy(loss_prefactors[sim_index])),
                        Const(length(timeframes)))

        l_validation = loss_multipleics(ps, truths_validation[sim_index], params_validation[sim_index], x₀s_validation[sim_index], ps_baseclosure, sts, NNs, loss_prefactors_validation[sim_index], length(timeframes))

        # if iter <= 40
            # Optimisers.adjust!(opt_state, eta=rule.eta * iter / 40)
        # end
        
        opt_state, ps = Optimisers.update!(opt_state, ps, dps)

        losses[iter] = l
        losses_validation[iter] = l_validation

        if iter == 1
            l_min = l
            l_min_validation = l_validation
        end
        
        @printf("%s, Δt %s, round %d, iter %d/%d, (l_train, l_validate) (%6.10e, %6.10e), (lmin_train, l_min_validate) (%6.5e, %6.5e) max NN weight %6.5e, gradient norm %6.5e\n",
                Dates.now(), prettytime(1e-9 * (time_ns() - wall_clock[1])), epoch, iter, maxiter, l, l_validation, l_min, l_min_validation,
                maximum(abs, ps), maximum(abs, dps))
        
        dps .= 0
        mean_loss = l
        mean_loss_validation = l_validation

        if mean_loss < l_min
            l_min = mean_loss
            iter_min = iter
            opt_statemin = deepcopy(opt_state)
            ps_min .= ps
        end

        if mean_loss_validation < l_min_validation
            l_min_validation = mean_loss_validation
            iter_min_validation = iter
            opt_statemin_validation = deepcopy(opt_state)
            ps_min_validation .= ps
        end

        jldopen("$(FILE_DIR)/model_weights_round$(epoch)_end$(timeframes[end]).jld2", "a") do file
            file["$(iter)"] = ps
        end

        jldopen("$(FILE_DIR)/model_loss_training_round$(epoch)_end$(timeframes[end]).jld2", "a+") do file
            file["$(iter)"] = l
        end

        jldopen("$(FILE_DIR)/model_loss_validation_round$(epoch)_end$(timeframes[end]).jld2", "a+") do file
            file["$(iter)"] = l_validation
        end

        if iter % 1000 == 0
            @info "Saving intermediate results"
            jldsave("$(FILE_DIR)/intermediate_training_results_round$(epoch)_epoch$(iter)_end$(timeframes[end]).jld2"; 
                     u_train = ps_min, state_train = opt_statemin, loss_train = l_min, iter_min = iter_min, scaling=scaling_params, model=NNs, sts=sts,
                     u_validation = ps_min_validation, state_validation = opt_statemin_validation, loss_validation = l_min_validation, iter_min_validation = iter_min_validation)
        end
        wall_clock = [time_ns()]
    end
    return ps_min, ps_min_validation, (; total=losses, total_validation=losses_validation), opt_statemin, opt_statemin_validation, iter_min, iter_min_validation
end

optimizers = [Optimisers.Adam(3e-4), Optimisers.Adam(1e-5), Optimisers.Adam(1e-5), Optimisers.Adam(3e-6), Optimisers.Adam(3e-6), Optimisers.Adam(3e-6)]
maxiters = [2000, 2000, 2000, 2000, 2000, 2000]
end_epochs = cumsum(maxiters)
training_timeframes = [timeframes[1][1:5], timeframes[1][1:10], timeframes[1][1:15], timeframes[1][1:20], timeframes[1][1:25], timeframes[1][1:27]]

# optimizers = [Optimisers.Adam(3e-4)]
# maxiters = [2000]
# end_epochs = cumsum(maxiters)
# training_timeframes = [timeframes[1][1:27]]

# sim_indices = 1:length(LES_FILE_DIRS)

# optimizers = [Optimisers.Adam(3e-3)]
# maxiters = [10]
# end_epochs = cumsum(maxiters)

# sim_indices = 1:length(LES_FILE_DIRS)

# training_timeframes = [timeframes[1][1:5]]

plot_timeframes = full_timeframes
sols = nothing
for (i, (epoch, optimizer, maxiter, training_timeframe, plot_timeframe)) in enumerate(zip(end_epochs, optimizers, maxiters, training_timeframes, plot_timeframes))
    global ps = ps
    global sols = sols
    ps, ps_validation, losses, opt_state, opt_state_validation, iter_min, iter_min_validation = train_NDE_multipleics(ps, params, ps_baseclosure, sts, NNs, 
                                                                                                                      truths, x₀s, truths_validation, x₀s_validation, params_validation, validation_data,
                                                                                                                      train_data_plot, training_timeframe, S_scaling, scaling_params; 
                                                                                                                      sim_index=sim_indices, epoch=i, maxiter=maxiter, rule=optimizer)
    
    jldsave("$(FILE_DIR)/training_results_epoch$(epoch)_end$(training_timeframe[end]).jld2";
            u_train = ps, state_train = opt_state, iter_min = iter_min,
            u_validation = ps_validation, state_validation = opt_state_validation, iter_min_validation = iter_min_validation,
            losses = losses, scaling=scaling_params, model=NNs, sts=sts)

    sols = [diagnose_fields(ps, param, x₀, ps_baseclosure, sts, NNs, data, length(timeframe)) for (data, x₀, param, timeframe) in zip(train_data_plot.data, x₀s_plot, params_plot, plot_timeframes)]

    for (i, sol) in enumerate(sols)
        animate_data(train_data_plot.data[i], sol.sols_dimensional, sol.fluxes, sol.diffusivities, sol.sols_dimensional_noNN, sol.fluxes_noNN, sol.diffusivities_noNN, i, FILE_DIR, length(plot_timeframes[i]); suffix="epoch$(epoch)_end$(training_timeframe[end])")
    end

    plot_loss(losses, FILE_DIR; suffix="epoch$(epoch)_end$(training_timeframe[end])")
end