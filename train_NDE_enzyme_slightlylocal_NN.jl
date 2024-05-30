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
    #   "--loss_type"
    #     help = "Loss function used"
    #     arg_type = String
    #     default = "mse"
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

LES_FILE_DIRS = ["./LES2/$(file)/instantaneous_timeseries.jld2" for file in LES_suite["train22"]]

FILE_DIR = "./training_output/NDE_$(length(LES_FILE_DIRS))sim_$(args["hidden_layer"])layer_$(args["hidden_layer_size"])_$(args["activation"])"
mkpath(FILE_DIR)
@info FILE_DIR

field_datasets = [FieldDataset(FILE_DIR, backend=OnDisk()) for FILE_DIR in LES_FILE_DIRS]

timeframes = [25:10:length(data["ubar"].times) for data in field_datasets]
full_timeframes = [25:length(data["ubar"].times) for data in field_datasets]
coarse_size = 32
train_data = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, timeframes, coarse_size)
scaling = train_data.scaling

truths = [(; u = data.profile.u.scaled, 
             v = data.profile.v.scaled, 
             T = data.profile.T.scaled, 
             S = data.profile.S.scaled, 
             ρ = data.profile.ρ.scaled, 
             ∂u∂z = data.profile.∂u∂z.scaled,
             ∂v∂z = data.profile.∂v∂z.scaled,
             ∂T∂z = data.profile.∂T∂z.scaled, 
             ∂S∂z = data.profile.∂S∂z.scaled, 
             ∂ρ∂z = data.profile.∂ρ∂z.scaled) for data in train_data.data]

train_data_plot = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, full_timeframes, coarse_size)

params = ODEParams(train_data)
params_plot = ODEParams(train_data_plot, scaling)

rng = Random.default_rng(123)

#%%
NN_layers = vcat(Dense(20, hidden_layer_size, activation), [Dense(hidden_layer_size, hidden_layer_size, activation) for _ in 1:N_hidden_layer-1]..., Dense(hidden_layer_size, 1))

uw_NN = Chain(NN_layers...)
vw_NN = Chain(NN_layers...)
wT_NN = Chain(NN_layers...)
wS_NN = Chain(NN_layers...)

ps_uw, st_uw = Lux.setup(rng, uw_NN)
ps_vw, st_vw = Lux.setup(rng, vw_NN)
ps_wT, st_wT = Lux.setup(rng, wT_NN)
ps_wS, st_wS = Lux.setup(rng, wS_NN)

ps_uw = ps_uw |> ComponentArray .|> Float64
ps_vw = ps_vw |> ComponentArray .|> Float64
ps_wT = ps_wT |> ComponentArray .|> Float64
ps_wS = ps_wS |> ComponentArray .|> Float64

ps_uw .= glorot_uniform(rng, Float64, length(ps_uw))
ps_vw .= glorot_uniform(rng, Float64, length(ps_vw))
ps_wT .= glorot_uniform(rng, Float64, length(ps_wT))
ps_wS .= glorot_uniform(rng, Float64, length(ps_wS))

# ps_uw .*= 0
# ps_vw .*= 0
# ps_wT .*= 0
# ps_wS .*= 0

x₀s = [(; u=data.profile.u.scaled[:, 1], v=data.profile.v.scaled[:, 1], T=data.profile.T.scaled[:, 1], S=data.profile.S.scaled[:, 1]) for data in train_data.data]

# PS_DIR = "./training_output/NDE_20sim_2layer_128_relu_localbaseclosure/training_results_epoch14000_end165.jld2"
# ps = jldopen(PS_DIR, "r")["u"]

# pw_uw = ps.uw
# pw_vw = ps.vw
# pw_wT = ps.wT
# pw_wS = ps.wS

ps = ComponentArray(; uw=ps_uw, vw=ps_vw, wT=ps_wT, wS=ps_wS)
NNs = (uw=uw_NN, vw=vw_NN, wT=wT_NN, wS=wS_NN)
sts = (uw=st_uw, vw=st_vw, wT=st_wT, wS=st_wS)

function predict_residual_flux(∂u∂z_hat, ∂v∂z_hat, ∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat, p, params, sts, NNs)
    common_variables = vcat(params.uw.scaled.top, params.vw.scaled.top, params.wT.scaled.top, params.wS.scaled.top, params.f_scaled)

    uw = zeros(params.coarse_size+1)
    vw = zeros(params.coarse_size+1)
    wT = zeros(params.coarse_size+1)
    wS = zeros(params.coarse_size+1)

    ∂u∂z_hat[1] = ∂u∂z_hat[2]
    ∂v∂z_hat[1] = ∂v∂z_hat[2]
    ∂T∂z_hat[1] = ∂T∂z_hat[2]
    ∂S∂z_hat[1] = ∂S∂z_hat[2]
    ∂ρ∂z_hat[1] = ∂ρ∂z_hat[2]

    ∂u∂z_hat[end] = ∂u∂z_hat[end-1]
    ∂v∂z_hat[end] = ∂v∂z_hat[end-1]
    ∂T∂z_hat[end] = ∂T∂z_hat[end-1]
    ∂S∂z_hat[end] = ∂S∂z_hat[end-1]
    ∂ρ∂z_hat[end] = ∂ρ∂z_hat[end-1]

    for i in 2:params.coarse_size
        x  = vcat(∂u∂z_hat[i-1:i+1], ∂v∂z_hat[i-1:i+1], ∂T∂z_hat[i-1:i+1], ∂S∂z_hat[i-1:i+1], ∂ρ∂z_hat[i-1:i+1], common_variables)
        uw[i] = first(NNs.uw(x, p.uw, sts.uw))[1]
        vw[i] = first(NNs.vw(x, p.vw, sts.vw))[1]
        wT[i] = first(NNs.wT(x, p.wT, sts.wT))[1]
        wS[i] = first(NNs.wS(x, p.wS, sts.wS))[1]
    end

    uw[1] = uw[2]
    vw[1] = vw[2]
    wT[1] = wT[2]
    wS[1] = wS[2]

    return uw, vw, wT, wS
end

function solve_NDE(ps, params, x₀, sts, NNs, Nt, timestep_multiple=10)
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

    uw_residual = zeros(coarse_size+1)
    vw_residual = zeros(coarse_size+1)
    wT_residual = zeros(coarse_size+1)
    wS_residual = zeros(coarse_size+1)

    uw_boundary = zeros(coarse_size+1)
    vw_boundary = zeros(coarse_size+1)
    wT_boundary = zeros(coarse_size+1)
    wS_boundary = zeros(coarse_size+1)

    sol_u = zeros(coarse_size, Nt_solve)
    sol_v = zeros(coarse_size, Nt_solve)
    sol_T = zeros(coarse_size, Nt_solve)
    sol_S = zeros(coarse_size, Nt_solve)
    sol_ρ = zeros(coarse_size, Nt_solve)

    sol_u[:, 1] .= u_hat
    sol_v[:, 1] .= v_hat
    sol_T[:, 1] .= T_hat
    sol_S[:, 1] .= S_hat

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

        uw_residual, vw_residual, wT_residual, wS_residual = predict_residual_flux(∂u∂z_hat, ∂v∂z_hat, ∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat, ps, params, sts, NNs)
        predict_boundary_flux!(uw_boundary, vw_boundary, wT_boundary, wS_boundary, params)

        u_RHS .= - τ / H * scaling.uw.σ / scaling.u.σ .* (Dᶜ_hat * (uw_boundary .+ uw_residual)) .+ f * τ ./ scaling.u.σ .* v
        v_RHS .= - τ / H * scaling.vw.σ / scaling.v.σ .* (Dᶜ_hat * (vw_boundary .+ vw_residual)) .- f * τ ./ scaling.v.σ .* u
        T_RHS .= - τ / H * scaling.wT.σ / scaling.T.σ .* (Dᶜ_hat * (wT_boundary .+ wT_residual))
        S_RHS .= - τ / H * scaling.wS.σ / scaling.S.σ .* (Dᶜ_hat * (wS_boundary .+ wS_residual))

        u_hat .= u_hat .+ Δt .* u_RHS
        v_hat .= v_hat .+ Δt .* v_RHS
        T_hat .= T_hat .+ Δt .* T_RHS
        S_hat .= S_hat .+ Δt .* S_RHS

        sol_u[:, i] .= u_hat
        sol_v[:, i] .= v_hat
        sol_T[:, i] .= T_hat
        sol_S[:, i] .= S_hat
    end

    sol_ρ[:, end] .= scaling.ρ.(TEOS10.ρ.(inv(scaling.T).(T_hat), inv(scaling.S).(S_hat), 0, Ref(eos)))

    return (; u=sol_u[:, 1:timestep_multiple:end], v=sol_v[:, 1:timestep_multiple:end], T=sol_T[:, 1:timestep_multiple:end], S=sol_S[:, 1:timestep_multiple:end], ρ=sol_ρ[:, 1:timestep_multiple:end])
end

sol_u, sol_v, sol_T, sol_S, sol_ρ = solve_NDE(ps, params[1], x₀s[1], sts, NNs, length(25:10:45))
# #%%
sol_index = 1
truth = truths[sol_index]
sol_u, sol_v, sol_T, sol_S, sol_ρ = solve_NDE(ps, params[sol_index], x₀s[sol_index], sts, NNs, length(25:10:45))

fig = Figure(size=(1800, 600))
axu = CairoMakie.Axis(fig[1, 1], xlabel="u", ylabel="z")
axv = CairoMakie.Axis(fig[1, 2], xlabel="v", ylabel="z")
axT = CairoMakie.Axis(fig[1, 3], xlabel="T", ylabel="z")
axS = CairoMakie.Axis(fig[1, 4], xlabel="S", ylabel="z")
axρ = CairoMakie.Axis(fig[1, 5], xlabel="ρ", ylabel="z")

lines!(axu, sol_u[:, 1], params[1].zC, label="initial")
lines!(axu, sol_u[:, end], params[1].zC, label="final")
lines!(axu, truth.u[:, length(25:10:45)], train_data.data[1].metadata["zC"], label="truth")

lines!(axv, sol_v[:, 1], params[1].zC, label="initial")
lines!(axv, sol_v[:, end], params[1].zC, label="final")
lines!(axv, truth.v[:, length(25:10:45)], train_data.data[1].metadata["zC"], label="truth")

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
display(fig)
save("$(FILE_DIR)/initial_final_truth.png", fig)
#%%
function individual_loss(ps, truth, params, x₀, sts, NNs, Nt, tstart=1, timestep_multiple=10)
    Dᶠ = params.Dᶠ
    scaling = params.scaling
    sol_u, sol_v, sol_T, sol_S, sol_ρ = solve_NDE(ps, params, x₀, sts, NNs, Nt, timestep_multiple)

    u_loss = mean((sol_u .- truth.u[:, tstart:tstart+Nt-1]).^2)
    v_loss = mean((sol_v .- truth.v[:, tstart:tstart+Nt-1]).^2)
    T_loss = mean((sol_T .- truth.T[:, tstart:tstart+Nt-1]).^2)
    S_loss = mean((sol_S .- truth.S[:, tstart:tstart+Nt-1]).^2)
    ρ_loss = mean((sol_ρ .- truth.ρ[:, tstart:tstart+Nt-1]).^2)

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

    ∂u∂z_loss = mean((∂u∂z[1:end-2,:] .- truth.∂u∂z[1:end-2, tstart:tstart+Nt-1]).^2)
    ∂v∂z_loss = mean((∂v∂z[1:end-2,:] .- truth.∂v∂z[1:end-2, tstart:tstart+Nt-1]).^2)
    ∂T∂z_loss = mean((∂T∂z[1:end-2,:] .- truth.∂T∂z[1:end-2, tstart:tstart+Nt-1]).^2)
    ∂S∂z_loss = mean((∂S∂z[1:end-2,:] .- truth.∂S∂z[1:end-2, tstart:tstart+Nt-1]).^2)
    ∂ρ∂z_loss = mean((∂ρ∂z[1:end-2,:] .- truth.∂ρ∂z[1:end-2, tstart:tstart+Nt-1]).^2)

    return (; u=u_loss, v=v_loss, T=T_loss, S=S_loss, ρ=ρ_loss, ∂u∂z=∂u∂z_loss, ∂v∂z=∂v∂z_loss, ∂T∂z=∂T∂z_loss, ∂S∂z=∂S∂z_loss, ∂ρ∂z=∂ρ∂z_loss)
end

function loss(ps, truth, params, x₀, sts, NNs, Nt, tstart=1, timestep_multiple=10, losses_prefactor=(; u=1, v=1, T=1, S=1, ρ=1, ∂u∂z=1, ∂v∂z=1, ∂T∂z=1, ∂S∂z=1, ∂ρ∂z=1))
    losses = individual_loss(ps, truth, params, x₀, sts, NNs, Nt, tstart, timestep_multiple)
    return sum(values(losses) .* values(losses_prefactor))
end

# loss(ps, truths[1], params[1], x₀s[1], sts, NNs, length(25:10:45))

# dps = deepcopy(ps) .= 0
# autodiff(Enzyme.ReverseWithPrimal, 
#          loss, 
#          Active, 
#          DuplicatedNoNeed(ps, dps), 
#          Const(truths[1]), 
#          Const(params[1]), 
#          DuplicatedNoNeed(x₀s[1], deepcopy(x₀s[1])), 
#          Const(sts), 
#          Const(NNs),
#          Const(length(25:10:45)))

function compute_loss_prefactor_density_contribution(individual_loss, contribution, S_scaling=1.0)
    u_loss, v_loss, T_loss, S_loss, ρ_loss, ∂u∂z_loss, ∂v∂z_loss, ∂T∂z_loss, ∂S∂z_loss, ∂ρ∂z_loss = values(individual_loss)
    
    total_contribution = contribution.T + contribution.S
    T_prefactor = total_contribution / contribution.T
    S_prefactor = total_contribution / contribution.S

    TS_loss = T_prefactor * T_loss + S_prefactor * S_loss

    ρ_prefactor = TS_loss / ρ_loss * 0.1 / 0.9

    if u_loss > eps(eltype(u_loss))
        u_prefactor = TS_loss / u_loss * 0.2 / 0.4
    else
        u_prefactor = TS_loss * 0.2 / 0.4
    end

    if v_loss > eps(eltype(v_loss))
        v_prefactor = TS_loss / v_loss * 0.2 / 0.4
    else
        v_prefactor = TS_loss * 0.2 / 0.4
    end

    ∂T∂z_prefactor = T_prefactor
    ∂S∂z_prefactor = S_prefactor

    ∂TS∂z_loss = ∂T∂z_loss + ∂S∂z_loss

    ∂ρ∂z_prefactor = ∂TS∂z_loss / ∂ρ∂z_loss * 0.1 / 0.9

    if ∂u∂z_loss > eps(eltype(∂u∂z_loss))
        ∂u∂z_prefactor = ∂TS∂z_loss / ∂u∂z_loss * 0.2 / 0.4
    else
        ∂u∂z_prefactor = ∂TS∂z_loss * 0.2 / 0.4
    end

    if ∂v∂z_loss > eps(eltype(∂v∂z_loss))
        ∂v∂z_prefactor = ∂TS∂z_loss / ∂v∂z_loss * 0.2 / 0.4
    else
        ∂v∂z_prefactor = ∂TS∂z_loss * 0.2 / 0.4
    end

    profile_loss = u_prefactor * u_loss + v_prefactor * v_loss + T_prefactor * T_loss + S_prefactor * S_loss + ρ_prefactor * ρ_loss
    gradient_loss = ∂u∂z_prefactor * ∂u∂z_loss + ∂v∂z_prefactor * ∂v∂z_loss + ∂T∂z_prefactor * ∂T∂z_loss + ∂S∂z_prefactor * ∂S∂z_loss + ∂ρ∂z_prefactor * ∂ρ∂z_loss

    gradient_prefactor = profile_loss / gradient_loss

    ∂ρ∂z_prefactor *= gradient_prefactor
    ∂T∂z_prefactor *= gradient_prefactor
    ∂S∂z_prefactor *= gradient_prefactor
    ∂u∂z_prefactor *= gradient_prefactor
    ∂v∂z_prefactor *= gradient_prefactor

    S_prefactor *= S_scaling
    ∂S∂z_prefactor *= S_scaling

    return (u=u_prefactor, v=v_prefactor, T=T_prefactor, S=S_prefactor, ρ=ρ_prefactor, ∂u∂z=∂u∂z_prefactor, ∂v∂z=∂v∂z_prefactor, ∂T∂z=∂T∂z_prefactor, ∂S∂z=∂S∂z_prefactor, ∂ρ∂z=∂ρ∂z_prefactor)
end

ind_losses = [individual_loss(ps, truth, param, x₀, sts, NNs, length(timeframe)) for (truth, x₀, param, timeframe) in zip(truths, x₀s, params, timeframes)]

loss_prefactors = compute_loss_prefactor_density_contribution.(ind_losses, compute_density_contribution.(train_data.data), S_scaling)

function loss_multipleics(ps, truths, params, x₀s, sts, NNs, losses_prefactor, Nt, tstart=1, timestep_multiple=10)
    losses = [loss(ps, truth, param, x₀, sts, NNs, Nt, tstart, timestep_multiple, loss_prefactor) for (truth, x₀, param, loss_prefactor) in zip(truths, x₀s, params, losses_prefactor)]
    return mean(losses)
end

# loss_multipleics(ps, [truths[1]], [params[1]], [x₀s[1]], sts, NNs, [loss_prefactors[1]], length(25:10:45))

# dps = deepcopy(ps) .= 0
# autodiff(Enzyme.ReverseWithPrimal, 
#          loss_multipleics, 
#          Active, 
#          DuplicatedNoNeed(ps, dps), 
#          DuplicatedNoNeed([truths[1]], deepcopy([truths[1]])), 
#          DuplicatedNoNeed([params[1]], deepcopy([params[1]])), 
#          DuplicatedNoNeed([x₀s[1]], deepcopy([x₀s[1]])), 
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
#          Const(sts), 
#          Const(NNs), 
#          DuplicatedNoNeed(loss_prefactors[1:2], deepcopy(loss_prefactors[1:2])),
#          Const(length(25:10:45)))

function predict_residual_flux_dimensional(∂u∂z_hat, ∂v∂z_hat, ∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat, p, params, sts, NNs)
    uw_hat, vw_hat, wT_hat, wS_hat = predict_residual_flux(∂u∂z_hat, ∂v∂z_hat, ∂T∂z_hat, ∂S∂z_hat, ∂ρ∂z_hat, p, params, sts, NNs)
    
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

function predict_boundary_flux_dimensional(params)
    _uw_boundary, _vw_boundary, _wT_boundary, _wS_boundary = predict_boundary_flux(params)

    uw = inv(params.scaling.uw).(_uw_boundary)
    vw = inv(params.scaling.vw).(_vw_boundary)
    wT = inv(params.scaling.wT).(_wT_boundary)
    wS = inv(params.scaling.wS).(_wS_boundary)

    return uw, vw, wT, wS
end

function diagnose_fields(ps, params, x₀, sts, NNs, train_data_plot, Nt, timestep_multiple=2)
    sols = solve_NDE(ps, params, x₀, sts, NNs, Nt, timestep_multiple)

    coarse_size = params.coarse_size
    Dᶠ = params.Dᶠ
    scaling = params.scaling

    us = inv(scaling.u).(sols.u)
    vs = inv(scaling.v).(sols.v)
    Ts = inv(scaling.T).(sols.T)
    Ss = inv(scaling.S).(sols.S)
    ρs = inv(scaling.ρ).(sols.ρ)

    ∂u∂z_hats = hcat([params.scaling.∂u∂z.(params.Dᶠ * u) for u in eachcol(us)]...)
    ∂v∂z_hats = hcat([params.scaling.∂v∂z.(params.Dᶠ * v) for v in eachcol(vs)]...)
    ∂T∂z_hats = hcat([params.scaling.∂T∂z.(params.Dᶠ * T) for T in eachcol(Ts)]...)
    ∂S∂z_hats = hcat([params.scaling.∂S∂z.(params.Dᶠ * S) for S in eachcol(Ss)]...)
    ∂ρ∂z_hats = hcat([params.scaling.∂ρ∂z.(params.Dᶠ * ρ) for ρ in eachcol(ρs)]...)

    eos = TEOS10EquationOfState()
    Ris_truth = hcat([calculate_Ri(u, v, ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf)) 
                      for (u, v, ρ) in zip(eachcol(train_data_plot.profile.u.unscaled), eachcol(train_data_plot.profile.v.unscaled), eachcol(train_data_plot.profile.ρ.unscaled))]...)
    Ris = hcat([calculate_Ri(u, v, ρ, Dᶠ, params.g, eos.reference_density, clamp_lims=(-Inf, Inf)) 
                      for (u, v, ρ) in zip(eachcol(us), eachcol(vs), eachcol(ρs))]...)
    
    uw_residuals = zeros(coarse_size+1, size(Ts, 2))
    vw_residuals = zeros(coarse_size+1, size(Ts, 2))
    wT_residuals = zeros(coarse_size+1, size(Ts, 2))
    wS_residuals = zeros(coarse_size+1, size(Ts, 2))

    uw_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))
    vw_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))
    wT_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))
    wS_diffusive_boundarys = zeros(coarse_size+1, size(Ts, 2))

    for i in 1:size(wT_residuals, 2)
        uw_residuals[:, i], vw_residuals[:, i], wT_residuals[:, i], wS_residuals[:, i] = predict_residual_flux_dimensional(∂u∂z_hats[:, i], ∂v∂z_hats[:, i], ∂T∂z_hats[:, i], ∂S∂z_hats[:, i], ∂ρ∂z_hats[:, i], ps, params, sts, NNs)
        uw_diffusive_boundarys[:, i], vw_diffusive_boundarys[:, i], wT_diffusive_boundarys[:, i], wS_diffusive_boundarys[:, i] = predict_boundary_flux_dimensional(params)        
    end

    uw_totals = uw_residuals .+ uw_diffusive_boundarys
    vw_totals = vw_residuals .+ vw_diffusive_boundarys
    wT_totals = wT_residuals .+ wT_diffusive_boundarys
    wS_totals = wS_residuals .+ wS_diffusive_boundarys

    fluxes = (; uw = (; diffusive_boundary=uw_diffusive_boundarys, residual=uw_residuals, total=uw_totals),
                vw = (; diffusive_boundary=vw_diffusive_boundarys, residual=vw_residuals, total=vw_totals),
                wT = (; diffusive_boundary=wT_diffusive_boundarys, residual=wT_residuals, total=wT_totals), 
                wS = (; diffusive_boundary=wS_diffusive_boundarys, residual=wS_residuals, total=wS_totals))

    diffusivities = (; Ri=Ris, Ri_truth=Ris_truth)

    sols_dimensional = (; u=us, v=vs, T=Ts, S=Ss, ρ=ρs)
    return (; sols_dimensional, fluxes, diffusivities)
end

function animate_data(train_data, sols, fluxes, diffusivities, index, FILE_DIR, Nframes; suffix=1)
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
    axdiffusivity = CairoMakie.Axis(fig[2, 5], title="Diffusivity", xlabel="Diffusivity (m² s⁻¹)", ylabel="z (m)")

    n = Observable(1)
    zC = train_data.metadata["zC"]
    zF = train_data.metadata["zF"]

    Ri_max = 3

    u_NDE = sols.u
    v_NDE = sols.v
    T_NDE = sols.T
    S_NDE = sols.S
    ρ_NDE = sols.ρ

    uw_residual = fluxes.uw.residual
    vw_residual = fluxes.vw.residual
    wT_residual = fluxes.wT.residual
    wS_residual = fluxes.wS.residual

    uw_total = fluxes.uw.total
    vw_total = fluxes.vw.total
    wT_total = fluxes.wT.total
    wS_total = fluxes.wS.total

    ulim = (find_min(u_NDE, train_data.profile.u.unscaled), find_max(u_NDE, train_data.profile.u.unscaled))
    vlim = (find_min(v_NDE, train_data.profile.v.unscaled), find_max(v_NDE, train_data.profile.v.unscaled))
    Tlim = (find_min(T_NDE, train_data.profile.T.unscaled), find_max(T_NDE, train_data.profile.T.unscaled))
    Slim = (find_min(S_NDE, train_data.profile.S.unscaled), find_max(S_NDE, train_data.profile.S.unscaled))
    ρlim = (find_min(ρ_NDE, train_data.profile.ρ.unscaled), find_max(ρ_NDE, train_data.profile.ρ.unscaled))

    uwlim = (find_min(uw_residual, train_data.flux.uw.column.unscaled), find_max(uw_residual, train_data.flux.uw.column.unscaled))
    vwlim = (find_min(vw_residual, train_data.flux.vw.column.unscaled), find_max(vw_residual, train_data.flux.vw.column.unscaled))
    wTlim = (find_min(wT_residual, train_data.flux.wT.column.unscaled), find_max(wT_residual, train_data.flux.wT.column.unscaled))
    wSlim = (find_min(wS_residual, train_data.flux.wS.column.unscaled), find_max(wS_residual, train_data.flux.wS.column.unscaled))

    Rilim = (find_min(diffusivities.Ri, diffusivities.Ri_truth), Ri_max)

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

    uw_totalₙ = @lift uw_total[:, $n]
    vw_totalₙ = @lift vw_total[:, $n]
    wT_totalₙ = @lift wT_total[:, $n]
    wS_totalₙ = @lift wS_total[:, $n]

    Ri_truthₙ = @lift clamp.(diffusivities.Ri_truth[:, $n], Ref(-Inf..Ri_max))
    Riₙ = @lift clamp.(diffusivities.Ri[:, $n], Ref(-Inf..Ri_max))

    Qᵁ = train_data.metadata["momentum_flux"]
    Qᵀ = train_data.metadata["temperature_flux"]
    Qˢ = train_data.metadata["salinity_flux"]
    f = train_data.metadata["coriolis_parameter"]
    times = train_data.times
    Nt = length(times)

    time_str = @lift "Qᵁ = $(Qᵁ) m² s⁻², Qᵀ = $(Qᵀ) m s⁻¹ °C, Qˢ = $(Qˢ) m s⁻¹ g kg⁻¹, f = $(f) s⁻¹, Time = $(round(times[$n]/24/60^2, digits=3)) days"

    lines!(axu, u_truthₙ, zC, label="Truth", linewidth=4, alpha=0.5)
    lines!(axu, u_NDEₙ, zC, label="NDE", color=:black)

    lines!(axv, v_truthₙ, zC, label="Truth", linewidth=4, alpha=0.5)
    lines!(axv, v_NDEₙ, zC, label="NDE", color=:black)

    lines!(axT, T_truthₙ, zC, label="Truth", linewidth=4, alpha=0.5)
    lines!(axT, T_NDEₙ, zC, label="NDE", color=:black)

    lines!(axS, S_truthₙ, zC, label="Truth", linewidth=4, alpha=0.5)
    lines!(axS, S_NDEₙ, zC, label="NDE", color=:black)

    lines!(axρ, ρ_truthₙ, zC, label="Truth", linewidth=4, alpha=0.5)
    lines!(axρ, ρ_NDEₙ, zC, label="NDE", color=:black)

    lines!(axuw, uw_truthₙ, zF, label="Truth", linewidth=4, alpha=0.5)
    lines!(axuw, uw_totalₙ, zF, label="NDE", color=:black)

    lines!(axvw, vw_truthₙ, zF, label="Truth", linewidth=4, alpha=0.5)
    lines!(axvw, vw_totalₙ, zF, label="NDE", color=:black)

    lines!(axwT, wT_truthₙ, zF, label="Truth", linewidth=4, alpha=0.5)
    lines!(axwT, wT_totalₙ, zF, label="NDE", color=:black)

    lines!(axwS, wS_truthₙ, zF, label="Truth", linewidth=4, alpha=0.5)
    lines!(axwS, wS_totalₙ, zF, label="NDE", color=:black)

    lines!(axRi, Ri_truthₙ, zF, label="Truth", linewidth=4, alpha=0.5)
    lines!(axRi, Riₙ, zF, label="NDE", color=:black)

    axislegend(axu, position=:lb)
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
    xlims!(axRi, Rilim)

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

function train_NDE_multipleics(ps, params, sts, NNs, truths, x₀s, train_data_plot, timeframes, S_scaling; sim_index=[1], epoch=1, maxiter=2, rule=Optimisers.Adam())
    opt_state = Optimisers.setup(rule, ps)
    opt_statemin = deepcopy(opt_state)
    l_min = Inf
    ps_min = deepcopy(ps)
    dps = deepcopy(ps) .= 0
    wall_clock = [time_ns()]
    losses = zeros(maxiter)
    mean_loss = mean(losses)
    ind_losses = [individual_loss(ps, truth, param, x₀, sts, NNs, length(timeframes)) for (truth, x₀, param) in zip(truths[sim_index], x₀s[sim_index], params[sim_index])]
    loss_prefactors = compute_loss_prefactor_density_contribution.(ind_losses, compute_density_contribution.(train_data.data), S_scaling)

    for iter in 1:maxiter
        _, l = autodiff(Enzyme.ReverseWithPrimal, 
                        loss_multipleics, 
                        Active, 
                        DuplicatedNoNeed(ps, dps), 
                        DuplicatedNoNeed(truths[sim_index], deepcopy(truths[sim_index])), 
                        DuplicatedNoNeed(params[sim_index], deepcopy(params[sim_index])), 
                        DuplicatedNoNeed(x₀s[sim_index], deepcopy(x₀s[sim_index])), 
                        Const(sts), 
                        Const(NNs), 
                        DuplicatedNoNeed(loss_prefactors, deepcopy(loss_prefactors)),
                        Const(length(timeframes)))
        if iter <= 40
            Optimisers.adjust!(opt_state, eta=rule.eta * iter / 40)
        end
        
        opt_state, ps = Optimisers.update!(opt_state, ps, dps)

        losses[iter] = l
        if iter == 1
            l_min = l
        end
        
        @printf("%s, Δt %s, round %d, iter %d/%d, loss average %6.10e, minimum loss %6.5e, max NN weight %6.5e, gradient norm %6.5e\n",
                Dates.now(), prettytime(1e-9 * (time_ns() - wall_clock[1])), epoch, iter, maxiter, l, l_min,
                maximum(abs, ps), maximum(abs, dps))
        
        dps .= 0
        mean_loss = l
        if mean_loss < l_min
            l_min = mean_loss
            opt_statemin = deepcopy(opt_state)
            ps_min .= ps
        end

        if iter % 500 == 0
            @info "Saving intermediate results"
            jldsave("$(FILE_DIR)/intermediate_training_results_round$(epoch)_epoch$(iter)_end$(timeframes[end]).jld2"; u=ps_min, state=opt_statemin, loss=l_min)
            # sols = [diagnose_fields(ps_min, param, x₀, ps_baseclosure, sts, NNs, data) for (data, x₀, param) in zip(train_data_plot.data, x₀s, params)]
            # for (index, sol) in enumerate(sols)
            #     animate_data(train_data_plot.data[index], sol.sols_dimensional, sol.fluxes, sol.diffusivities, sol.sols_dimensional_noNN, sol.fluxes_noNN, sol.diffusivities_noNN, index, FILE_DIR; epoch="intermediateround$(epoch)_$(iter)")
            # end
        end
        wall_clock = [time_ns()]
    end
    return ps_min, (; total=losses), opt_statemin
end

optimizers = [Optimisers.Adam(3e-4), Optimisers.Adam(3e-5), Optimisers.Adam(3e-5), Optimisers.Adam(1e-5), Optimisers.Adam(1e-5), Optimisers.Adam(1e-5), Optimisers.Adam(1e-5)]
maxiters = [2000, 5000, 5000, 2000, 2000, 2000, 2000]
end_epochs = cumsum(maxiters)

# optimizers = [Optimisers.Adam(1e-5), Optimisers.Adam(1e-5), Optimisers.Adam(1e-5)]
# maxiters = [2000, 2000, 2000]
# end_epochs = cumsum(maxiters) .+ 14000

sim_indices = 1:length(LES_FILE_DIRS)

training_timeframes = [timeframes[1][1:5], timeframes[1][1:5], timeframes[1][1:10], timeframes[1][1:15], timeframes[1][1:20], timeframes[1][1:25], timeframes[1][1:27]]
# training_timeframes = [timeframes[1][1:20], timeframes[1][1:25], timeframes[1][1:27]]


# optimizers = [Optimisers.Adam(3e-4)]
# maxiters = [5]
# end_epochs = cumsum(maxiters)

# training_timeframes = [timeframes[1][1:5]]

plot_timeframes = [training_timeframe[1]:training_timeframe[end] for training_timeframe in training_timeframes]
sols = nothing
for (i, (epoch, optimizer, maxiter, training_timeframe, plot_timeframe)) in enumerate(zip(end_epochs, optimizers, maxiters, training_timeframes, plot_timeframes))
    global ps = ps
    global sols = sols
    # ps, losses, opt_state = train_NDE_multipleics(ps, params, ps_baseclosure, sts, NNs, truths, x₀s, train_data_plot, training_timeframe, S_scaling; sim_index=sim_indices, epoch=i, maxiter=maxiter, rule=optimizer)
    ps, losses, opt_state = train_NDE_multipleics(ps, params, sts, NNs, truths, x₀s, train_data_plot, training_timeframe, S_scaling; sim_index=sim_indices, epoch=i, maxiter=maxiter, rule=optimizer)
    
    jldsave("$(FILE_DIR)/training_results_epoch$(epoch)_end$(training_timeframe[end]).jld2"; u=ps, losses=losses, state=opt_state)

    sols = [diagnose_fields(ps, param, x₀, sts, NNs, data, length(plot_timeframe)) for (data, x₀, param) in zip(train_data_plot.data[sim_indices], x₀s[sim_indices], params_plot[sim_indices])]

    for (i, index) in enumerate(sim_indices)
        sol = sols[i]
        animate_data(train_data_plot.data[index], sol.sols_dimensional, sol.fluxes, sol.diffusivities, index, FILE_DIR, length(plot_timeframe); suffix="epoch$(epoch)_end$(training_timeframe[end])")
    end

    plot_loss(losses, FILE_DIR; suffix="epoch$(epoch)_end$(training_timeframe[end])")

end