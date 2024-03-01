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

function find_min(a...)
    return minimum(minimum.([a...]))
end
  
function find_max(a...)
    return maximum(maximum.([a...]))
end

FILE_DIR = "./training_output/first_training"
mkpath(FILE_DIR)

LES_FILE_DIRS = [
    "./LES_training/linearTS_dTdz_0.0013_dSdz_-0.0014_QU_-0.0002_QT_3.0e-5_QS_-3.0e-5_T_4.3_S_33.5_f_-0.00012_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.0002_QT_0.0003_QS_-3.0e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.0002_QT_0.0003_QS_-3.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_-0.025_dSdz_-0.0045_QU_-0.0002_QT_-0.0003_QS_-3.0e-5_T_-3.6_S_33.9_f_-0.000125_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
]

field_datasets = [FieldDataset(FILE_DIR, backend=OnDisk()) for FILE_DIR in LES_FILE_DIRS]

full_timeframes = [1:length(data["ubar"].times) for data in field_datasets]
timeframes = [5:10:length(data["ubar"].times) for data in field_datasets]
train_data = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, timeframes)
coarse_size = 32

rng = Random.default_rng(123)

uw_NN = Chain(Dense(132, 512, leakyrelu), Dense(512, 31))
vw_NN = Chain(Dense(132, 512, leakyrelu), Dense(512, 31))
wT_NN = Chain(Dense(132, 512, leakyrelu), Dense(512, 31))
wS_NN = Chain(Dense(132, 512, leakyrelu), Dense(512, 31))

ps_uw, st_uw = Lux.setup(rng, uw_NN)
ps_vw, st_vw = Lux.setup(rng, vw_NN)
ps_wT, st_wT = Lux.setup(rng, wT_NN)
ps_wS, st_wS = Lux.setup(rng, wS_NN)

ps_uw = ps_uw |> ComponentArray .|> Float64
ps_vw = ps_vw |> ComponentArray .|> Float64
ps_wT = ps_wT |> ComponentArray .|> Float64
ps_wS = ps_wS |> ComponentArray .|> Float64

# ps_uw .*= 1e-6
# ps_vw .*= 1e-6
# ps_wT .*= 1e-6
# ps_wS .*= 1e-6

st_uw = st_uw
st_vw = st_vw
st_wT = st_wT
st_wS = st_wS

NNs = (uw=uw_NN, vw=vw_NN, wT=wT_NN, wS=wS_NN)
ps_NN = ComponentArray(uw=ps_uw, vw=ps_vw, wT=ps_wT, wS=ps_wS)
st_NN = (uw=st_uw, vw=st_vw, wT=st_wT, wS=st_wS)

function train_NDE(train_data, NNs, ps_NN, st_NN; coarse_size=32, dev=cpu_device(), maxiter=10)
    train_data = train_data |> dev
    x₀s = [vcat(data.profile.u.scaled[:, 1], data.profile.v.scaled[:, 1], data.profile.T.scaled[:, 1], data.profile.S.scaled[:, 1]) for data in train_data.data] |> dev
    eos = TEOS10EquationOfState()

    uw_NN, vw_NN, wT_NN, wS_NN = NNs.uw, NNs.vw, NNs.wT, NNs.wS

    params = [(                   f = data.metadata["coriolis_parameter"],
                                  τ = data.times[end] - data.times[1],
                        scaled_time = (data.times .- data.times[1]) ./ (data.times[end] - data.times[1]),
               scaled_original_time = data.metadata["original_times"] ./ (data.metadata["original_times"][end] - data.metadata["original_times"][1]),
                                 zC = data.metadata["zC"],
                                  H = data.metadata["original_grid"].Lz,
                        coarse_size = coarse_size, 
                                 Dᶜ = Dᶜ(coarse_size, data.metadata["zC"][2] - data.metadata["zC"][1]), 
                                 Dᶠ = Dᶠ(coarse_size, data.metadata["zF"][3] - data.metadata["zF"][2]),
                                 uw = (top=data.flux.uw.surface.scaled, bottom=data.flux.uw.bottom.scaled),
                                 vw = (top=data.flux.vw.surface.scaled, bottom=data.flux.vw.bottom.scaled),
                                 wT = (top=data.flux.wT.surface.scaled, bottom=data.flux.wT.bottom.scaled),
                                 wS = (top=data.flux.wS.surface.scaled, bottom=data.flux.wS.bottom.scaled),
                            scaling = train_data.scaling
               ) for data in train_data.data] |> dev
    
    function predict_NN_flux(x, p, params, st)
        x′ = vcat(x, params.uw.top, params.vw.top, params.wT.top, params.wS.top)

        uw = vcat(params.uw.bottom, first(uw_NN(x′, p.uw, st.uw)) .+ params.uw.bottom, params.uw.top)
        vw = vcat(params.vw.bottom, first(vw_NN(x′, p.vw, st.vw)) .+ params.vw.bottom, params.vw.top)
        wT = vcat(params.wT.bottom, first(wT_NN(x′, p.wT, st.wT)) .+ params.wT.bottom, params.wT.top)
        wS = vcat(params.wS.bottom, first(wS_NN(x′, p.wS, st.wS)) .+ params.wS.bottom, params.wS.top)

        return uw, vw, wT, wS
    end

    function NDE(x, p, t, params, st)
        coarse_size = params.coarse_size
        f = params.f
        Dᶜ = params.Dᶜ
        scaling = params.scaling
        τ, H = params.τ, params.H

        u = x[1:coarse_size]
        v = x[coarse_size+1:2*coarse_size]

        uw, vw, wT, wS = predict_NN_flux(x, p, params, st)

        du = -τ ./ H ./ scaling.u.σ .* (Dᶜ * inv(scaling.uw).(uw)) .+ f * τ ./ scaling.u.σ .* inv(scaling.v).(v)
        dv = -τ ./ H ./ scaling.v.σ .* (Dᶜ * inv(scaling.vw).(vw)) .- f * τ ./ scaling.v.σ .* inv(scaling.u).(u)
        dT = -τ ./ H ./ scaling.T.σ .* (Dᶜ * inv(scaling.wT).(wT))
        dS = -τ ./ H ./ scaling.S.σ .* (Dᶜ * inv(scaling.wS).(wS))

        return vcat(du, dv, dT, dS)
    end

    function predict_NDE(p)
        probs = [ODEProblem((x, p′, t) -> NDE(x, p′, t, param, st_NN), x₀, (param.scaled_time[1], param.scaled_time[end]), p) for (x₀, param) in zip(x₀s, params)]
        sols = [Array(solve(prob, DP5(), saveat=param.scaled_time, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), reltol=1e-3)) for (param, prob) in zip(params, probs)]
        return sols
    end

    function predict_NDE_posttraining(p)
        probs = [ODEProblem((x, p′, t) -> NDE(x, p′, t, param, st_NN), x₀, (param.scaled_original_time[1], param.scaled_original_time[end]), p) for (x₀, param) in zip(x₀s, params)]
        sols = [solve(prob, DP5(), saveat=param.scaled_original_time, reltol=1e-3) for (param, prob) in zip(params, probs)]
        return sols
    end

    function loss_NDE(p)
        preds = predict_NDE(p)

        us = [@view(pred[1:coarse_size, :]) for pred in preds]
        vs = [@view(pred[coarse_size+1:2*coarse_size, :]) for pred in preds]
        Ts = [@view(pred[2*coarse_size+1:3*coarse_size, :]) for pred in preds]
        Ss = [@view(pred[3*coarse_size+1:4*coarse_size, :]) for pred in preds]
        ρs = [param.scaling.ρ.(TEOS10.ρ′.(inv(param.scaling.T).(T), inv(param.scaling.S).(S), param.zC, Ref(eos)) .+ eos.reference_density) for (T, S, param) in zip(Ts, Ss, params)]

        vel_prefactor = 1e-2
        u_loss = mean(mean.([(data.profile.u.scaled .- u).^2 for (data, u) in zip(train_data.data, us)])) * vel_prefactor
        v_loss = mean(mean.([(data.profile.v.scaled .- v).^2 for (data, v) in zip(train_data.data, vs)])) * vel_prefactor
        T_loss = mean(mean.([(data.profile.T.scaled .- T).^2 for (data, T) in zip(train_data.data, Ts)]))
        S_loss = mean(mean.([(data.profile.S.scaled .- S).^2 for (data, S) in zip(train_data.data, Ss)]))
        ρ_loss = mean(mean.([(data.profile.ρ.scaled .- ρ).^2 for (data, ρ) in zip(train_data.data, ρs)]))

        loss = u_loss + v_loss + T_loss + S_loss + ρ_loss

        individual_loss = (u=u_loss, v=v_loss, T=T_loss, S=S_loss, ρ=ρ_loss)
        return loss, preds, individual_loss
    end

    iter = 0

    losses = zeros(maxiter+1)
    u_losses = zeros(maxiter+1)
    v_losses = zeros(maxiter+1)
    T_losses = zeros(maxiter+1)
    S_losses = zeros(maxiter+1)
    ρ_losses = zeros(maxiter+1)

    wall_clock = [time_ns()]

    callback = function (p, l, pred, ind_loss)
        @printf("%s, Δt %s, iter %d/%d, loss total %6.5e, u %6.5e, v %6.5e, T %6.5e, S %6.5e, ρ %6.5e\n",
                Dates.now(), prettytime(1e-9 * (time_ns() - wall_clock[1])), iter, maxiter, l, ind_loss.u, ind_loss.v, ind_loss.T, ind_loss.S, ind_loss.ρ)
        losses[iter+1] = l
        u_losses[iter+1] = ind_loss.u
        v_losses[iter+1] = ind_loss.v
        T_losses[iter+1] = ind_loss.T
        S_losses[iter+1] = ind_loss.S
        ρ_losses[iter+1] = ind_loss.ρ

        iter += 1
        wall_clock[1] = time_ns()
        return false
    end

    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_NDE(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, ps_NN)

    res = Optimization.solve(optprob, OptimizationOptimisers.ADAM(0.01), callback=callback, maxiters=maxiter)

    sols_posttraining = predict_NDE_posttraining(res.u)

    uw_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    vw_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    wT_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]
    wS_posttraining = [zeros(coarse_size+1, length(param.scaled_original_time)) for param in params]

    for (i, sol) in enumerate(sols_posttraining)
        for t in eachindex(sol.t)
            uw, vw, wT, wS = predict_NN_flux(sol[:, t], res.u, params[i], st_NN)
            uw_posttraining[i][:, t] .= uw
            vw_posttraining[i][:, t] .= vw
            wT_posttraining[i][:, t] .= wT
            wS_posttraining[i][:, t] .= wS
        end
    end

    flux_posttraining = (uw=uw_posttraining, vw=vw_posttraining, wT=wT_posttraining, wS=wS_posttraining)
    losses = (total=losses, u=u_losses, v=v_losses, T=T_losses, S=S_losses, ρ=ρ_losses)

    return res, loss_NDE(res.u), sols_posttraining, flux_posttraining, losses
end

@time res, loss, sols, fluxes, losses = train_NDE(train_data, NNs, ps_NN, st_NN, maxiter=3)

train_data_plot = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, full_timeframes)

#%%
fig = Figure(size=(1000, 600))
axtotalloss = CairoMakie.Axis(fig[1, 1], title="Total Loss", xlabel="Iterations", ylabel="Loss", yscale=log10)
axindividualloss = CairoMakie.Axis(fig[1, 2], title="Individual Loss", xlabel="Iterations", ylabel="Loss", yscale=log10)

lines!(axtotalloss, losses.total, label="Total Loss")

lines!(axindividualloss, losses.u, label="u")
lines!(axindividualloss, losses.v, label="v")
lines!(axindividualloss, losses.T, label="T")
lines!(axindividualloss, losses.S, label="S")
lines!(axindividualloss, losses.ρ, label="ρ")

axislegend(axindividualloss, position=:rt)
save("$(FILE_DIR)/losses.png", fig, px_per_unit=8)

#%%
function animate_data(train_data, sols, fluxes, index, FILE_DIR, coarse_size=32)
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
    # axRi = CairoMakie.Axis(fig[2, 5], title="Ri", xlabel="Ri", ylabel="z (m)")

    n = Observable(1)
    zC = train_data.data[index].metadata["zC"]
    zF = train_data.data[index].metadata["zF"]

    u_NDE = inv(train_data.scaling.u).(sols[index][1:coarse_size, :])
    v_NDE = inv(train_data.scaling.v).(sols[index][coarse_size+1:2*coarse_size, :])
    T_NDE = inv(train_data.scaling.T).(sols[index][2*coarse_size+1:3*coarse_size, :])
    S_NDE = inv(train_data.scaling.S).(sols[index][3*coarse_size+1:4*coarse_size, :])
    ρ_NDE = TEOS10.ρ′.(T_NDE, S_NDE, zC, Ref(TEOS10EquationOfState())) .+ TEOS10EquationOfState().reference_density

    uw_NDE = inv(train_data.scaling.uw).(fluxes.uw[index])
    vw_NDE = inv(train_data.scaling.vw).(fluxes.vw[index])
    wT_NDE = inv(train_data.scaling.wT).(fluxes.wT[index])
    wS_NDE = inv(train_data.scaling.wS).(fluxes.wS[index])

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

    uw_NDEₙ = @lift uw_NDE[:, $n]
    vw_NDEₙ = @lift vw_NDE[:, $n]
    wT_NDEₙ = @lift wT_NDE[:, $n]
    wS_NDEₙ = @lift wS_NDE[:, $n]


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
    lines!(axuw, uw_NDEₙ, zF, label="NDE")

    lines!(axvw, vw_truthₙ, zF, label="Truth")
    lines!(axvw, vw_NDEₙ, zF, label="NDE")

    lines!(axwT, wT_truthₙ, zF, label="Truth")
    lines!(axwT, wT_NDEₙ, zF, label="NDE")

    lines!(axwS, wS_truthₙ, zF, label="Truth")
    lines!(axwS, wS_NDEₙ, zF, label="NDE")

    Legend(fig[2, 5], axu, tellwidth=false)
    Label(fig[0, :], time_str, font=:bold, tellwidth=false)

    # display(fig)

    CairoMakie.record(fig, "$(FILE_DIR)/training_$(index).mp4", 1:Nt, framerate=15) do nn
        xlims!(axu, nothing, nothing)
        xlims!(axv, nothing, nothing)
        xlims!(axT, nothing, nothing)
        xlims!(axS, nothing, nothing)
        xlims!(axρ, nothing, nothing)
        xlims!(axuw, nothing, nothing)
        xlims!(axvw, nothing, nothing)
        xlims!(axwT, nothing, nothing)
        xlims!(axwS, nothing, nothing)
        n[] = nn
    end
end

for i in eachindex(field_datasets)
    animate_data(train_data_plot, sols, fluxes, i, FILE_DIR)
end
#%%
#=
train_data_plot = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, [1:145 for i in 1:2])
#%%
fig = Figure(size=(1000, 1000))
axu = CairoMakie.Axis(fig[1, 1], title="u", xlabel="", ylabel="z")
axv = CairoMakie.Axis(fig[1, 2], title="v", xlabel="", ylabel="z")
axT = CairoMakie.Axis(fig[2, 1], title="T", xlabel="", ylabel="z")
axS = CairoMakie.Axis(fig[2, 2], title="S", xlabel="", ylabel="z")

n = Observable(1)
sim = 1
Nt = length(timeframes[sim])

zC = train_data.data[sim].metadata["zC"]

u_truthₙ = @lift train_data.data[sim].profile.u.scaled[:, $n]
v_truthₙ = @lift train_data.data[sim].profile.v.scaled[:, $n]
T_truthₙ = @lift train_data.data[sim].profile.T.scaled[:, $n]
S_truthₙ = @lift train_data.data[sim].profile.S.scaled[:, $n]

u_trainₙ = @lift res[2][2][sim][1:coarse_size, $n]
v_trainₙ = @lift res[2][2][sim][coarse_size+1:2*coarse_size, $n]
T_trainₙ = @lift res[2][2][sim][2*coarse_size+1:3*coarse_size, $n]
S_trainₙ = @lift res[2][2][sim][3*coarse_size+1:4*coarse_size, $n]

lines!(axu, u_truthₙ, zC, label="Truth")
lines!(axu, u_trainₙ, zC, label="Training")

lines!(axv, v_truthₙ, zC, label="Truth")
lines!(axv, v_trainₙ, zC, label="Training")

lines!(axT, T_truthₙ, zC, label="Truth")
lines!(axT, T_trainₙ, zC, label="Training")

lines!(axS, S_truthₙ, zC, label="Truth")
lines!(axS, S_trainₙ, zC, label="Training")

Legend(fig[3, :], axu, orientation=:horizontal)

display(fig)

CairoMakie.record(fig, "./Data/NDE_training_test.mp4", 1:Nt, framerate=1) do nn
    n[] = nn
    xlims!(axu, nothing, nothing)
    xlims!(axv, nothing, nothing)
    xlims!(axT, nothing, nothing)
    xlims!(axS, nothing, nothing)
end
#%%
u_collocation = [collocate_data(data.profile.u.scaled, data.times ./ (data.times[end] - data.times[1]), EpanechnikovKernel()) for data in train_data.data]
v_collocation = [collocate_data(data.profile.v.scaled, data.times ./ (data.times[end] - data.times[1]), EpanechnikovKernel()) for data in train_data.data]
T_collocation = [collocate_data(data.profile.T.scaled, data.times ./ (data.times[end] - data.times[1]), EpanechnikovKernel()) for data in train_data.data]
S_collocation = [collocate_data(data.profile.S.scaled, data.times ./ (data.times[end] - data.times[1]), EpanechnikovKernel()) for data in train_data.data]

#%%
fig = Figure()
ax = CairoMakie.Axis(fig[1, 1])

for i in 1:32
    lines!(T_collocation[1][2][i, :], linestyle=:dash)
    lines!(ax, train_data.data[1].profile.T.scaled[i, :])
end

display(fig)
#%%
function train_NDE_collocation(train_data, u_collocation, v_collocation, T_collocation, S_collocation, rng, uw_NN, vw_NN, wT_NN, wS_NN, coarse_size=32)
    dev = cpu_device()
    train_data = train_data |> dev

    ps_uw, st_uw = Lux.setup(rng, uw_NN)
    ps_vw, st_vw = Lux.setup(rng, vw_NN)
    ps_wT, st_wT = Lux.setup(rng, wT_NN)
    ps_wS, st_wS = Lux.setup(rng, wS_NN)

    ps_uw = ps_uw |> ComponentArray .|> Float64 |> dev
    ps_vw = ps_vw |> ComponentArray .|> Float64 |> dev
    ps_wT = ps_wT |> ComponentArray .|> Float64 |> dev
    ps_wS = ps_wS |> ComponentArray .|> Float64 |> dev

    ps_uw .*= 1e-6
    ps_vw .*= 1e-6
    ps_wT .*= 1e-6
    ps_wS .*= 1e-6

    st_uw = st_uw |> dev
    st_vw = st_vw |> dev
    st_wT = st_wT |> dev
    st_wS = st_wS |> dev

    ps_NN = ComponentArray(uw=ps_uw, vw=ps_vw, wT=ps_wT, wS=ps_wS)
    st_NN = (uw=st_uw, vw=st_vw, wT=st_wT, wS=st_wS)

    params = [(          f = data.metadata["coriolis_parameter"],
                         τ = data.times[end] - data.times[1],
               scaled_time = data.times ./ (data.times[end] - data.times[1]),
                         H = data.metadata["original_grid"].Lz,
               coarse_size = coarse_size, 
                        Dᶜ = Dᶜ(coarse_size, data.metadata["zC"][2] - data.metadata["zC"][1]), 
                        Dᶠ = Dᶠ(coarse_size, data.metadata["zF"][3] - data.metadata["zF"][2]),
                        uw = (top=data.flux.uw.surface.scaled, bottom=data.flux.uw.bottom.scaled),
                        vw = (top=data.flux.vw.surface.scaled, bottom=data.flux.vw.bottom.scaled),
                        wT = (top=data.flux.wT.surface.scaled, bottom=data.flux.wT.bottom.scaled),
                        wS = (top=data.flux.wS.surface.scaled, bottom=data.flux.wS.bottom.scaled),
                   scaling = train_data.scaling
               ) for data in train_data.data] |> dev

    function NDE(x, p, t, params, st)
        coarse_size = params.coarse_size
        f = params.f
        Dᶜ = params.Dᶜ
        scaling = params.scaling
        τ, H = params.τ, params.H

        uw_boundary_flux = params.uw
        vw_boundary_flux = params.vw
        wT_boundary_flux = params.wT
        wS_boundary_flux = params.wS

        u = x[1:coarse_size]
        v = x[coarse_size+1:2*coarse_size]

        uw = vcat(uw_boundary_flux.bottom, first(uw_NN(x, p.uw, st.uw)), uw_boundary_flux.top)
        vw = vcat(vw_boundary_flux.bottom, first(vw_NN(x, p.vw, st.vw)), vw_boundary_flux.top)
        wT = vcat(wT_boundary_flux.bottom, first(wT_NN(x, p.wT, st.wT)), wT_boundary_flux.top)
        wS = vcat(wS_boundary_flux.bottom, first(wS_NN(x, p.wS, st.wS)), wS_boundary_flux.top)

        du = -τ ./ H ./ scaling.u.σ .* (Dᶜ * inv(scaling.uw).(uw)) .+ f * τ ./ scaling.u.σ .* inv(scaling.v).(v)
        dv = -τ ./ H ./ scaling.v.σ .* (Dᶜ * inv(scaling.vw).(vw)) .- f * τ ./ scaling.v.σ .* inv(scaling.u).(u)
        dT = -τ ./ H ./ scaling.T.σ .* (Dᶜ * inv(scaling.wT).(wT))
        dS = -τ ./ H ./ scaling.S.σ .* (Dᶜ * inv(scaling.wS).(wS))

        return vcat(du, dv, dT, dS)
    end

    function loss_collocation(p)
        loss = 0
        for (i, param) in enumerate(params)
            for n in eachindex(param.scaled_time)
                x = vcat(u_collocation[i][2][:, n], v_collocation[i][2][:, n], T_collocation[i][2][:, n], S_collocation[i][2][:, n])
                dx = vcat(u_collocation[i][1][:, n], v_collocation[i][1][:, n], T_collocation[i][1][:, n], S_collocation[i][1][:, n])
                t = param.scaled_time[n]
                se = (NDE(x, p, t, param, st_NN) .- dx).^2
                loss += 1e-3 * mean(se[1:2*coarse_size]) + mean(se[2*coarse_size+1:4*coarse_size])
            end
        end
        return loss
    end
        
    loss_collocation(ps_NN)

    callback = function (p, l)
        @info l
        return false
    end

    callback(ps_NN, loss_collocation(ps_NN))

    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_collocation(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, ps_NN)

    res = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01), callback=callback, maxiters=1000)

    x₀s = [vcat(data.profile.u.scaled[:, 1], data.profile.v.scaled[:, 1], data.profile.T.scaled[:, 1], data.profile.S.scaled[:, 1]) for data in train_data.data] |> dev
    tspan = (0, 1)

    function predict_NDE(p)
        probs = [ODEProblem((x, p′, t) -> NDE(x, p′, t, param, st_NN), x₀, tspan, p) for (x₀, param) in zip(x₀s, params)]
        sols = [Array(solve(prob, ROCK4(), saveat=param.scaled_time, reltol=1e-3)) for (param, prob) in zip(params, probs)]
        return sols
    end

    return res, loss_collocation(res.u), predict_NDE(res.u)
end

res_collocation = train_NDE_collocation(train_data, u_collocation, v_collocation, T_collocation, S_collocation, rng, uw_NN, vw_NN, wT_NN, wS_NN)
#%%
fig = Figure(size=(1000, 1000))
axu = CairoMakie.Axis(fig[1, 1], title="u", xlabel="", ylabel="z")
axv = CairoMakie.Axis(fig[1, 2], title="v", xlabel="", ylabel="z")
axT = CairoMakie.Axis(fig[2, 1], title="T", xlabel="", ylabel="z")
axS = CairoMakie.Axis(fig[2, 2], title="S", xlabel="", ylabel="z")

n = Observable(1)
sim = 1
Nt = length(timeframes[sim])

zC = train_data.data[sim].metadata["zC"]

u_truthₙ = @lift train_data.data[sim].profile.u.scaled[:, $n]
v_truthₙ = @lift train_data.data[sim].profile.v.scaled[:, $n]
T_truthₙ = @lift train_data.data[sim].profile.T.scaled[:, $n]
S_truthₙ = @lift train_data.data[sim].profile.S.scaled[:, $n]

u_trainₙ = @lift res_collocation[3][sim][1:coarse_size, $n]
v_trainₙ = @lift res_collocation[3][sim][coarse_size+1:2*coarse_size, $n]
T_trainₙ = @lift res_collocation[3][sim][2*coarse_size+1:3*coarse_size, $n]
S_trainₙ = @lift res_collocation[3][sim][3*coarse_size+1:4*coarse_size, $n]

lines!(axu, u_truthₙ, zC, label="Truth")
lines!(axu, u_trainₙ, zC, label="Training (Collocation)")

lines!(axv, v_truthₙ, zC, label="Truth")
lines!(axv, v_trainₙ, zC, label="Training (Collocation)")

lines!(axT, T_truthₙ, zC, label="Truth")
lines!(axT, T_trainₙ, zC, label="Training (Collocation)")

lines!(axS, S_truthₙ, zC, label="Truth")
lines!(axS, S_trainₙ, zC, label="Training (Collocation)")

Legend(fig[3, :], axu, orientation=:horizontal)

display(fig)

CairoMakie.record(fig, "./Data/NDE_training_collocation_test.mp4", 1:Nt, framerate=1) do nn
    n[] = nn
    xlims!(axu, nothing, nothing)
    xlims!(axv, nothing, nothing)
    xlims!(axT, nothing, nothing)
    xlims!(axS, nothing, nothing)
end
#%%
fig = Figure()
ax = CairoMakie.Axis(fig[1, 1])
lines!(ax, train_data.data[1].profile.T.unscaled[:, 1], range(0, stop=1, length=32))
lines!(ax, train_data.data[1].profile.T.unscaled[:, end], range(0, stop=1, length=32))
lines!(ax, interior(field_datasets[1]["Tbar"][131], 1, 1, :), range(0, stop=1, length=128), linestyle=:dash)
display(fig)

#%%
=#