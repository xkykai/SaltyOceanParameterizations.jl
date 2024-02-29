using SaltyOceanParameterizations
using SaltyOceanParameterizations.DataWrangling
using Oceananigans
using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL, OptimizationOptimisers, Random, SciMLSensitivity, LuxCUDA
using Statistics
using CairoMakie
using SeawaterPolynomials.TEOS10

field_datasets = [FieldDataset("./LES/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0003_QS_3.0e-5_T_18.0_S_36.6_WENO9nu0_Lxz_256.0_128.0_Nxz_256_128/instantaneous_timeseries.jld2", backend=OnDisk()),
                  FieldDataset("./LES/linearTS_dTdz_-0.025_dSdz_-0.0045_QU_0.0_QT_-0.0003_QS_-3.0e-5_T_-3.6_S_33.9_WENO9nu0_Lxz_256.0_128.0_Nxz_256_128/instantaneous_timeseries.jld2", backend=OnDisk())]

timeframes = [5:10:145, 5:10:145]
train_data = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, timeframes)
coarse_size = 32

rng = Random.default_rng(123)

uw_NN = Chain(Dense(132, 50, leakyrelu), Dense(50, 31))
vw_NN = Chain(Dense(132, 50, leakyrelu), Dense(50, 31))
wT_NN = Chain(Dense(132, 50, leakyrelu), Dense(50, 31))
wS_NN = Chain(Dense(132, 50, leakyrelu), Dense(50, 31))

function train_NDE(train_data, rng, uw_NN, vw_NN, wT_NN, wS_NN, coarse_size=32)
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

    # ps_uw .*= 1e-6
    # ps_vw .*= 1e-6
    # ps_wT .*= 1e-6
    # ps_wS .*= 1e-6

    st_uw = st_uw |> dev
    st_vw = st_vw |> dev
    st_wT = st_wT |> dev
    st_wS = st_wS |> dev

    x₀s = [vcat(data.profile.u.scaled[:, 1], data.profile.v.scaled[:, 1], data.profile.T.scaled[:, 1], data.profile.S.scaled[:, 1]) for data in train_data.data] |> dev
    tspan = (0, 1)
    times = [data.times for data in train_data.data]

    ps_NN = ComponentArray(uw=ps_uw, vw=ps_vw, wT=ps_wT, wS=ps_wS)
    st_NN = (uw=st_uw, vw=st_vw, wT=st_wT, wS=st_wS)

    eos = TEOS10EquationOfState()

    params = [(          f = data.metadata["coriolis_parameter"],
                         τ = data.times[end] - data.times[1],
               scaled_time = (data.times .- data.times[1]) ./ (data.times[end] - data.times[1]),
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

    # p_NDE = (ps_NN=ps_NN, metadata=p_metadata, st_NN=st_NN)

    function NDE(x, p, t, params, st)
        coarse_size = params.coarse_size
        f = params.f
        Dᶜ = params.Dᶜ
        scaling = params.scaling
        τ, H = params.τ, params.H

        u = x[1:coarse_size]
        v = x[coarse_size+1:2*coarse_size]

        x′ = vcat(x, params.uw.top, params.vw.top, params.wT.top, params.wS.top)

        uw = vcat(params.uw.bottom, first(uw_NN(x′, p.uw, st.uw)) .+ params.uw.bottom , params.uw.top)
        vw = vcat(params.vw.bottom, first(vw_NN(x′, p.vw, st.vw)) .+ params.vw.bottom , params.vw.top)
        wT = vcat(params.wT.bottom, first(wT_NN(x′, p.wT, st.wT)) .+ params.wT.bottom , params.wT.top)
        wS = vcat(params.wS.bottom, first(wS_NN(x′, p.wS, st.wS)) .+ params.wS.bottom , params.wS.top)

        du = -τ ./ H ./ scaling.u.σ .* (Dᶜ * inv(scaling.uw).(uw)) .+ f * τ ./ scaling.u.σ .* inv(scaling.v).(v)
        dv = -τ ./ H ./ scaling.v.σ .* (Dᶜ * inv(scaling.vw).(vw)) .- f * τ ./ scaling.v.σ .* inv(scaling.u).(u)
        dT = -τ ./ H ./ scaling.T.σ .* (Dᶜ * inv(scaling.wT).(wT))
        dS = -τ ./ H ./ scaling.S.σ .* (Dᶜ * inv(scaling.wS).(wS))

        return vcat(du, dv, dT, dS)
    end

    function predict_NDE(p)
        probs = [ODEProblem((x, p′, t) -> NDE(x, p′, t, param, st_NN), x₀, tspan, p) for (x₀, param) in zip(x₀s, params)]
        sols = [Array(solve(prob, DP5(), saveat=param.scaled_time, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), reltol=1e-3)) for (param, prob) in zip(params, probs)]
        return sols
    end

    function loss_NDE(p)
        preds = predict_NDE(p)

        us = [@view(pred[1:coarse_size, :]) for pred in preds]
        vs = [@view(pred[coarse_size+1:2*coarse_size, :]) for pred in preds]
        Ts = [@view(pred[2*coarse_size+1:3*coarse_size, :]) for pred in preds]
        Ss = [@view(pred[3*coarse_size+1:4*coarse_size, :]) for pred in preds]
        ρs = [param.scaling.ρ.(TEOS10.ρ′.(inv(param.scaling.T).(T), inv(param.scaling.S).(S), param.zC, Ref(eos)) .+ eos.reference_density) for (T, S, param) in zip(Ts, Ss, params)]

        u_loss = mean(mean.([(data.profile.u.scaled .- u).^2 for (data, u) in zip(train_data.data, us)]))
        v_loss = mean(mean.([(data.profile.v.scaled .- v).^2 for (data, v) in zip(train_data.data, vs)]))
        T_loss = mean(mean.([(data.profile.T.scaled .- T).^2 for (data, T) in zip(train_data.data, Ts)]))
        S_loss = mean(mean.([(data.profile.S.scaled .- S).^2 for (data, S) in zip(train_data.data, Ss)]))
        ρ_loss = mean(mean.([(data.profile.ρ.scaled .- ρ).^2 for (data, ρ) in zip(train_data.data, ρs)]))

        loss = 1e-5 * (u_loss + v_loss) + T_loss + S_loss + ρ_loss
        return loss, preds
    end

    iter = 0
    maxiter = 10
    callback = function (p, l, pred)
        @info "iter $(iter)/$(maxiter), loss = $l"
        iter += 1
        return false
    end

    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_NDE(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, ps_NN)

    res = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01), callback=callback, maxiters=maxiter)
    return res, loss_NDE(res.u)
end

@time res = train_NDE(train_data, rng, uw_NN, vw_NN, wT_NN, wS_NN)

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
