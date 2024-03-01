using Oceananigans
using JLD2
using SeawaterPolynomials.TEOS10
using SeawaterPolynomials
using Oceananigans.BuoyancyModels: g_Earth
using SeawaterPolynomials.TEOS10: ζ, r′, τ, s
using CairoMakie
using Optimization
using OptimizationOptimJL
using OptimizationBBO
using Zygote
using Statistics

FILE_DIR = "./LES/linearTS_dTdz_0.015625_dSdz_-0.00390625_QU_-0.001_QT_0.0001_QS_-0.0002_T_4.1_S_0.0_WENO9nu1e-5_Lxz_128.0_256.0_Nxz_64_128_t"

Tbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "Tbar")
Sbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "Sbar")

parameters = jldopen("$(FILE_DIR)/instantaneous_timeseries.jld2", "r") do file
    return Dict([(key, file["metadata/parameters/$(key)"]) for key in keys(file["metadata/parameters"])])
end 

const Nz = Tbar_data.grid.Nz
const zC = Tbar_data.grid.zᵃᵃᶜ[1:Nz]

bbar = zeros(size(interior(Tbar_data)))
cbar = zeros(size(interior(Tbar_data)))

const eos = TEOS10EquationOfState()
const ρ₀ = eos.reference_density

for k in axes(bbar, 3), l in axes(bbar, 4)
    T = interior(Tbar_data)[1, 1, k, l]
    S = interior(Sbar_data)[1, 1, k, l]
    z = zC[k]
    ρ = r′(τ(T), s(S), ζ(z))
    bbar[1, 1, k, l] = -g_Earth * (ρ - ρ₀) / ρ₀
end

for k in axes(cbar, 3), l in axes(cbar, 4)
    T = interior(Tbar_data)[1, 1, k, l]
    S = interior(Sbar_data)[1, 1, k, l]
    z = zC[k]
    α = SeawaterPolynomials.thermal_sensitivity(T, S, z, eos) / ρ₀
    β = SeawaterPolynomials.haline_sensitivity(T, S, z, eos) / ρ₀
    cbar[1, 1, k, l] = g_Earth * (α * T + β * S)
end

#%%
fig = Figure(size=(800, 400))
axb = Axis(fig[1, 1], xlabel="<b> (m/s²)", ylabel="z")
axc = Axis(fig[1, 2], xlabel="<c> (m/s²)", ylabel="z")

lines!(axb, bbar[1, 1, :, 1], zC)
lines!(axc, cbar[1, 1, :, 1], zC)
display(fig)
#%%
function objective(u, p)
    T_hat = u[1]
    S_hat = u[2]
    z = p.z
    b = p.b
    c = p.c

    ρ_hat = r′(τ(T_hat), s(S_hat), ζ(z))
    b_hat = -g_Earth * (ρ_hat - ρ₀) / ρ₀

    α_hat = SeawaterPolynomials.thermal_sensitivity(T_hat, S_hat, z, eos) / ρ₀
    β_hat = SeawaterPolynomials.haline_sensitivity(T_hat, S_hat, z, eos) / ρ₀
    c_hat = g_Earth * (α_hat * T_hat + β_hat * S_hat)

    return (b - b_hat)^2 + (10*(c-c_hat))^2
    # return abs(b - b_hat) + abs(c-c_hat)
end

# index = 128
# timeframe = 1
# p = (z=zC[index], b=bbar[1, 1, index, timeframe], c=cbar[1, 1, index, timeframe])
# u0 = [4., 0]

# objf = OptimizationFunction(objective, Optimization.AutoForwardDiff())
# prob = OptimizationProblem(objf, u0, p, lb = [-0.1, -0.1], ub = [4.5, 2])

# TS_hats = zeros(2, 200)
# losses = zeros(200)
# for i in 1:200
#     # sol = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited())
#     # sol = solve(prob, BBO_adaptive_de_rand_1_bin())
#     sol = solve(prob, BBO_generating_set_search())
#     # sol = solve(prob, BBO_probabilistic_descent())
#     TS_hats[:, i] .= sol.u
#     losses[i] = sol.minimum
# end

# TS_hats[1, argmin(losses)]
# interior(Tbar_data)[1, 1, index, timeframe]
# median(TS_hats[1, :])

# lines(TS_hats[1, :])
# # sol = solve(prob, LBFGS())

# sol.u[1] - interior(Tbar_data)[1, 1, index, timeframe]
# sol.u[2] - interior(Sbar_data)[1, 1, index, timeframe]

#%%
# objective([interior(Tbar_data)[1, 1, index, timeframe], interior(Sbar_data)[1, 1, index, timeframe]], p)

# objective(sol.u, p)

#%%

function objective_column(u, p)
    b = p.b
    c = p.c

    T_hat = u[1:Nz]
    S_hat = u[Nz+1:end]

    ρ_hat = r′.(τ.(T_hat), s.(S_hat), ζ.(zC))
    b_hat = -g_Earth .* (ρ_hat .- ρ₀) ./ ρ₀

    α_hat = [SeawaterPolynomials.thermal_sensitivity(T_hat[i], S_hat[i], zC[i], eos) / ρ₀ for i in eachindex(zC)]
    β_hat = [SeawaterPolynomials.haline_sensitivity(T_hat[i], S_hat[i], zC[i], eos) / ρ₀ for i in eachindex(zC)]
    c_hat = g_Earth .* (α_hat .* T_hat .+ β_hat .* S_hat)

    return mean((b .- b_hat).^2) + mean((1 .* (c .- c_hat)).^2) + 0.0001 * mean(diff(T_hat).^2) + 0.001 * mean(diff(S_hat).*2)
    # return abs(b - b_hat) + abs(c-c_hat)
    # return mean((10 .* (b .- b_hat)).^2) + mean((1 .* (c .- c_hat)).^2)
end

timeframe = 1
p = (b=bbar[1, 1, :, timeframe], c=cbar[1, 1, :, timeframe])
u0 = vcat(2 .* ones(Nz), 0.5 .* ones(Nz))

objf = OptimizationFunction(objective_column, Optimization.AutoZygote())
prob = OptimizationProblem(objf, u0, p, lb = vcat(-0.01 .* ones(Nz), -0.01 .* ones(Nz)), ub = vcat(4.5 .* ones(Nz), 2 .* ones(Nz)))

sol = solve(prob, GradientDescent())

mean(sol.u[1:Nz] .- interior(Tbar_data)[1, 1, :, timeframe])
mean(sol.u[Nz+1:end] .- interior(Sbar_data)[1, 1, :, timeframe])
#%%
fig = Figure(size=(800, 400))
axT = Axis(fig[1, 1], title="T")
axS = Axis(fig[1, 2], title="S")
lines!(axT, interior(Tbar_data)[1, 1, :, timeframe], zC, label="truth")
lines!(axT, sol.u[1:Nz], zC, label="estimation")
lines!(axS, interior(Sbar_data)[1, 1, :, timeframe], zC, label="truth")
lines!(axS, sol.u[Nz+1:end], zC, label="estimation")
axislegend(axS)

display(fig)
#%%
function find_TS(bbar, cbar, zC)
    T_hatbar_argmin = zeros(size(bbar))
    S_hatbar_argmin = zeros(size(bbar))
    T_hatbar_median = zeros(size(bbar))
    S_hatbar_median = zeros(size(bbar))
    objf = OptimizationFunction(objective, Optimization.AutoForwardDiff())
    u0 = [2., 0]

    Threads.@threads for k in axes(T_hatbar_argmin, 3)
        z = zC[k]
        @info "z = $(z)"
        for l in axes(T_hatbar_argmin,4)
            b = bbar[1, 1, k, l]
            c = cbar[1, 1, k, l]
            p = (z=z, b=b, c=c)
            prob = OptimizationProblem(objf, u0, p, lb = [-0.001, -0.001], ub = [4.5, 2])
            TS_hats = zeros(2, 100)
            losses = zeros(size(TS_hats, 2))

            for iter in axes(TS_hats, 2)
                sol = solve(prob, BBO_generating_set_search())
                TS_hats[:, iter] .= sol.u
                losses[iter] = sol.minimum
            end

            index = argmin(losses)

            T_hatbar_argmin[1, 1, k, l] = TS_hats[1, index]
            S_hatbar_argmin[1, 1, k, l] = TS_hats[2, index]

            T_hatbar_median[1, 1, k, l] = median(@view(TS_hats[1, :]))
            S_hatbar_median[1, 1, k, l] = median(@view(TS_hats[2, :]))
            # u0[1] = sol.u[1]
            # u0[2] = sol.u[2]
        end
    end
    return T_hatbar_argmin, S_hatbar_argmin, T_hatbar_median, S_hatbar_median
end

# T_hatbar, S_hatbar = find_TS(bbar, cbar, zC)
T_hatbar_argmin, S_hatbar_argmin, T_hatbar_median, S_hatbar_median = find_TS(bbar[:, :, :, 576:577], cbar[:, :, :, 576:577], zC)

#%%
##
fig = Figure(size=(2000, 2000))

Nt = length(Tbar_data.times)

axbbar = Axis(fig[1, 1], title="<b>", xlabel="m s⁻²", ylabel="z")
axcbar = Axis(fig[1, 2], title="<c>", xlabel="m s⁻²", ylabel="z")
axTbar = Axis(fig[2, 1], title="<T>", xlabel="°C", ylabel="z")
axSbar = Axis(fig[2, 2], title="<S>", xlabel="g kg⁻¹", ylabel="z")

function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

bbarlim = (minimum(bbar), maximum(bbar))
cbarlim = (minimum(cbar), maximum(cbar))
Tbarlim = (find_min(interior(Tbar_data), T_hatbar_argmin, T_hatbar_median),
           find_max(interior(Tbar_data), T_hatbar_argmin, T_hatbar_median))

Sbarlim = (find_min(interior(Sbar_data), S_hatbar_argmin, S_hatbar_median),
           find_max(interior(Sbar_data), S_hatbar_argmin, S_hatbar_median))

n = Observable(576)

Qᵁ = parameters["momentum_flux"]
Qᵀ = parameters["temperature_flux"]
Qˢ = parameters["salinity_flux"]

time_str = @lift "Qᵁ = $(Qᵁ), Qᵀ = $(Qᵀ), Qˢ = $(Qˢ), Time = $(round(Tbar_data.times[$n]/24/60^2, digits=3)) days"
title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)

bbarₙ = @lift bbar[1, 1, :, $n]
cbarₙ = @lift cbar[1, 1, :, $n]

Tbarₙ = @lift interior(Tbar_data[$n], 1, 1, :)
Sbarₙ = @lift interior(Sbar_data[$n], 1, 1, :)
T_hatbar_argminₙ = @lift T_hatbar_argmin[1, 1, :, $n - 575]
S_hatbar_argminₙ = @lift S_hatbar_argmin[1, 1, :, $n - 575]
T_hatbar_medianₙ = @lift T_hatbar_median[1, 1, :, $n - 575]
S_hatbar_medianₙ = @lift S_hatbar_median[1, 1, :, $n - 575]

lines!(axbbar, bbarₙ, zC)
lines!(axcbar, cbarₙ, zC)

lines!(axTbar, Tbarₙ, zC, label="Truth")
scatter!(axTbar, T_hatbar_argminₙ, zC, label="Reconstructed (argmin(L))")
scatter!(axTbar, T_hatbar_medianₙ, zC, label="Reconstructed (median)")
axislegend(axTbar)

lines!(axSbar, Sbarₙ, zC, label="Truth")
scatter!(axSbar, S_hatbar_argminₙ, zC, label="Reconstructed (argmin(L))")
scatter!(axSbar, S_hatbar_medianₙ, zC, label="Reconstructed (median)")
axislegend(axSbar)

xlims!(axbbar, bbarlim)
xlims!(axcbar, cbarlim)
xlims!(axTbar, Tbarlim)
xlims!(axSbar, Sbarlim)

trim!(fig.layout)
save("$(FILE_DIR)/TS_inversion_n_$(n.val)_abs.png", fig, px_per_unit=4)
display(fig)
#%%
record(fig, "$(FILE_DIR)/inversion_video_small_s.mp4", 1:Nt, framerate=15) do nn
    n[] = nn
end

@info "Animation completed"
#%%