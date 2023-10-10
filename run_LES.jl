using Oceananigans
using Oceananigans.Units
using JLD2
using FileIO
using Printf
using CairoMakie
using Oceananigans.Grids: halo_size
using Oceananigans.Operators
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.BuoyancyModels
using SeawaterPolynomials.TEOS10
using Random
using Statistics

Random.seed!(123)

const Lz = 256meter    # depth [m]
const Lx = 512meter
const Ly = 512meter

const Nz = 128
const Nx = 256
const Ny = 256

# const Nz = 32
# const Nx = 64
# const Ny = 64

const Qᵁ = -1e-6
const Qᵀ = 2e-8
const Qˢ = 2e-5

const Pr = 1
const ν = 1e-5
const κ = ν / Pr

const f = 1e-4

const λᵀ = 8e-5
const λˢ = 6e-4

const T_surface = 20
const S_surface = 35

FILE_NAME = "QU_$(Qᵁ)_QT_$(Qᵀ)_Qs_$(Qˢ)"
FILE_DIR = "LES/$(FILE_NAME)"
mkpath(FILE_DIR)

grid = RectilinearGrid(GPU(), Float64,
                       size = (Nx, Ny, Nz),
                       halo = (5, 5, 5),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (-Lz, 0),
                       topology = (Periodic, Periodic, Bounded))

noise(x, y, z) = rand() * exp(z / 8)

T_initial(x, y, z) = T_surface * exp(λᵀ*z) + 1e-6 * noise(x, y, z)
S_initial(x, y, z) = S_surface * exp(λˢ*z) + 1e-6 * noise(x, y, z)

dTdz_bot = T_surface * λᵀ * exp(-λᵀ*Lz)
dSdz_bot = S_surface * λˢ * exp(-λˢ*Lz)

T_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵀ), bottom=GradientBoundaryCondition(dTdz_bot))
S_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qˢ), bottom=GradientBoundaryCondition(dSdz_bot))
u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵁ))

const eos = TEOS10EquationOfState()

model = NonhydrostaticModel(; 
            grid = grid,
            closure = ScalarDiffusivity(ν=ν, κ=κ),
            coriolis = FPlane(f=f),
            buoyancy = SeawaterBuoyancy(equation_of_state=eos),
            tracers = (:T, :S, :b),
            timestepper = :RungeKutta3,
            advection = WENO(order=9),
            boundary_conditions = (T=T_bcs, S=S_bcs, u=u_bcs)
            )

set!(model, T=T_initial, S=S_initial)
# set!(model, T=20, S=32)

T = model.tracers.T
S = model.tracers.S
b = model.tracers.b
u, v, w = model.velocities

const T₀ = mean(T)
const S₀ = mean(S)
const ρ₀ = TEOS10.ρ(T₀, S₀, 0, eos)

simulation = Simulation(model, Δt=0.1second, stop_time=0.5days)

wizard = TimeStepWizard(max_change=1.05, max_Δt=10minutes, cfl=0.6)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

wall_clock = [time_ns()]

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, max(T) %6.3e, max(S) %6.3e, next Δt: %s\n",
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            maximum(abs, sim.model.tracers.T),
            maximum(abs, sim.model.tracers.S),
            prettytime(sim.Δt))

    wall_clock[1] = time_ns()

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(1000))

function init_save_some_metadata!(file, model)
    file["metadata/author"] = "Xin Kai Lee"
    file["metadata/parameters/coriolis_parameter"] = f
    file["metadata/parameters/momentum_flux"] = Qᵁ
    file["metadata/parameters/temperature_flux"] = Qᵀ
    file["metadata/parameters/salinity_flux"] = Qˢ
    file["metadata/parameters/temperature_escale"] = λᵀ
    file["metadata/parameters/salinity_escale"] = λˢ
    return nothing
end

function calculate_thermal_sensitivity(i, j, k, grid, T, S, eos)
    return - Oceananigans.BuoyancyModels.thermal_expansionᶜᶜᶜ(i, j, k, grid, eos, T, S)
end

function calculate_haline_sensitivity(i, j, k, grid, T, S, eos)
    return Oceananigans.BuoyancyModels.haline_contractionᶜᶜᶜ(i, j, k, grid, eos, T, S)
end

α_op = KernelFunctionOperation{Center, Center, Center}(calculate_thermal_sensitivity, grid, T, S, eos)
α = Field(α_op)
compute!(α)

β_op = KernelFunctionOperation{Center, Center, Center}(calculate_haline_sensitivity, grid, T, S, eos)
β = Field(β_op)
compute!(β)

αΔT = Field(α * (T - T₀))
compute!(αΔT)

βΔS = Field(β * (S - S₀))
compute!(βΔS)

function get_buoyancy_perturbation(i, j, k, grid, b, C)
    T, S = Oceananigans.BuoyancyModels.get_temperature_and_salinity(b, C)
    ρ = Oceananigans.BuoyancyModels.ρ′(i, j, k, grid, b.model.equation_of_state, T, S) + b.model.equation_of_state.reference_density
    ρ′ = ρ - ρ₀
    return - b.model.gravitational_acceleration * ρ′ / ρ₀
end

b_op = KernelFunctionOperation{Center, Center, Center}(get_buoyancy_perturbation, grid, model.buoyancy, model.tracers)
b = Field(b_op)
compute!(b)

ubar = Average(u, dims=(1, 2))
vbar = Average(v, dims=(1, 2))

bbar = Average(b, dims=(1, 2))
b′bar = Average(model.buoyancy.model.gravitational_acceleration*(αΔT - βΔS), dims=(1, 2))
Tbar = Average(T, dims=(1, 2))
Sbar = Average(S, dims=(1, 2))

uw = Average(w * u, dims=(1, 2))
vw = Average(w * v, dims=(1, 2))
wb = Average(w * b, dims=(1, 2))
wT = Average(w * T, dims=(1, 2))
wS = Average(w * S, dims=(1, 2))

field_outputs = merge(model.velocities, model.tracers)

simulation.output_writers[:jld2] = JLD2OutputWriter(model, field_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_fields.jld2",
                                                          schedule = TimeInterval(10minutes),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:timeseries] = JLD2OutputWriter(model, (; ubar, vbar, Tbar, Sbar, uw, vw, wb, wT, wS),
                                                          filename = "$(FILE_DIR)/instantaneous_timeseries.jld2",
                                                          schedule = TimeInterval(10minutes),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=TimeInterval(1day), prefix="$(FILE_DIR)/model_checkpoint")

# run!(simulation, pickup="$(FILE_DIR)/model_checkpoint_iteration97574.jld2")
run!(simulation)

T_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "T", backend=OnDisk())
S_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "S", backend=OnDisk())

ubar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "ubar")
vbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "vbar")
Tbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "Tbar")
Sbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "Sbar")

uw_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "uw")
vw_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "vw")
wT_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wT")
wS_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wS")

Nt = length(T_data.times)

xC = T_data.grid.xᶜᵃᵃ[1:Nx]
yC = T_data.grid.xᶜᵃᵃ[1:Ny]
zC = T_data.grid.zᵃᵃᶜ[1:Nz]

zF = uw_data.grid.zᵃᵃᶠ[1:Nz+1]
##
fig = Figure(resolution=(1500, 1500))

axT = Axis3(fig[1:2, 1:2], title="T", xlabel="x", ylabel="y", zlabel="z", viewmode=:fitzoom, aspect=:data)
axS = Axis3(fig[1:2, 3:4], title="S", xlabel="x", ylabel="y", zlabel="z", viewmode=:fitzoom, aspect=:data)

axubar = Axis(fig[3, 1], title="ū", xlabel="ū", ylabel="z")
axvbar = Axis(fig[3, 2], title="v̄", xlabel="v̄", ylabel="z")
axTbar = Axis(fig[3, 3], title="T̄", xlabel="T̄", ylabel="z")
axSbar = Axis(fig[3, 4], title="S̄", xlabel="S̄", ylabel="z")

axuw = Axis(fig[4, 1], title="uw", xlabel="uw", ylabel="z")
axvw = Axis(fig[4, 2], title="vw", xlabel="vw", ylabel="z")
axwT = Axis(fig[4, 3], title="wT", xlabel="wT", ylabel="z")
axwS = Axis(fig[4, 4], title="wS", xlabel="wS", ylabel="z")

xs_xy = xC
ys_xy = yC
zs_xy = [zC[Nz] for x in xs_xy, y in ys_xy]

ys_yz = yC
xs_yz = range(xC[1], stop=xC[1], length=length(zC))
zs_yz = zeros(length(xs_yz), length(ys_yz))
for j in axes(zs_yz, 2)
  zs_yz[:, j] .= zC
end

xs_xz = xC
ys_xz = range(yC[1], stop=yC[1], length=length(zC))
zs_xz = zeros(length(xs_xz), length(ys_xz))
for i in axes(zs_xz, 1)
  zs_xz[i, :] .= zC
end

Tlim = (minimum(T_data), maximum(T_data))
Slim = (minimum(S_data), maximum(S_data))

colormap = Reverse(:RdBu_10)
T_color_range = Tlim
S_color_range = Slim

ubarlim = (minimum(ubar_data), maximum(ubar_data))
vbarlim = (minimum(vbar_data), maximum(vbar_data))
Tbarlim = (minimum(Tbar_data), maximum(Tbar_data))
Sbarlim = (minimum(Sbar_data), maximum(Sbar_data))

uwlim = (minimum(uw_data), maximum(uw_data))
vwlim = (minimum(vw_data), maximum(vw_data))
wTlim = (minimum(wT_data), maximum(wT_data))
wSlim = (minimum(wS_data), maximum(wS_data))

n = Observable(1)

Tₙ_xy = @lift interior(T_data[$n], :, :, Nz)
Tₙ_yz = @lift transpose(interior(T_data[$n], 1, :, :))
Tₙ_xz = @lift interior(T_data[$n], :, 1, :)

Sₙ_xy = @lift interior(S_data[$n], :, :, Nz)
Sₙ_yz = @lift transpose(interior(S_data[$n], 1, :, :))
Sₙ_xz = @lift interior(S_data[$n], :, 1, :)

time_str = @lift "Qᵁ = $(Qᵁ), Qᵀ = $(Qᵀ), Qˢ = $(Qˢ), Time = $(round(T_data.times[$n]/24/60^2, digits=3)) days"
title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)

T_xy_surface = surface!(axT, xs_xy, ys_xy, zs_xy, color=Tₙ_xy, colormap=colormap, colorrange = T_color_range)
T_yz_surface = surface!(axT, xs_yz, ys_yz, zs_yz, color=Tₙ_yz, colormap=colormap, colorrange = T_color_range)
T_xz_surface = surface!(axT, xs_xz, ys_xz, zs_xz, color=Tₙ_xz, colormap=colormap, colorrange = T_color_range)

S_xy_surface = surface!(axS, xs_xy, ys_xy, zs_xy, color=Sₙ_xy, colormap=colormap, colorrange = S_color_range)
S_yz_surface = surface!(axS, xs_yz, ys_yz, zs_yz, color=Sₙ_yz, colormap=colormap, colorrange = S_color_range)
S_xz_surface = surface!(axS, xs_xz, ys_xz, zs_xz, color=Sₙ_xz, colormap=colormap, colorrange = S_color_range)

ubarₙ = @lift interior(ubar_data[$n], 1, 1, :)
vbarₙ = @lift interior(vbar_data[$n], 1, 1, :)
Tbarₙ = @lift interior(Tbar_data[$n], 1, 1, :)
Sbarₙ = @lift interior(Sbar_data[$n], 1, 1, :)

uwₙ = @lift interior(uw_data[$n], 1, 1, :)
vwₙ = @lift interior(vw_data[$n], 1, 1, :)
wTₙ = @lift interior(wT_data[$n], 1, 1, :)
wSₙ = @lift interior(wS_data[$n], 1, 1, :)

lines!(axubar, ubarₙ, zC)
lines!(axvbar, vbarₙ, zC)
lines!(axTbar, Tbarₙ, zC)
lines!(axSbar, Sbarₙ, zC)

lines!(axuw, uwₙ, zF)
lines!(axvw, vwₙ, zF)
lines!(axwT, wTₙ, zF)
lines!(axwS, wSₙ, zF)

xlims!(axubar, ubarlim)
xlims!(axvbar, vbarlim)
xlims!(axTbar, Tbarlim)
xlims!(axSbar, Sbarlim)

xlims!(axuw, uwlim)
xlims!(axvw, vwlim)
xlims!(axwT, wTlim)
xlims!(axwS, wSlim)

trim!(fig.layout)

record(fig, "$(FILE_DIR)/$(FILE_NAME).mp4", 1:Nt, framerate=15) do nn
    n[] = nn
end

@info "Animation completed"
