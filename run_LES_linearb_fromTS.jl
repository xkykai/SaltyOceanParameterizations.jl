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
using Oceananigans.BuoyancyModels: Zᶜᶜᶠ, Zᶜᶜᶜ, g_Earth
using SeawaterPolynomials.TEOS10
using SeawaterPolynomials
using Random
using Statistics
using ArgParse
using SeawaterPolynomials.TEOS10: ζ, r₀, r′, τ, s, R₀₀, R₀₁, R₀₂, R₀₃, R₀₄, R₀₅, r′₀, r′₁, r′₂, r′₃
using LinearAlgebra
include("correct_reduction_oceananigans.jl")

import Dates
using GibbsSeaWater

function parse_commandline()
    s = ArgParseSettings()
  
    @add_arg_table! s begin
      "--QU"
        help = "surface momentum flux (m²/s²)"
        arg_type = Float64
        default = 0.
      "--QT"
        help = "surface temperature flux (°C m/s)"
        arg_type = Float64
        default = 0.
      "--QS"
        help = "surface salinity flux (m/s g/kg)"
        arg_type = Float64
        default = 0.
      "--T_surface"
        help = "surface temperature (°C)"
        arg_type = Float64
        default = 20.
      "--S_surface"
        help = "surface salinity (g/kg)"
        arg_type = Float64
        default = 35.
      "--dTdz"
        help = "Initial temperature gradient (°C/m)"
        arg_type = Float64
        default = 1 / 256
      "--dSdz"
        help = "Initial salinity gradient (g/kg/m))"
        arg_type = Float64
        default = -0.25 / 256
      "--f"
        help = "Coriolis parameter (s⁻¹)"
        arg_type = Float64
        default = 1e-4
      "--Nz"
        help = "Number of grid points in z-direction"
        arg_type = Int64
        default = 128
      "--Nx"
        help = "Number of grid points in x-direction"
        arg_type = Int64
        default = 256
      "--Ny"
        help = "Number of grid points in y-direction"
        arg_type = Int64
        default = 256
      "--Lz"
        help = "Domain depth"
        arg_type = Float64
        default = 256.
      "--Lx"
        help = "Domain width in x-direction"
        arg_type = Float64
        default = 512.
      "--Ly"
        help = "Domain width in y-direction"
        arg_type = Float64
        default = 512.
      "--dt"
        help = "Initial timestep to take (seconds)"
        arg_type = Float64
        default = 0.1
      "--max_dt"
        help = "Maximum timestep (seconds)"
        arg_type = Float64
        default = 10. * 60
      "--stop_time"
        help = "Stop time of simulation (days)"
        arg_type = Float64
        default = 4.
      "--time_interval"
        help = "Time interval of output writer (minutes)"
        arg_type = Float64
        default = 10.
      "--fps"
        help = "Frames per second of animation"
        arg_type = Float64
        default = 15.
      "--pickup"
        help = "Whether to pickup from latest checkpoint"
        arg_type = Bool
        default = true
      "--advection"
        help = "Advection scheme used"
        arg_type = String
        default = "WENO9nu1e-5"
      "--pickup_interval"
        help = "Interval of pickup writer (days)"
        arg_type = Float64
        default = 0.2
    end
    return parse_args(s)
end

args = parse_commandline()

Random.seed!(123)

# const Lz = args["Lz"]
# const Lx = args["Lx"]
# const Ly = args["Ly"]

# const Nz = args["Nz"]
# const Nx = args["Nx"]
# const Ny = args["Ny"]

# const Qᵁ = args["QU"]
# const Qᵀ = args["QT"]
# const Qˢ = args["QS"]

const Lz = 128
const Lx = 64
const Ly = 64

const Nz = 64
const Nx = 32
const Ny = 32

const Qᵁ = -4e-6
const Qᵀ = 1e-4
const Qˢ = -2e-4

const Pr = 1

if args["advection"] == "WENO9nu1e-5"
    advection = WENO(order=9)
    closure = ScalarDiffusivity(ν=1e-5, κ=1e-5/Pr)
elseif args["advection"] == "WENO9nu0"
    advection = WENO(order=9)
    closure = nothing
elseif args["advection"] == "WENO9AMD"
    advection = WENO(order=9)
    closure = AnisotropicMinimumDissipation()
elseif args["advection"] == "AMD"
    advection = CenteredSecondOrder()
    closure = AnisotropicMinimumDissipation()
end

const f = args["f"]

const dTdz = args["dTdz"]
const dSdz = args["dSdz"]

const T_surface = args["T_surface"]
const S_surface = args["S_surface"]

const pickup = args["pickup"]

FILE_NAME = "linearb_fromTS_dTdz_$(dTdz)_dSdz_$(dSdz)_QU_$(Qᵁ)_QT_$(Qᵀ)_QS_$(Qˢ)_T_$(T_surface)_S_$(S_surface)_$(args["advection"])_Lxz_$(Lx)_$(Lz)_Nxz_$(Nx)_$(Nz)"
FILE_DIR = "LES/$(FILE_NAME)"
mkpath(FILE_DIR)

size_halo = 5

const eos = TEOS10EquationOfState()
const ρ₀ = eos.reference_density

grid = RectilinearGrid(GPU(), Float64,
                       size = (Nx, Ny, Nz),
                       halo = (size_halo, size_halo, size_halo),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (-Lz, 0),
                       topology = (Periodic, Periodic, Bounded))

noise(x, y, z) = rand() * exp(z / 8)

T_vertical(z) = dTdz * z + T_surface
S_vertical(z) = dSdz * z + S_surface

T_initial(x, y, z) = T_vertical(z)
S_initial(x, y, z) = S_vertical(z)

T = Field{Nothing, Nothing, Center}(grid)
S = Field{Nothing, Nothing, Center}(grid)

set!(T, T_vertical)
set!(S, S_vertical)

@inline function b_initial(x, y, z)
    ρ = r′(τ(T_initial(x, y, z)), s(S_initial(x, y, z)), ζ(z))
    return -g_Earth * (ρ - ρ₀) / ρ₀
end

@inline function c_initial(x, y, z)
    α = SeawaterPolynomials.thermal_sensitivity(T_initial(x, y, z), S_initial(x, y, z), z, eos) / ρ₀
    β = SeawaterPolynomials.haline_sensitivity(T_initial(x, y, z), S_initial(x, y, z), z, eos) / ρ₀
    return g_Earth * (α * T_initial(x, y, z) + β * S_initial(x, y, z))
end

const dbdz_bot = (b_initial(nothing, nothing, Zᶜᶜᶜ(1, 1, 2, grid)) - b_initial(nothing, nothing, Zᶜᶜᶜ(1, 1, 1, grid))) / (Zᶜᶜᶜ(1, 1, 2, grid) - Zᶜᶜᶜ(1, 1, 1, grid))
const dcdz_bot = (c_initial(nothing, nothing, Zᶜᶜᶜ(1, 1, 2, grid)) - c_initial(nothing, nothing, Zᶜᶜᶜ(1, 1, 1, grid))) / (Zᶜᶜᶜ(1, 1, 2, grid) - Zᶜᶜᶜ(1, 1, 1, grid))

const α_top = SeawaterPolynomials.thermal_sensitivity(T_surface, S_surface, 0, eos) / ρ₀
const β_top = SeawaterPolynomials.haline_sensitivity(T_surface, S_surface, 0, eos) / ρ₀

Qᴮ = g_Earth * (α_top * Qᵀ - β_top * Qˢ)
Qᶜ = g_Earth * (α_top * Qᵀ + β_top * Qˢ)

b_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᴮ), bottom=GradientBoundaryCondition(dbdz_bot))
c_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᶜ), bottom=GradientBoundaryCondition(dcdz_bot))
u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵁ))

damping_rate = 1/5minute

b_target(x, y, z, t) = b_initial(x, y, z)
c_target(x, y, z, t) = c_initial(x, y, z)

bottom_mask = GaussianMask{:z}(center=-grid.Lz, width=grid.Lz/10)

uvw_sponge = Relaxation(rate=damping_rate, mask=bottom_mask)
b_sponge = Relaxation(rate=damping_rate, mask=bottom_mask, target=b_target)
c_sponge = Relaxation(rate=damping_rate, mask=bottom_mask, target=c_target)

model = NonhydrostaticModel(; 
            grid = grid,
            closure = closure,
            coriolis = FPlane(f=f),
            buoyancy = BuoyancyTracer(),
            tracers = (:b, :c),
            timestepper = :RungeKutta3,
            advection = advection,
            forcing = (u=uvw_sponge, v=uvw_sponge, w=uvw_sponge, b=b_sponge, c=c_sponge),
            boundary_conditions = (b=b_bcs, c=c_bcs, u=u_bcs)
            )

set!(model, b=b_initial, c=c_initial, w=noise)

b = model.tracers.b
c = model.tracers.c
u, v, w = model.velocities

simulation = Simulation(model, Δt=args["dt"]second, stop_time=args["stop_time"]days)
# simulation = Simulation(model, Δt=args["dt"]second, stop_time=100minutes)

wizard = TimeStepWizard(max_change=1.05, max_Δt=args["max_dt"]minutes, cfl=0.6)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

wall_clock = [time_ns()]

function print_progress(sim)
    @printf("%s [%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, max(b) %6.3e, max(c) %6.3e, next Δt: %s\n",
            Dates.now(),
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            maximum(abs, sim.model.tracers.b),
            maximum(abs, sim.model.tracers.c),
            prettytime(sim.Δt))

    wall_clock[1] = time_ns()

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(100))

function init_save_some_metadata!(file, model)
    file["metadata/author"] = "Xin Kai Lee"
    file["metadata/parameters/coriolis_parameter"] = f
    file["metadata/parameters/momentum_flux"] = Qᵁ
    file["metadata/parameters/temperature_flux"] = Qᵀ
    file["metadata/parameters/salinity_flux"] = Qˢ
    file["metadata/parameters/buoyancy_flux"] = Qᴮ
    file["metadata/parameters/spice_flux"] = Qᶜ
    file["metadata/parameters/surface_temperature"] = T_surface
    file["metadata/parameters/surface_salinity"] = S_surface
    file["metadata/parameters/temperature_gradient"] = dTdz
    file["metadata/parameters/salinity_gradient"] = dSdz
    file["metadata/parameters/buoyancy_gradient"] = dbdz
    file["metadata/parameters/spice_gradient"] = dcdz
    file["metadata/parameters/equation_of_state"] = eos
    file["metadata/parameters/gravitational_acceleration"] = g_Earth
    file["metadata/parameters/reference_density"] = ρ₀
    return nothing
end

ubar = Field(Average(u, dims=(1, 2)))
vbar = Field(Average(v, dims=(1, 2)))
bbar = Field(Average(b, dims=(1, 2)))
cbar = Field(Average(c, dims=(1, 2)))

uw = Field(Average(w * u, dims=(1, 2)))
vw = Field(Average(w * v, dims=(1, 2)))
wb = Field(Average(w * b, dims=(1, 2)))
wc = Field(Average(w * c, dims=(1, 2)))

field_outputs = merge(model.velocities, model.tracers)
timeseries_outputs = (; ubar, vbar, bbar, cbar,
                        uw, vw, wb, wc)
                        # p_sensitivity_bar, p_sensitivity_bulk)

# simulation.output_writers[:jld2] = JLD2OutputWriter(model, (; T, S, w_center),
#                                                     filename = "$(FILE_DIR)/instantaneous_fields.jld2",
#                                                     schedule = TimeInterval(10minutes),
#                                                     with_halos = true,
#                                                     init = init_save_some_metadata!)

simulation.output_writers[:xy_jld2] = JLD2OutputWriter(model, field_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_fields_xy.jld2",
                                                          schedule = TimeInterval(10minutes),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!,
                                                          indices = (:, :, Nz))

simulation.output_writers[:yz_jld2] = JLD2OutputWriter(model, field_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_fields_yz.jld2",
                                                          schedule = TimeInterval(10minutes),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!,
                                                          indices = (1, :, :))

simulation.output_writers[:xz_jld2] = JLD2OutputWriter(model, field_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_fields_xz.jld2",
                                                          schedule = TimeInterval(10minutes),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!,
                                                          indices = (:, 1, :))

simulation.output_writers[:timeseries] = JLD2OutputWriter(model, timeseries_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_timeseries.jld2",
                                                          schedule = TimeInterval(10minutes),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=TimeInterval(args["pickup_interval"]days), prefix="$(FILE_DIR)/model_checkpoint")

if pickup
    files = readdir(FILE_DIR)
    checkpoint_files = files[occursin.("model_checkpoint_iteration", files)]
    if !isempty(checkpoint_files)
        checkpoint_iters = parse.(Int, [filename[findfirst("iteration", filename)[end]+1:findfirst(".jld2", filename)[1]-1] for filename in checkpoint_files])
        pickup_iter = maximum(checkpoint_iters)
        run!(simulation, pickup="$(FILE_DIR)/model_checkpoint_iteration$(pickup_iter).jld2")
    else
        run!(simulation)
    end
else
    run!(simulation)
end

T_xy_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xy.jld2", "T", backend=OnDisk())
T_xz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz.jld2", "T", backend=OnDisk())
T_yz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz.jld2", "T", backend=OnDisk())

S_xy_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xy.jld2", "S", backend=OnDisk())
S_xz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz.jld2", "S", backend=OnDisk())
S_yz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz.jld2", "S", backend=OnDisk())

ubar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "ubar")
vbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "vbar")
Tbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "Tbar")
Sbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "Sbar")

uw_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "uw")
vw_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "vw")
wT_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wT")
wS_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wS")
wb_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wb")
wb_nor₀_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wb_nor₀")
wb′_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wb′")

Nt = length(T_xy_data.times)

xC = T_xy_data.grid.xᶜᵃᵃ[1:Nx]
yC = T_xy_data.grid.yᵃᶜᵃ[1:Ny]
zC = T_xy_data.grid.zᵃᵃᶜ[1:Nz]

zF = uw_data.grid.zᵃᵃᶠ[1:Nz+1]
##
fig = Figure(resolution=(1800, 1500))

axT = Axis3(fig[1:2, 1:2], title="T", xlabel="x", ylabel="y", zlabel="z", viewmode=:fitzoom, aspect=:data)
axS = Axis3(fig[1:2, 4:5], title="S", xlabel="x", ylabel="y", zlabel="z", viewmode=:fitzoom, aspect=:data)

axubar = Axis(fig[3, 1], title="<u>", xlabel="m s⁻¹", ylabel="z")
axvbar = Axis(fig[3, 2], title="<v>", xlabel="m s⁻¹", ylabel="z")
axTbar = Axis(fig[3, 3], title="<T>", xlabel="°C", ylabel="z")
axSbar = Axis(fig[3, 4], title="<S>", xlabel="g kg⁻¹", ylabel="z")

axuw = Axis(fig[4, 1], title="uw", xlabel="m² s⁻²", ylabel="z")
axvw = Axis(fig[4, 2], title="vw", xlabel="m² s⁻²", ylabel="z")
axwT = Axis(fig[4, 3], title="wT", xlabel="m s⁻¹ °C", ylabel="z")
axwS = Axis(fig[4, 4], title="wS", xlabel="m s⁻¹ g kg⁻¹", ylabel="z")
axwb = Axis(fig[4, 5], title="wb", xlabel="m² s⁻³", ylabel="z")

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

function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

Tlim = (find_min(T_xy_data, T_yz_data, T_xz_data), find_max(T_xy_data, T_yz_data, T_xz_data))
Slim = (find_min(S_xy_data, S_yz_data, S_xz_data), find_max(S_xy_data, S_yz_data, S_xz_data))

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
wblim = (find_min(wb_data, wb′_data), find_max(wb_data, wb′_data))

n = Observable(1)

Tₙ_xy = @lift interior(T_xy_data[$n], :, :, 1)
Tₙ_yz = @lift transpose(interior(T_yz_data[$n], 1, :, :))
Tₙ_xz = @lift interior(T_xz_data[$n], :, 1, :)

Sₙ_xy = @lift interior(S_xy_data[$n], :, :, 1)
Sₙ_yz = @lift transpose(interior(S_yz_data[$n], 1, :, :))
Sₙ_xz = @lift interior(S_xz_data[$n], :, 1, :)

time_str = @lift "Qᵁ = $(Qᵁ), Qᵀ = $(Qᵀ), Qˢ = $(Qˢ), Time = $(round(T_xy_data.times[$n]/24/60^2, digits=3)) days"
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
wbₙ = @lift interior(wb_data[$n], 1, 1, :)
wb_nor₀ₙ = @lift interior(wb_nor₀_data[$n], 1, 1, :)
wb′ₙ = @lift interior(wb′_data[$n], 1, 1, :)

lines!(axubar, ubarₙ, zC)
lines!(axvbar, vbarₙ, zC)
lines!(axTbar, Tbarₙ, zC)
lines!(axSbar, Sbarₙ, zC)

lines!(axuw, uwₙ, zF)
lines!(axvw, vwₙ, zF)
lines!(axwT, wTₙ, zF)
lines!(axwS, wSₙ, zF)
lines!(axwb, wbₙ, zF, label="<wb>")
lines!(axwb, wb_nor₀ₙ, zF, label="<wb> no r₀")
lines!(axwb, wb′ₙ, zF, label="g<αwT - βwS>")
axislegend(axwb, position=:rb)

xlims!(axubar, ubarlim)
xlims!(axvbar, vbarlim)
xlims!(axTbar, Tbarlim)
xlims!(axSbar, Sbarlim)

xlims!(axuw, uwlim)
xlims!(axvw, vwlim)
xlims!(axwT, wTlim)
xlims!(axwS, wSlim)
xlims!(axwb, wblim)

trim!(fig.layout)

record(fig, "$(FILE_DIR)/$(FILE_NAME).mp4", 1:Nt, framerate=15) do nn
    n[] = nn
end

@info "Animation completed"
