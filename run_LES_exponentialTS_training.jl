using Oceananigans
using Oceananigans.Units
using JLD2
using FileIO
using Printf
using CairoMakie
using Oceananigans.Operators
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.BuoyancyModels
using SeawaterPolynomials.TEOS10
using SeawaterPolynomials
using Random
using Statistics
using ArgParse
using LinearAlgebra
using Glob
include("correct_reduction_oceananigans.jl")

import Dates

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
      "--T_bottom"
        help = "bottom temperature (°C)"
        arg_type = Float64
        default = 15.
      "--S_surface"
        help = "surface salinity (g/kg)"
        arg_type = Float64
        default = 35.
      "--S_bottom"
        help = "bottom salinity (g/kg)"
        arg_type = Float64
        default = 33.
      "--dT_lengthscale"
        help = "Temperature lengthscale (m)"
        arg_type = Float64
        default = 64.
      "--dS_lengthscale"
        help = "Salinity lengthscale (m)"
        arg_type = Float64
        default = 64.
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
      "--field_time_interval"
        help = "Time interval of output writer for fields (minutes)"
        arg_type = Float64
        default = 180.
      "--checkpoint_interval"
        help = "Time interval of checkpoint writer (days)"
        arg_type = Float64
        default = 1.
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
      "--file_location"
        help = "Location to save files"
        arg_type = String
        default = "."
    end
    return parse_args(s)
end

args = parse_commandline()

Random.seed!(123)

const Lz = args["Lz"]
const Lx = args["Lx"]
const Ly = args["Ly"]

const Nz = args["Nz"]
const Nx = args["Nx"]
const Ny = args["Ny"]

const Qᵁ = args["QU"]
const Qᵀ = args["QT"]
const Qˢ = args["QS"]

# const Lz = 128
# const Lx = 64
# const Ly = 64

# const Nz = 64
# const Nx = 32
# const Ny = 32

# const Qᵁ = -4e-6
# const Qᵀ = 0
# const Qˢ = 0

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

const dT_lengthscale = args["dT_lengthscale"]
const dS_lengthscale = args["dS_lengthscale"]

const T_surface = args["T_surface"]
const S_surface = args["S_surface"]

const T_bottom = args["T_bottom"]
const S_bottom = args["S_bottom"]

const pickup = args["pickup"]

const eos = TEOS10EquationOfState()

FILE_NAME = "exponentialTS_T_$(T_surface)_$(T_bottom)_S_$(S_surface)_$(S_bottom)_f_$(f)_dTscale_$(dT_lengthscale)_dSscale_$(dS_lengthscale)_QU_$(Qᵁ)_QT_$(Qᵀ)_QS_$(Qˢ)_$(args["advection"])_Lxz_$(Lx)_$(Lz)_Nxz_$(Nx)_$(Nz)"
FILE_DIR = "$(args["file_location"])/LES/$(FILE_NAME)"
mkpath(FILE_DIR)

size_halo = 5

function find_min(a...)
  return minimum(minimum.([a...]))
end

function find_max(a...)
  return maximum(maximum.([a...]))
end

grid = RectilinearGrid(GPU(), Float64,
                       size = (Nx, Ny, Nz),
                       halo = (size_halo, size_halo, size_halo),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (-Lz, 0),
                       topology = (Periodic, Periodic, Bounded))

noise(x, y, z) = rand() * exp(z / 8)

T_initial(x, y, z) = (T_surface - T_bottom) * exp(z / dT_lengthscale) + T_bottom
S_initial(x, y, z) = (S_surface - S_bottom) * exp(z / dS_lengthscale) + S_bottom

T_initial_noisy(x, y, z) = T_initial(x, y, z) + 1e-6 * noise(x, y, z)
S_initial_noisy(x, y, z) = S_initial(x, y, z) + 1e-6 * noise(x, y, z)

T_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵀ))
S_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qˢ))
u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵁ))

damping_rate = 1/15minute

T_target(x, y, z, t) = T_initial(x, y, z)
S_target(x, y, z, t) = S_initial(x, y, z)

bottom_mask = GaussianMask{:z}(center=-grid.Lz, width=grid.Lz/10)

uvw_sponge = Relaxation(rate=damping_rate, mask=bottom_mask)
T_sponge = Relaxation(rate=damping_rate, mask=bottom_mask, target=T_target)
S_sponge = Relaxation(rate=damping_rate, mask=bottom_mask, target=S_target)

model = NonhydrostaticModel(; 
                            grid = grid,
                            closure = closure,
                            coriolis = FPlane(f=f),
                            buoyancy = SeawaterBuoyancy(equation_of_state=eos),
                            tracers = (:T, :S),
                            timestepper = :RungeKutta3,
                            advection = advection,
                            forcing = (u=uvw_sponge, v=uvw_sponge, w=uvw_sponge, T=T_sponge, S=S_sponge),
                            boundary_conditions = (T=T_bcs, S=S_bcs, u=u_bcs))

const ρ₀ = eos.reference_density
const g = model.buoyancy.model.gravitational_acceleration

set!(model, T=T_initial_noisy, S=S_initial_noisy)

T = model.tracers.T
S = model.tracers.S
u, v, w = model.velocities

simulation = Simulation(model, Δt=args["dt"]second, stop_time=args["stop_time"]days)

wizard = TimeStepWizard(max_change=1.05, max_Δt=args["max_dt"]minutes, cfl=0.6)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

wall_clock = [time_ns()]

function print_progress(sim)
    @printf("%s [%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, max(T) %6.3e, max(S) %6.3e, next Δt: %s\n",
            Dates.now(),
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

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(100))

function init_save_some_metadata!(file, model)
    file["metadata/author"] = "Xin Kai Lee"
    file["metadata/coriolis_parameter"] = f
    file["metadata/momentum_flux"] = Qᵁ
    file["metadata/temperature_flux"] = Qᵀ
    file["metadata/salinity_flux"] = Qˢ
    file["metadata/surface_temperature"] = T_surface
    file["metadata/surface_salinity"] = S_surface
    file["metadata/temperature_gradient"] = dTdz
    file["metadata/salinity_gradient"] = dSdz
    file["metadata/equation_of_state"] = eos
    file["metadata/gravitational_acceleration"] = g
    file["metadata/reference_density"] = ρ₀
    return nothing
end

@inline function get_buoyancy(i, j, k, grid, b, C)
  T, S = Oceananigans.BuoyancyModels.get_temperature_and_salinity(b, C)
  @inbounds ρ = TEOS10.ρ(T[i, j, k], S[i, j, k], 0, eos)
  ρ′ = ρ - ρ₀
  return -g * ρ′ / ρ₀
end

@inline function get_density(i, j, k, grid, b, C)
  T, S = Oceananigans.BuoyancyModels.get_temperature_and_salinity(b, C)
  @inbounds ρ = TEOS10.ρ(T[i, j, k], S[i, j, k], 0, eos)
  return ρ
end

b_op = KernelFunctionOperation{Center, Center, Center}(get_buoyancy, model.grid, model.buoyancy, model.tracers)
b = Field(b_op)
compute!(b)

ρ_op = KernelFunctionOperation{Center, Center, Center}(get_density, model.grid, model.buoyancy, model.tracers)
ρ = Field(ρ_op)
compute!(ρ)

ubar = Field(Average(u, dims=(1, 2)))
vbar = Field(Average(v, dims=(1, 2)))
Tbar = Field(Average(T, dims=(1, 2)))
Sbar = Field(Average(S, dims=(1, 2)))
bbar = Field(Average(b, dims=(1, 2)))
ρbar = Field(Average(ρ, dims=(1, 2)))

uw = Field(Average(w * u, dims=(1, 2)))
vw = Field(Average(w * v, dims=(1, 2)))
wb = Field(Average(w * b, dims=(1, 2)))
wT = Field(Average(w * T, dims=(1, 2)))
wS = Field(Average(w * S, dims=(1, 2)))
wρ = Field(Average(w * ρ, dims=(1, 2)))

field_outputs = merge(model.velocities, model.tracers)
timeseries_outputs = (; ubar, vbar, Tbar, Sbar, bbar, ρbar,
                        uw, vw, wT, wS, wb, wρ)

# simulation.output_writers[:u] = JLD2OutputWriter(model, (; u),
#                                                           filename = "$(FILE_DIR)/instantaneous_fields_u.jld2",
#                                                           schedule = TimeInterval(args["field_time_interval"]minutes),
#                                                           with_halos = true,
#                                                           init = init_save_some_metadata!)

# simulation.output_writers[:v] = JLD2OutputWriter(model, (; v),
#                                                           filename = "$(FILE_DIR)/instantaneous_fields_v.jld2",
#                                                           schedule = TimeInterval(args["field_time_interval"]minutes),
#                                                           with_halos = true,
#                                                           init = init_save_some_metadata!)

# simulation.output_writers[:w] = JLD2OutputWriter(model, (; w),
#                                                           filename = "$(FILE_DIR)/instantaneous_fields_w.jld2",
#                                                           schedule = TimeInterval(args["field_time_interval"]minutes),
#                                                           with_halos = true,
#                                                           init = init_save_some_metadata!)

# simulation.output_writers[:T] = JLD2OutputWriter(model, (; T),
#                                                           filename = "$(FILE_DIR)/instantaneous_fields_T.jld2",
#                                                           schedule = TimeInterval(args["field_time_interval"]minutes),
#                                                           with_halos = true,
#                                                           init = init_save_some_metadata!)

# simulation.output_writers[:S] = JLD2OutputWriter(model, (; S),
#                                                           filename = "$(FILE_DIR)/instantaneous_fields_S.jld2",
#                                                           schedule = TimeInterval(args["field_time_interval"]minutes),
#                                                           with_halos = true,
#                                                           init = init_save_some_metadata!)

# simulation.output_writers[:b] = JLD2OutputWriter(model, (; b),
#                                                           filename = "$(FILE_DIR)/instantaneous_fields_b.jld2",
#                                                           schedule = TimeInterval(args["field_time_interval"]minutes),
#                                                           with_halos = true,
#                                                           init = init_save_some_metadata!)

# simulation.output_writers[:ρ] = JLD2OutputWriter(model, (; ρ),
#                                                           filename = "$(FILE_DIR)/instantaneous_fields_rho.jld2",
#                                                           schedule = TimeInterval(args["field_time_interval"]minutes),
#                                                           with_halos = true,
#                                                           init = init_save_some_metadata!)

simulation.output_writers[:timeseries] = JLD2OutputWriter(model, timeseries_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_timeseries.jld2",
                                                          schedule = TimeInterval(args["time_interval"]minutes),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=TimeInterval(args["checkpoint_interval"]days), prefix="$(FILE_DIR)/model_checkpoint")

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

checkpointers = glob("$(FILE_DIR)/model_checkpoint_iteration*.jld2")
if !isempty(checkpointers)
    rm.(checkpointers)
end

#%%
ubar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "ubar")
vbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "vbar")
Tbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "Tbar")
Sbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "Sbar")
bbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar")
ρbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "ρbar")

uw_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "uw")
vw_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "vw")
wT_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wT")
wS_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wS")
wb_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wb")
wρ_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wρ")

xC = Tbar_data.grid.xᶜᵃᵃ[1:Nx]
yC = Tbar_data.grid.yᵃᶜᵃ[1:Ny]
zC = Tbar_data.grid.zᵃᵃᶜ[1:Nz]

zF = uw_data.grid.zᵃᵃᶠ[1:Nz+1]

Nt = length(Tbar_data.times)

##
fig = Figure(size=(2500, 1800))

axTbar = Axis(fig[1:2, 1:2], title="<T>", xlabel="°C", ylabel="z")
axSbar = Axis(fig[1:2, 3:4], title="<S>", xlabel="g kg⁻¹", ylabel="z")
axubar = Axis(fig[1, 5], title="<u>", xlabel="m s⁻¹", ylabel="z")
axvbar = Axis(fig[1, 6], title="<v>", xlabel="m s⁻¹", ylabel="z")
axbbar = Axis(fig[2, 5], title="<b>", xlabel="m s⁻²", ylabel="z")
axρbar = Axis(fig[2, 6], title="<ρ>", xlabel="kg m⁻³", ylabel="z")

axwT = Axis(fig[3:4, 1:2], title="wT", xlabel="m s⁻¹ °C", ylabel="z")
axwS = Axis(fig[3:4, 3:4], title="wS", xlabel="m s⁻¹ g kg⁻¹", ylabel="z")
axuw = Axis(fig[3, 5], title="uw", xlabel="m² s⁻²", ylabel="z")
axvw = Axis(fig[3, 6], title="vw", xlabel="m² s⁻²", ylabel="z")
axwb = Axis(fig[4, 5], title="wb", xlabel="m² s⁻³", ylabel="z")
axwρ = Axis(fig[4, 6], title="wρ", xlabel="kg m⁻² s⁻²", ylabel="z")

ubarlim = (minimum(ubar_data), maximum(ubar_data))
vbarlim = (minimum(vbar_data), maximum(vbar_data))
Tbarlim = (minimum(Tbar_data), maximum(Tbar_data))
Sbarlim = (minimum(Sbar_data), maximum(Sbar_data))
bbarlim = (minimum(bbar_data), maximum(bbar_data))
ρbarlim = (minimum(ρbar_data), maximum(ρbar_data))

startframe_lim = 30
uwlim = (minimum(uw_data[1, 1, :, startframe_lim:end]), maximum(uw_data[1, 1, :, startframe_lim:end]))
vwlim = (minimum(vw_data[1, 1, :, startframe_lim:end]), maximum(vw_data[1, 1, :, startframe_lim:end]))
wTlim = (minimum(wT_data[1, 1, :, startframe_lim:end]), maximum(wT_data[1, 1, :, startframe_lim:end]))
wSlim = (minimum(wS_data[1, 1, :, startframe_lim:end]), maximum(wS_data[1, 1, :, startframe_lim:end]))
wblim = (minimum(wb_data[1, 1, :, startframe_lim:end]), maximum(wb_data[1, 1, :, startframe_lim:end]))
wρlim = (minimum(wρ_data[1, 1, :, startframe_lim:end]), maximum(wρ_data[1, 1, :, startframe_lim:end]))

n = Observable(1)

time_str = @lift "Qᵁ = $(Qᵁ), Qᵀ = $(Qᵀ), Qˢ = $(Qˢ), f = $(f), Time = $(round(Tbar_data.times[$n]/24/60^2, digits=3)) days"
title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)

ubarₙ = @lift interior(ubar_data[$n], 1, 1, :)
vbarₙ = @lift interior(vbar_data[$n], 1, 1, :)
Tbarₙ = @lift interior(Tbar_data[$n], 1, 1, :)
Sbarₙ = @lift interior(Sbar_data[$n], 1, 1, :)
bbarₙ = @lift interior(bbar_data[$n], 1, 1, :)
ρbarₙ = @lift interior(ρbar_data[$n], 1, 1, :)

uwₙ = @lift interior(uw_data[$n], 1, 1, :)
vwₙ = @lift interior(vw_data[$n], 1, 1, :)
wTₙ = @lift interior(wT_data[$n], 1, 1, :)
wSₙ = @lift interior(wS_data[$n], 1, 1, :)
wbₙ = @lift interior(wb_data[$n], 1, 1, :)
wρₙ = @lift interior(wρ_data[$n], 1, 1, :)

lines!(axubar, ubarₙ, zC)
lines!(axvbar, vbarₙ, zC)
lines!(axTbar, Tbarₙ, zC)
lines!(axSbar, Sbarₙ, zC)
lines!(axbbar, bbarₙ, zC)
lines!(axρbar, ρbarₙ, zC)

lines!(axuw, uwₙ, zF)
lines!(axvw, vwₙ, zF)
lines!(axwT, wTₙ, zF)
lines!(axwS, wSₙ, zF)
lines!(axwb, wbₙ, zF)
lines!(axwρ, wρₙ, zF)

xlims!(axubar, ubarlim)
xlims!(axvbar, vbarlim)
xlims!(axTbar, Tbarlim)
xlims!(axSbar, Sbarlim)
xlims!(axbbar, bbarlim)
xlims!(axρbar, ρbarlim)

xlims!(axuw, uwlim)
xlims!(axvw, vwlim)
xlims!(axwT, wTlim)
xlims!(axwS, wSlim)
xlims!(axwb, wblim)
xlims!(axwρ, wρlim)

trim!(fig.layout)
display(fig)

@info "Begin animating..."

CairoMakie.record(fig, "$(FILE_DIR)/$(FILE_NAME)_timeseries.mp4", 1:Nt, framerate=15) do nn
    n[] = nn
end

@info "Timeseries animation completed"
# #%%
# w_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_w.jld2", "w", backend=OnDisk())
# b_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_b.jld2", "b", backend=OnDisk())
# T_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_T.jld2", "T", backend=OnDisk())
# S_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_S.jld2", "S", backend=OnDisk())

# Nt = length(b_data.times)

# xC = T_data.grid.xᶜᵃᵃ[1:Nx]
# yC = T_data.grid.yᵃᶜᵃ[1:Ny]
# zC = T_data.grid.zᵃᵃᶜ[1:Nz]
# zF = T_data.grid.zᵃᵃᶠ[1:Nz+1]

# xCs_xy = xC
# yCs_xy = yC
# zCs_xy = [zC[Nz] for x in xCs_xy, y in yCs_xy]

# yCs_yz = yC
# xCs_yz = range(xC[1], stop=xC[1], length=length(zC))
# zCs_yz = zeros(length(xCs_yz), length(yCs_yz))
# for j in axes(zCs_yz, 2)
#   zCs_yz[:, j] .= zC
# end

# xCs_xz = xC
# yCs_xz = range(yC[1], stop=yC[1], length=length(zC))
# zCs_xz = zeros(length(xCs_xz), length(yCs_xz))
# for i in axes(zCs_xz, 1)
#   zCs_xz[i, :] .= zC
# end

# xFs_xy = xC
# yFs_xy = yC
# # zFs_xy = [zF[Nz+1] for x in xFs_xy, y in yFs_xy]
# zFs_xy = [zF[Nz] for x in xFs_xy, y in yFs_xy]

# yFs_yz = yC
# xFs_yz = range(xC[1], stop=xC[1], length=length(zF))
# zFs_yz = zeros(length(xFs_yz), length(yFs_yz))
# for j in axes(zFs_yz, 2)
#   zFs_yz[:, j] .= zF
# end

# xFs_xz = xC
# yFs_xz = range(yC[1], stop=yC[1], length=length(zF))
# zFs_xz = zeros(length(xFs_xz), length(yFs_xz))
# for i in axes(zFs_xz, 1)
#   zFs_xz[i, :] .= zF
# end
# #%%
# fig = Figure(size=(1800, 1800))

# axw = Axis3(fig[1, 1], title="w", xlabel="x", ylabel="y", zlabel="z", viewmode=:fitzoom, aspect=:data)
# axb = Axis3(fig[1, 2], title="b", xlabel="x", ylabel="y", zlabel="z", viewmode=:fitzoom, aspect=:data)
# axT = Axis3(fig[2, 1], title="T", xlabel="x", ylabel="y", zlabel="z", viewmode=:fitzoom, aspect=:data)
# axS = Axis3(fig[2, 2], title="S", xlabel="x", ylabel="y", zlabel="z", viewmode=:fitzoom, aspect=:data)

# colormap = Reverse(:RdBu_10)

# n = Observable(1)

# # wₙ_xy = @lift interior(w_data[$n], :, :, Nz+1)
# wₙ_xy = @lift interior(w_data[$n], :, :, Nz)
# wₙ_yz = @lift transpose(interior(w_data[$n], 1, :, :))
# wₙ_xz = @lift interior(w_data[$n], :, 1, :)

# bₙ_xy = @lift interior(b_data[$n], :, :, Nz)
# bₙ_yz = @lift transpose(interior(b_data[$n], 1, :, :))
# bₙ_xz = @lift interior(b_data[$n], :, 1, :)

# Tₙ_xy = @lift interior(T_data[$n], :, :, Nz)
# Tₙ_yz = @lift transpose(interior(T_data[$n], 1, :, :))
# Tₙ_xz = @lift interior(T_data[$n], :, 1, :)

# Sₙ_xy = @lift interior(S_data[$n], :, :, Nz)
# Sₙ_yz = @lift transpose(interior(S_data[$n], 1, :, :))
# Sₙ_xz = @lift interior(S_data[$n], :, 1, :)

# wlim = @lift (find_min(interior(w_data[$n], :, :, Nz), interior(w_data[$n], 1, :, :), interior(w_data[$n], :, 1, :)), 
#               find_max(interior(w_data[$n], :, :, Nz), interior(w_data[$n], 1, :, :), interior(w_data[$n], :, 1, :)))
# blim = @lift (find_min(interior(b_data[$n], :, :, Nz), interior(b_data[$n], 1, :, :), interior(b_data[$n], :, 1, :)), 
#               find_max(interior(b_data[$n], :, :, Nz), interior(b_data[$n], 1, :, :), interior(b_data[$n], :, 1, :)))
# Tlim = @lift (find_min(interior(T_data[$n], :, :, Nz), interior(T_data[$n], 1, :, :), interior(T_data[$n], :, 1, :)), 
#               find_max(interior(T_data[$n], :, :, Nz), interior(T_data[$n], 1, :, :), interior(T_data[$n], :, 1, :)))
# Slim = @lift (find_min(interior(S_data[$n], :, :, Nz), interior(S_data[$n], 1, :, :), interior(S_data[$n], :, 1, :)),
#               find_max(interior(S_data[$n], :, :, Nz), interior(S_data[$n], 1, :, :), interior(S_data[$n], :, 1, :)))

# # wlim = (minimum(w_data), maximum(w_data))
# # blim = (minimum(b_data), maximum(b_data))
# # Tlim = (minimum(T_data), maximum(T_data))
# # Slim = (minimum(S_data), maximum(S_data))

# time_str = @lift "Qᵁ = $(Qᵁ), Qᵀ = $(Qᵀ), Qˢ = $(Qˢ), Time = $(round(T_data.times[$n]/24/60^2, digits=3)) days"
# title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)

# w_xy_surface = surface!(axw, xFs_xy, yFs_xy, zFs_xy, color=wₙ_xy, colormap=colormap, colorrange=wlim)
# w_yz_surface = surface!(axw, xFs_yz, yFs_yz, zFs_yz, color=wₙ_yz, colormap=colormap, colorrange=wlim)
# w_xz_surface = surface!(axw, xFs_xz, yFs_xz, zFs_xz, color=wₙ_xz, colormap=colormap, colorrange=wlim)

# # b_xy_surface = surface!(axb, xCs_xy, yCs_xy, zCs_xy, color=bₙ_xy, colormap=colormap, colorrange=blim)
# # b_yz_surface = surface!(axb, xCs_yz, yCs_yz, zCs_yz, color=bₙ_yz, colormap=colormap, colorrange=blim)
# # b_xz_surface = surface!(axb, xCs_xz, yCs_xz, zCs_xz, color=bₙ_xz, colormap=colormap, colorrange=blim)

# # T_xy_surface = surface!(axT, xCs_xy, yCs_xy, zCs_xy, color=Tₙ_xy, colormap=colormap, colorrange=Tlim)
# # T_yz_surface = surface!(axT, xCs_yz, yCs_yz, zCs_yz, color=Tₙ_yz, colormap=colormap, colorrange=Tlim)
# # T_xz_surface = surface!(axT, xCs_xz, yCs_xz, zCs_xz, color=Tₙ_xz, colormap=colormap, colorrange=Tlim)

# # S_xy_surface = surface!(axS, xCs_xy, yCs_xy, zCs_xy, color=Sₙ_xy, colormap=colormap, colorrange=Slim)
# # S_yz_surface = surface!(axS, xCs_yz, yCs_yz, zCs_yz, color=Sₙ_yz, colormap=colormap, colorrange=Slim)
# # S_xz_surface = surface!(axS, xCs_xz, yCs_xz, zCs_xz, color=Sₙ_xz, colormap=colormap, colorrange=Slim)

# trim!(fig.layout)

# record(fig, "$(FILE_DIR)/$(FILE_NAME)_fields.mp4", 1:Nt, framerate=1) do nn
#     n[] = nn
# end

# @info "Animation completed"
# #%%