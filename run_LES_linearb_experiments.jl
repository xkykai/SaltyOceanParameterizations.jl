using Oceananigans
using Oceananigans.Units
using JLD2
using FileIO
using Printf
using CairoMakie
using Oceananigans.Grids: halo_size
using Oceananigans.Operators
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields
using Random
using Statistics
using ArgParse
using Glob

import Dates

function parse_commandline()
    s = ArgParseSettings()
  
    @add_arg_table! s begin
      "--QU"
        help = "surface momentum flux (m²/s²)"
        arg_type = Float64
        default = 0.
      "--QB"
        help = "surface buoyancy flux (m²/s³)"
        arg_type = Float64
        default = 0.
      "--b_surface"
        help = "surface buoyancy (m/s²)"
        arg_type = Float64
        default = 0.
      "--dbdz"
        help = "Initial buoyancy gradient (s⁻²)"
        arg_type = Float64
        default = 5e-3 / 256
      "--f"
        help = "Coriolis parameter (s⁻¹)"
        arg_type = Float64
        default = 1e-4
      "--Nz"
        help = "Number of grid points in z-direction"
        arg_type = Int64
        default = 4
      "--Nx"
        help = "Number of grid points in x-direction"
        arg_type = Int64
        default = 4
      "--Ny"
        help = "Number of grid points in y-direction"
        arg_type = Int64
        default = 4
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
      "--AMD_C2_coefficient"
        help = "Poincare Coefficient for AMD (C²)"
        arg_type = Float64
        default = 1/12
      "--WENO_order"
        help = "Order of WENO scheme"
        arg_type = Int64
        default = 9
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
        default = 2.
      "--time_interval"
        help = "Time interval of output writer (seconds)"
        arg_type = Float64
        default = 10.
      "--field_time_interval"
        help = "Time interval of output writer for fields (seconds)"
        arg_type = Float64
        default = 60.
      "--checkpoint_interval"
        help = "Time interval of checkpoint writer (days)"
        arg_type = Float64
        default = 1.
      "--fps"
        help = "Frames per second of animation"
        arg_type = Int64
        default = 15
      "--pickup"
        help = "Whether to pickup from latest checkpoint"
        arg_type = Bool
        default = true
      "--advection"
        help = "Advection scheme used"
        arg_type = String
        default = "AMD"
      "--file_location"
        help = "Location to save files"
        arg_type = String
        default = "."
    end
    return parse_args(s)
end

args = parse_commandline()

Random.seed!(123)

const Pr = 1

const Lz = args["Lz"]
const Lx = args["Lx"]
const Ly = args["Ly"]

const Nz = args["Nz"]
const Nx = args["Nx"]
const Ny = args["Ny"]

const Qᵁ = args["QU"]
const Qᴮ = args["QB"]

if args["advection"] == "WENOnu1e-5"
    advection = WENO(order=args["WENO_order"])
    const ν, κ = 1e-5, 1e-5/Pr
    closure = ScalarDiffusivity(ν=ν, κ=κ)
    advection_str = "WENO$(args["WENO_order"])nu1e-5"
elseif args["advection"] == "WENOnu0"
    advection = WENO(order=args["WENO_order"])
    const ν, κ = 0, 0
    closure = nothing
    advection_str = "WENO$(args["WENO_order"])nu0"
elseif args["advection"] == "WENOAMD"
    advection = WENO(order=9)
    const ν, κ = 0, 0
    closure = AnisotropicMinimumDissipation(C=args["AMD_C2_coefficient"])
    advection_str = "$(args["advection"])_C2_$(args["AMD_C2_coefficient"])"
elseif args["advection"] == "AMD"
    advection = CenteredSecondOrder()
    const ν, κ = 0, 0
    closure = AnisotropicMinimumDissipation(C=args["AMD_C2_coefficient"])
    advection_str = "$(args["advection"])_C2_$(args["AMD_C2_coefficient"])"
elseif args["advection"] == "AMDC4O"
    advection = CenteredFourthOrder()
    const ν, κ = 0, 0
    closure = AnisotropicMinimumDissipation(C=args["AMD_C2_coefficient"])
    advection_str = "$(args["advection"])_C2_$(args["AMD_C2_coefficient"])"
elseif args["advection"] == "SmagorinskyLilly"
    advection = CenteredSecondOrder()
    closure = SmagorinskyLilly()
    advection_str = args["advection"]
end

const f = args["f"]

const dbdz = args["dbdz"]
const b_surface = args["b_surface"]

const pickup = args["pickup"]

FILE_NAME = "linearb_experiments_dbdz_$(dbdz)_QU_$(Qᵁ)_QB_$(Qᴮ)_b_$(b_surface)_f_$(f)_$(advection_str)_Lxz_$(Lx)_$(Lz)_Nxz_$(Nx)_$(Nz)"
FILE_DIR = "$(args["file_location"])/LES/$(FILE_NAME)"
# FILE_DIR = "/storage6/xinkai/LES/$(FILE_NAME)"
mkpath(FILE_DIR)

size_halo = 5

grid = RectilinearGrid(Oceananigans.GPU(), Float64,
                       size = (Nx, Ny, Nz),
                       halo = (size_halo, size_halo, size_halo),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (-Lz, 0),
                       topology = (Periodic, Periodic, Bounded))

noise(x, y, z) = rand() * exp(z / 8)

b_initial(x, y, z) = dbdz * z + b_surface
b_initial_noisy(x, y, z) = b_initial(x, y, z) + 1e-6 * noise(x, y, z)

b_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᴮ), bottom=GradientBoundaryCondition(dbdz))
u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵁ))

damping_rate = 1 / 5minute
damping_width = Lz / 10

b_target(x, y, z, t) = b_initial(x, y, z)

bottom_mask = GaussianMask{:z}(center=-grid.Lz, width=damping_width)

uvw_sponge = Relaxation(rate=damping_rate, mask=bottom_mask)
b_sponge = Relaxation(rate=damping_rate, mask=bottom_mask, target=b_target)

model = NonhydrostaticModel(; 
            grid = grid,
            closure = closure,
            coriolis = FPlane(f=f),
            buoyancy = BuoyancyTracer(),
            tracers = (:b),
            timestepper = :RungeKutta3,
            advection = advection,
            forcing = (u=uvw_sponge, v=uvw_sponge, w=uvw_sponge, b=b_sponge),
            boundary_conditions = (b=b_bcs, u=u_bcs),
            )

set!(model, b=b_initial_noisy)

b = model.tracers.b
u, v, w = model.velocities

ubar = Field(Average(u, dims=(1, 2)))
vbar = Field(Average(v, dims=(1, 2)))
bbar = Field(Average(b, dims=(1, 2)))

simulation = Simulation(model, Δt=args["dt"]second, stop_time=args["stop_time"]days)

wizard = TimeStepWizard(max_change=1.05, max_Δt=args["max_dt"]second, cfl=0.6)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

wall_clock = [time_ns()]

function print_progress(sim)
    @printf("%s [%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, max(b) %6.3e, next Δt: %s\n",
            Dates.now(),
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            maximum(sim.model.velocities.u),
            maximum(sim.model.velocities.v),
            maximum(sim.model.velocities.w),
            maximum(sim.model.tracers.b),
            prettytime(sim.Δt))

    wall_clock[1] = time_ns()

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(100))

function init_save_some_metadata!(file, model)
    file["metadata/author"] = "Xin Kai Lee"
    file["metadata/coriolis_parameter"] = f
    file["metadata/momentum_flux"] = Qᵁ
    file["metadata/buoyancy_flux"] = Qᴮ
    file["metadata/surface_buoyancy"] = b_surface
    file["metadata/buoyancy_gradient"] = dbdz
    file["metadata/damping_rate"] = damping_rate
    file["metadata/damping_width"] = damping_width
    return nothing
end

uw = Field(Average(w * u, dims=(1, 2)))
vw = Field(Average(w * v, dims=(1, 2)))
wb = Field(Average(w * b, dims=(1, 2)))

νₑ, κₑ = model.diffusivity_fields.νₑ, model.diffusivity_fields.κₑ.b

νₑbar = Field(Average(νₑ, dims=(1, 2)))
κₑbar = Field(Average(κₑ, dims=(1, 2)))

timeseries_outputs = (; ubar, vbar, bbar,
                        uw, vw, wb,
                        νₑbar, κₑbar)

simulation.output_writers[:u] = JLD2OutputWriter(model, (; model.velocities.u),
                                                          filename = "$(FILE_DIR)/instantaneous_fields_u.jld2",
                                                          schedule = TimeInterval(args["field_time_interval"]seconds),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:v] = JLD2OutputWriter(model, (; model.velocities.v),
                                                          filename = "$(FILE_DIR)/instantaneous_fields_v.jld2",
                                                          schedule = TimeInterval(args["field_time_interval"]seconds),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:w] = JLD2OutputWriter(model, (; model.velocities.w),
                                                          filename = "$(FILE_DIR)/instantaneous_fields_w.jld2",
                                                          schedule = TimeInterval(args["field_time_interval"]seconds),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:b] = JLD2OutputWriter(model, (; model.tracers.b),
                                                          filename = "$(FILE_DIR)/instantaneous_fields_b.jld2",
                                                          schedule = TimeInterval(args["field_time_interval"]seconds),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:timeseries] = JLD2OutputWriter(model, timeseries_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_timeseries.jld2",
                                                          schedule = TimeInterval(args["time_interval"]seconds),
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

ubar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "ubar")
vbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "vbar")
bbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar")

uw_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "uw")
vw_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "vw")
wb_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wb")

νₑbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "νₑbar")
κₑbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "κₑbar")

Nt = length(bbar_data.times)

xC = bbar_data.grid.xᶜᵃᵃ[1:Nx]
yC = bbar_data.grid.yᵃᶜᵃ[1:Ny]
zC = bbar_data.grid.zᵃᵃᶜ[1:Nz]

zF = uw_data.grid.zᵃᵃᶠ[1:Nz+1]
##
fig = Figure(size=(1900, 900))

axubar = Axis(fig[1, 1], title="<u>", xlabel="m s⁻¹", ylabel="z")
axvbar = Axis(fig[1, 2], title="<v>", xlabel="m s⁻¹", ylabel="z")
axbbar = Axis(fig[1, 3], title="<b>", xlabel="m s⁻²", ylabel="z")
axdiffusivity = Axis(fig[1, 4], title="<νₑ>, <κₑ>", xlabel="m² s⁻¹", ylabel="z")

axuw = Axis(fig[2, 1], title="uw", xlabel="m² s⁻²", ylabel="z")
axvw = Axis(fig[2, 2], title="vw", xlabel="m² s⁻²", ylabel="z")
axwb = Axis(fig[2, 3], title="wb", xlabel="m² s⁻³", ylabel="z")

function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

ubarlim = (minimum(ubar_data), maximum(ubar_data))
vbarlim = (minimum(vbar_data), maximum(vbar_data))
bbarlim = (minimum(bbar_data), maximum(bbar_data))
diffusivitylim = (find_min(νₑbar_data, κₑbar_data), find_max(νₑbar_data, κₑbar_data))

startframe_lim = 30
uwlim = (minimum(uw_data[1, 1, :, startframe_lim:end]), maximum(uw_data[1, 1, :, startframe_lim:end]))
vwlim = (minimum(vw_data[1, 1, :, startframe_lim:end]), maximum(vw_data[1, 1, :, startframe_lim:end]))
wblim = (minimum(wb_data[1, 1, :, startframe_lim:end]), maximum(wb_data[1, 1, :, startframe_lim:end]))

n = Observable(1)

time_str = @lift "Qᵁ = $(Qᵁ), Qᴮ = $(Qᴮ), Time = $(round(bbar_data.times[$n]/24/60^2, digits=3)) days"
title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)

ubarₙ = @lift interior(ubar_data[$n], 1, 1, :)
vbarₙ = @lift interior(vbar_data[$n], 1, 1, :)
bbarₙ = @lift interior(bbar_data[$n], 1, 1, :)
νₑbarₙ = @lift interior(νₑbar_data[$n], 1, 1, :)
κₑbarₙ = @lift interior(κₑbar_data[$n], 1, 1, :)

uwₙ = @lift interior(uw_data[$n], 1, 1, :)
vwₙ = @lift interior(vw_data[$n], 1, 1, :)
wbₙ = @lift interior(wb_data[$n], 1, 1, :)

lines!(axubar, ubarₙ, zC)
lines!(axvbar, vbarₙ, zC)
lines!(axbbar, bbarₙ, zC)
lines!(axdiffusivity, νₑbarₙ, zC, label="νₑ")
lines!(axdiffusivity, κₑbarₙ, zC, label="κₑ")

lines!(axuw, uwₙ, zF)
lines!(axvw, vwₙ, zF)
lines!(axwb, wbₙ, zF)

xlims!(axubar, ubarlim)
xlims!(axvbar, vbarlim)
xlims!(axbbar, bbarlim)

xlims!(axuw, uwlim)
xlims!(axvw, vwlim)
xlims!(axwb, wblim)

axislegend(axdiffusivity, position=:rt)

trim!(fig.layout)

record(fig, "$(FILE_DIR)/$(FILE_NAME).mp4", 1:Nt, framerate=args["fps"]) do nn
    n[] = nn
end

@info "Animation completed"
#%%