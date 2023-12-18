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
using Oceananigans.Utils: ConsecutiveIterations
using Random
using Statistics
using ArgParse

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
        default = 2.
      "--time_interval"
        help = "Time interval of output writer (minutes)"
        arg_type = Float64
        default = 10.
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

const Pr = 1

const Lz = args["Lz"]
const Lx = args["Lx"]
const Ly = args["Ly"]

const Nz = args["Nz"]
const Nx = args["Nx"]
const Ny = args["Ny"]

const Qᵁ = args["QU"]
const Qᴮ = args["QB"]

if args["advection"] == "WENO9nu1e-5"
    advection = WENO(order=9)
    const ν, κ = 1e-5, 1e-5/Pr
    closure = ScalarDiffusivity(ν=ν, κ=κ)
elseif args["advection"] == "WENO9nu0"
    advection = WENO(order=9)
    const ν, κ = 0, 0
    closure = nothing
elseif args["advection"] == "WENO9AMD"
    advection = WENO(order=9)
    const ν, κ = 0, 0
    closure = AnisotropicMinimumDissipation()
elseif args["advection"] == "AMD"
    advection = CenteredSecondOrder()
    const ν, κ = 0, 0
    closure = AnisotropicMinimumDissipation()
end

# const Lz = 128
# const Lx = 64
# const Ly = 64

# # const Nz = 64
# # const Nx = 32
# # const Ny = 32

# const Nz = 8
# const Nx = 4
# const Ny = 4

# const Qᵁ = -4e-6
# const Qᴮ = 0

advection = WENO(order=9)
const ν, κ = 1e-5, 1e-5/Pr
closure = ScalarDiffusivity(ν=ν, κ=κ)

const f = args["f"]

const dbdz = args["dbdz"]

const pickup = args["pickup"]

FILE_NAME = "linearb_2layer_turbulencestatistics_dbdz_$(dbdz)_QU_$(Qᵁ)_QB_$(Qᴮ)_$(args["advection"])_Lxz_$(Lx)_$(Lz)_Nxz_$(Nx)_$(Nz)_f"
FILE_DIR = "$(args["file_location"])/LES/$(FILE_NAME)"
# FILE_DIR = "/storage6/xinkai/LES/$(FILE_NAME)"
mkpath(FILE_DIR)

size_halo = 5

grid = RectilinearGrid(GPU(), Float64,
                       size = (Nx, Ny, Nz),
                       halo = (size_halo, size_halo, size_halo),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (-Lz, 0),
                       topology = (Periodic, Periodic, Bounded))

noise(x, y, z) = rand() * exp(z / 8)

const z_surface = -20
const b_surface = dbdz * z_surface

@inline function b_initial(x, y, z)
    if z >= z_surface
        return b_surface
    else
        return dbdz * z
    end
end

b_initial_noisy(x, y, z) = b_initial(x, y, z) + 1e-6 * noise(x, y, z)

b_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᴮ), bottom=GradientBoundaryCondition(dbdz))
u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵁ))

damping_rate = 1/5minute

b_target(x, y, z, t) = b_initial(x, y, z)

bottom_mask = GaussianMask{:z}(center=-grid.Lz, width=grid.Lz/10)

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
            boundary_conditions = (b=b_bcs, u=u_bcs)
            )

set!(model, b=b_initial_noisy)

b = model.tracers.b
u, v, w = model.velocities
w_center = @at (Center, Center, Center) w

ubar = Field(Average(u, dims=(1, 2)))
vbar = Field(Average(v, dims=(1, 2)))
bbar = Field(Average(b, dims=(1, 2)))

simulation = Simulation(model, Δt=args["dt"]second, stop_time=args["stop_time"]days)
# simulation = Simulation(model, Δt=args["dt"]second, stop_time=100minutes)

wizard = TimeStepWizard(max_change=1.05, max_Δt=args["max_dt"]minutes, cfl=0.6)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

wall_clock = [time_ns()]

function print_progress(sim)
    @printf("%s [%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, max(b) %6.3e, next Δt: %s\n",
            Dates.now(),
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            maximum(abs, sim.model.tracers.b),
            prettytime(sim.Δt))

    wall_clock[1] = time_ns()

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(100))

function init_save_some_metadata!(file, model)
    file["metadata/author"] = "Xin Kai Lee"
    file["metadata/parameters/coriolis_parameter"] = f
    file["metadata/parameters/momentum_flux"] = Qᵁ
    file["metadata/parameters/buoyancy_flux"] = Qᴮ
    file["metadata/parameters/surface_buoyancy"] = b_surface
    file["metadata/parameters/mixed_layer_depth"] = z_surface
    file["metadata/parameters/buoyancy_gradient"] = dbdz
    return nothing
end

uw = Field(Average(w * u, dims=(1, 2)))
vw = Field(Average(w * v, dims=(1, 2)))
wb = Field(Average(w * b, dims=(1, 2)))

timeseries_outputs = (; ubar, vbar, bbar,
                        uw, vw, wb)

simulation.output_writers[:u] = JLD2OutputWriter(model, (; model.velocities.u),
                                                          filename = "$(FILE_DIR)/instantaneous_fields_u.jld2",
                                                          schedule = TimeInterval(1hour),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:v] = JLD2OutputWriter(model, (; model.velocities.v),
                                                          filename = "$(FILE_DIR)/instantaneous_fields_v.jld2",
                                                          schedule = TimeInterval(1hour),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!)
                                                          # max_filesize=50e9)

simulation.output_writers[:w] = JLD2OutputWriter(model, (; model.velocities.w),
                                                          filename = "$(FILE_DIR)/instantaneous_fields_w.jld2",
                                                          schedule = TimeInterval(1hour),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!)
                                                          # max_filesize=50e9)

simulation.output_writers[:w_center] = JLD2OutputWriter(model, (; w_center),
                                                          filename = "$(FILE_DIR)/instantaneous_fields_w_center.jld2",
                                                          schedule = TimeInterval(1hour),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!)
                                                          # max_filesize=50e9)

simulation.output_writers[:b] = JLD2OutputWriter(model, (; model.tracers.b),
                                                          filename = "$(FILE_DIR)/instantaneous_fields_b.jld2",
                                                          schedule = TimeInterval(1hour),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:u_consecutive] = JLD2OutputWriter(model, (; model.velocities.u),
                                                          filename = "$(FILE_DIR)/instantaneous_fields_u_consecutive.jld2",
                                                          schedule = ConsecutiveIterations(TimeInterval(1hour)),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:v_consecutive] = JLD2OutputWriter(model, (; model.velocities.v),
                                                          filename = "$(FILE_DIR)/instantaneous_fields_v_consecutive.jld2",
                                                          schedule = ConsecutiveIterations(TimeInterval(1hour)),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:w_consecutive] = JLD2OutputWriter(model, (; model.velocities.w),
                                                          filename = "$(FILE_DIR)/instantaneous_fields_w_consecutive.jld2",
                                                          schedule = ConsecutiveIterations(TimeInterval(1hour)),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:w_center_consecutive] = JLD2OutputWriter(model, (; w_center),
                                                          filename = "$(FILE_DIR)/instantaneous_fields_w_center_consecutive.jld2",
                                                          schedule = ConsecutiveIterations(TimeInterval(1hour)),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:b_consecutive] = JLD2OutputWriter(model, (; model.tracers.b),
                                                          filename = "$(FILE_DIR)/instantaneous_fields_b_consecutive.jld2",
                                                          schedule = ConsecutiveIterations(TimeInterval(1hour)),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:pHY] = JLD2OutputWriter(model, (; model.pressures.pHY′),
                                                          filename = "$(FILE_DIR)/instantaneous_fields_pHY.jld2",
                                                          schedule = TimeInterval(1hour),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!)
                                                          # max_filesize=50e9)

simulation.output_writers[:pNHS] = JLD2OutputWriter(model, (; model.pressures.pNHS),
                                                          filename = "$(FILE_DIR)/instantaneous_fields_pNHS.jld2",
                                                          schedule = TimeInterval(1hour),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!)
                                                          # max_filesize=50e9)

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