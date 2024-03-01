using Oceananigans
using Oceananigans.Units
using JLD2
using FileIO
using Printf
using CairoMakie
using Oceananigans.Grids: halo_size
using Random
using Statistics
include("correct_reduction_oceananigans.jl")

import Dates

Random.seed!(123)

const Pr = 1

const Lz = 128
const Lx = 64
const Ly = 64

const Nz = 256
const Nx = 128
const Ny = 128

const Qᵁ = -1e-3
const Qᴮ = 1e-7

const f = 1e-4

const dbdz = 5e-3 / 256
const b_surface = 0

FILE_NAME = "linearb_dbdz_$(dbdz)_QU_$(Qᵁ)_QB_$(Qᴮ)_b_$(b_surface)_WENOnu0_Lxz_$(Lx)_$(Lz)_Nxz_$(Nx)_$(Nz)"
FILE_DIR = "./$(FILE_NAME)"
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

b_initial(x, y, z) = dbdz * z + b_surface
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
            closure = ScalarDiffusivity(ν=0, κ=0),
            coriolis = FPlane(f=f),
            buoyancy = BuoyancyTracer(),
            tracers = (:b),
            timestepper = :RungeKutta3,
            advection = WENO(order=9),
            forcing = (u=uvw_sponge, v=uvw_sponge, w=uvw_sponge, b=b_sponge),
            boundary_conditions = (b=b_bcs, u=u_bcs))

set!(model, b=b_initial_noisy)

b = model.tracers.b
u, v, w = model.velocities

ubar = Field(Average(u, dims=(1, 2)))
vbar = Field(Average(v, dims=(1, 2)))
bbar = Field(Average(b, dims=(1, 2)))

simulation = Simulation(model, Δt=0.1second, stop_time=2days)

wizard = TimeStepWizard(max_change=1.05, max_Δt=1minute, cfl=0.6)
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
    file["metadata/parameters/buoyancy_gradient"] = dbdz
    return nothing
end

uw = Field(Average(w * u, dims=(1, 2)))
vw = Field(Average(w * v, dims=(1, 2)))
wb = Field(Average(w * b, dims=(1, 2)))

field_outputs = merge(model.velocities, model.tracers)
timeseries_outputs = (; ubar, vbar, bbar,
                        uw, vw, wb)

simulation.output_writers[:jld2] = JLD2OutputWriter(model, field_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_fields.jld2",
                                                          schedule = TimeInterval(1minute),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!,
                                                          max_filesize=50e9)

simulation.output_writers[:timeseries] = JLD2OutputWriter(model, timeseries_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_timeseries.jld2",
                                                          schedule = TimeInterval(1minute),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=TimeInterval(args["checkpoint_interval"]days), prefix="$(FILE_DIR)/model_checkpoint")

files = readdir(FILE_DIR)
checkpoint_files = files[occursin.("model_checkpoint_iteration", files)]
if !isempty(checkpoint_files)
    checkpoint_iters = parse.(Int, [filename[findfirst("iteration", filename)[end]+1:findfirst(".jld2", filename)[1]-1] for filename in checkpoint_files])
    pickup_iter = maximum(checkpoint_iters)
    run!(simulation, pickup="$(FILE_DIR)/model_checkpoint_iteration$(pickup_iter).jld2")
else
    run!(simulation)
end