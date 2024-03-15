using Oceananigans
using Oceananigans.Units
using JLD2
using FileIO
using Printf
using Random
using Statistics
import Dates

Random.seed!(123)

const Pr = 1

const Lz = 1
const Lx = 2
const Ly = 2

const Nz = 64
const Nx = 128
const Ny = 128

const Qᵁ = 0
const Qᴮ = 1e-6

const f = 0
const dbdz = 0.08
const b_surface = 0

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

damping_rate = 1 / 5minute
damping_width = Lz / 10

b_target(x, y, z, t) = b_initial(x, y, z)

bottom_mask = GaussianMask{:z}(center=-grid.Lz, width=damping_width)

uvw_sponge = Relaxation(rate=damping_rate, mask=bottom_mask)
b_sponge = Relaxation(rate=damping_rate, mask=bottom_mask, target=b_target)

model = NonhydrostaticModel(; 
            grid = grid,
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

simulation = Simulation(model, Δt=0.1second, stop_time=0.2days)

wizard = TimeStepWizard(max_change=1.05, max_Δt=600second, cfl=0.6)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

wall_clock = [time_ns()]

function print_progress(sim)
    @printf("%s [%05.2f%%] i: %d, t: %s, wall time: %s, next Δt: %s\n",
            Dates.now(),
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            prettytime(sim.Δt))

    wall_clock[1] = time_ns()

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(100))

run!(simulation)