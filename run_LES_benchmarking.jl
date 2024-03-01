using Oceananigans
using Oceananigans.Units
using Printf
using Random
using Statistics

import Dates

grid = RectilinearGrid(GPU(), Float64,
                       size = (128, 128, 128),
                       x = (0, 1),
                       y = (0, 1),
                       z = (0, 1))

closure = SmagorinskyLilly()
model = NonhydrostaticModel(; grid, closure)

simulation = Simulation(model, Δt=1second, stop_iteration=1000)

wall_clock = [time_ns()]

function print_progress(sim)
    @printf("%s [%05.2f%%] i: %d, wall time: %s\n",
            Dates.now(),
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(1e-9 * (time_ns() - wall_clock[1])))

    wall_clock[1] = time_ns()

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(100))

run!(simulation)