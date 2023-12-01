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

# advection = WENO(order=9)
# const ν, κ = 1e-5, 1e-5/Pr
# closure = ScalarDiffusivity(ν=ν, κ=κ)

const f = args["f"]

const dbdz = args["dbdz"]
const b_surface = args["b_surface"]

const pickup = args["pickup"]

FILE_NAME = "linearb_turbulencestatistics_dbdz_$(dbdz)_QU_$(Qᵁ)_QB_$(Qᴮ)_b_$(b_surface)_$(args["advection"])_Lxz_$(Lx)_$(Lz)_Nxz_$(Nx)_$(Nz)_f"
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

ubar = Field(Average(u, dims=(1, 2)))
vbar = Field(Average(v, dims=(1, 2)))
bbar = Field(Average(b, dims=(1, 2)))

u′ = u - ubar
v′ = v - vbar
b′ = b - bbar
w′ = w

u′w′ = Field(Average(w′ * u′, dims=(1, 2)))
v′w′ = Field(Average(w′ * v′, dims=(1, 2)))
w′b′ = Field(Average(w′ * b′, dims=(1, 2)))
w′² = Field(Average(w′^2, dims=(1, 2)))

u′²w′ = Field(Average(w′ * u′^2, dims=(1, 2)))
v′²w′ = Field(Average(w′ * v′^2, dims=(1, 2)))
b′²w′ = Field(Average(w′ * b′^2, dims=(1, 2)))
w′³ = Field(Average(w′^3, dims=(1, 2)))

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
    file["metadata/parameters/buoyancy_gradient"] = dbdz
    return nothing
end

uw = Field(Average(w * u, dims=(1, 2)))
vw = Field(Average(w * v, dims=(1, 2)))
wb = Field(Average(w * b, dims=(1, 2)))

# Calculate advection terms
@inline function get_u_udot∇u(i, j, k, grid, u, v, w)
  @inbounds begin
    advection = ℑxᶜᵃᵃ(i, j, k, grid, u) * ∂xᶜᶜᶜ(i, j, k, grid, u) + ℑyᵃᶜᵃ(i, j, k, grid, v) * ℑxyᶜᶜᵃ(i, j, k, grid, ∂yᶠᶠᶜ, u) + ℑzᵃᵃᶜ(i, j, k, grid, w) * ℑxzᶜᵃᶜ(i, j, k, grid, ∂zᶠᶜᶠ, u)
    return ℑxᶜᵃᵃ(i, j, k, grid, u) * advection
  end
end

@inline function get_v_udot∇v(i, j, k, grid, u, v, w)
  @inbounds begin
    advection = ℑxᶜᵃᵃ(i, j, k, grid, u) * ℑxyᶜᶜᵃ(i, j, k, grid, ∂xᶠᶠᶜ, v) + ℑyᵃᶜᵃ(i, j, k, grid, v) * ∂yᶜᶜᶜ(i, j, k, grid, v) + ℑzᵃᵃᶜ(i, j, k, grid, w) * ℑyzᵃᶜᶜ(i, j, k, grid, ∂zᶜᶠᶠ, v)
    return ℑyᵃᶜᵃ(i, j, k, grid, v) * advection
  end
end

@inline function get_w_udot∇w(i, j, k, grid, u, v, w)
  @inbounds begin
    advection = ℑxᶜᵃᵃ(i, j, k, grid, u) * ℑxzᶜᵃᶜ(i, j, k, grid, ∂xᶠᶜᶠ, w) + ℑyᵃᶜᵃ(i, j, k, grid, v) * ℑyzᵃᶜᶜ(i, j, k, grid, ∂yᶜᶠᶠ, w) + ℑzᵃᵃᶜ(i, j, k, grid, w) * ∂zᶜᶜᶜ(i, j, k, grid, w)
    return ℑzᵃᵃᶜ(i, j, k, grid, w) * advection
  end
end

@inline function get_b_udot∇b(i, j, k, grid, u, v, w, b)
  @inbounds begin
    advection = ℑxᶜᵃᵃ(i, j, k, grid, u) * ℑxᶜᵃᵃ(i, j, k, grid, ∂xᶠᶜᶜ, b) + ℑyᵃᶜᵃ(i, j, k, grid, v) * ℑyᵃᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, b) + ℑzᵃᵃᶜ(i, j, k, grid, w) * ℑzᵃᵃᶜ(i, j, k, grid, ∂zᶜᶜᶠ, b)
    return b[i, j, k] * advection
  end
end

u_udot∇u_op = KernelFunctionOperation{Center, Center, Center}(get_u_udot∇u, grid, u, v, w)
v_udot∇v_op = KernelFunctionOperation{Center, Center, Center}(get_v_udot∇v, grid, u, v, w)
b_udot∇b_op = KernelFunctionOperation{Center, Center, Center}(get_b_udot∇b, grid, u, v, w, b)
w_udot∇w_op = KernelFunctionOperation{Center, Center, Center}(get_w_udot∇w, grid, u, v, w)

u_udot∇u = Field(Average(Field(u_udot∇u_op), dims=(1, 2)))
v_udot∇v = Field(Average(Field(v_udot∇v_op), dims=(1, 2)))
b_udot∇b = Field(Average(Field(b_udot∇b_op), dims=(1, 2)))
w_udot∇w = Field(Average(Field(w_udot∇w_op), dims=(1, 2)))

u∂u′w′∂z = ubar * ∂z(u′w′)
v∂v′w′∂z = vbar * ∂z(v′w′)
b∂w′b′∂z = bbar * ∂z(w′b′)

u′w′∂u∂z = @at (Nothing, Nothing, Center) u′w′ * ∂z(ubar)
v′w′∂v∂z = @at (Nothing, Nothing, Center) v′w′ * ∂z(vbar)
w′b′∂b∂z = @at (Nothing, Nothing, Center) w′b′ * ∂z(bbar)

∂u′²w′∂z = ∂z(u′²w′)
∂v′²w′∂z = ∂z(v′²w′)
∂b′²w′∂z = ∂z(b′²w′)
∂w′³∂z = ∂z(w′³)

u_udot∇u′ = Field(u∂u′w′∂z + u′w′∂u∂z + 0.5 * ∂u′²w′∂z)
v_udot∇v′ = Field(v∂v′w′∂z + v′w′∂v∂z + 0.5 * ∂v′²w′∂z)
b_udot∇b′ = Field(b∂w′b′∂z + w′b′∂b∂z + 0.5 * ∂b′²w′∂z)
w_udot∇w′ = Field(0.5*∂w′³∂z)

# Calculate dissipation terms
if closure == AnisotropicMinimumDissipation()
  νₑ, κₑ = model.diffusivity_fields.νₑ, model.diffusivity_fields.κₑ.b
else
  νₑ, κₑ = ν, κ
end

u_∂xνₑ∂u∂x = Field(Average(u * ∂x(∂x(u) * νₑ), dims=(1, 2)))
u_∂yνₑ∂u∂y = Field(Average(u * ∂y(∂y(u) * νₑ), dims=(1, 2)))
u_∂zνₑ∂u∂z = Field(Average(u * ∂z(∂z(u) * νₑ), dims=(1, 2)))

v_∂xνₑ∂v∂x = Field(Average(v * ∂x(∂x(v) * νₑ), dims=(1, 2)))
v_∂yνₑ∂v∂y = Field(Average(v * ∂y(∂y(v) * νₑ), dims=(1, 2)))
v_∂zνₑ∂v∂z = Field(Average(v * ∂z(∂z(v) * νₑ), dims=(1, 2)))

b_∂xκₑ∂b∂x = Field(Average(b * ∂x(∂x(b) * κₑ), dims=(1, 2)))
b_∂yκₑ∂b∂y = Field(Average(b * ∂y(∂y(b) * κₑ), dims=(1, 2)))
b_∂zκₑ∂b∂z = Field(Average(b * ∂z(∂z(b) * κₑ), dims=(1, 2)))

w_∂xνₑ∂w∂x = Field(Average(@at((Center, Center, Center), w * ∂x(∂x(w) * νₑ)), dims=(1, 2)))
w_∂yνₑ∂w∂y = Field(Average(@at((Center, Center, Center), w * ∂y(∂y(w) * νₑ)), dims=(1, 2)))
w_∂zνₑ∂w∂z = Field(Average(@at((Center, Center, Center), w * ∂z(∂z(w) * νₑ)), dims=(1, 2)))

u_∇dotνₑ∇u = Field(u_∂xνₑ∂u∂x + u_∂yνₑ∂u∂y + u_∂zνₑ∂u∂z)
v_∇dotνₑ∇v = Field(v_∂xνₑ∂v∂x + v_∂yνₑ∂v∂y + v_∂zνₑ∂v∂z)
b_∇dotκₑ∇b = Field(b_∂xκₑ∂b∂x + b_∂yκₑ∂b∂y + b_∂zκₑ∂b∂z)
w_∇dotνₑ∇w = Field(w_∂xνₑ∂w∂x + w_∂yνₑ∂w∂y + w_∂zνₑ∂w∂z)

∂zνₑ∂u²∂z = Field(Average(∂z(∂z(u^2) * νₑ), dims=(1, 2)))
∂zνₑ∂v²∂z = Field(Average(∂z(∂z(v^2) * νₑ), dims=(1, 2)))
∂zκₑ∂b²∂z = Field(Average(∂z(∂z(b^2) * κₑ), dims=(1, 2)))
∂zνₑ∂w²∂z = @at (Nothing, Nothing, Center) Field(Average(∂z(∂z(w^2) * νₑ), dims=(1, 2)))

νₑ∇u∇u = Field(Average(Field((∂x(u)^2 + ∂y(u)^2 + ∂z(u)^2) * νₑ), dims=(1, 2)))
νₑ∇v∇v = Field(Average(Field((∂x(v)^2 + ∂y(v)^2 + ∂z(v)^2) * νₑ), dims=(1, 2)))
κₑ∇b∇b = Field(Average(Field((∂x(b)^2 + ∂y(b)^2 + ∂z(b)^2) * κₑ), dims=(1, 2)))
νₑ∇w∇w = Field(Average(Field((∂x(w)^2 + ∂y(w)^2 + ∂z(w)^2) * νₑ), dims=(1, 2)))

u_∇dotνₑ∇u′ = Field(0.5 * ∂zνₑ∂u²∂z - νₑ∇u∇u)
v_∇dotνₑ∇v′ = Field(0.5 * ∂zνₑ∂v²∂z - νₑ∇v∇v)
b_∇dotκₑ∇b′ = Field(0.5 * ∂zκₑ∂b²∂z - κₑ∇b∇b)
w_∇dotνₑ∇w′ = Field(@at (Nothing, Nothing, Center) 0.5 * ∂zνₑ∂w²∂z - νₑ∇w∇w)

# Calculate pressure terms
pNHS, pHY′ = model.pressures.pNHS, model.pressures.pHY′

u∂p∂x = Field(Average(u * ∂x(pNHS + pHY′), dims=(1, 2)))
v∂p∂y = Field(Average(v * ∂y(pNHS + pHY′), dims=(1, 2)))
w∂p∂z = Field(Average(w * ∂z(pNHS), dims=(1, 2)))

fuv = Field(Average(f * u * v, dims=(1, 2)))

# Calculate variance terms
# @inline function get_∂u²∂t_nodissipation(i, j, k, grid, u, v, w, pressures)
#   @inbounds begin
#     pressure = ∂xᶜᶜᶜ(i, j, k, grid, pressures.pNHS) + ∂xᶜᶜᶜ(i, j, k, grid, pressures.pHY′)
#     advection = u[i, j, k] * ∂xᶜᶜᶜ(i, j, k, grid, u) + v[i, j, k] * ∂yᶜᶜᶜ(i, j, k, grid, u) + w[i, j, k] * ∂zᶜᶜᶜ(i, j, k, grid, u)
#     coriolis = f * v[i, j, k]
#     return 2 * u[i, j, k] * (-advection - pressure + coriolis)
#   end
# end

# @inline function get_∂v²∂t_nodissipation(i, j, k, grid, u, v, w, pressures)
#   @inbounds begin
#     pressure = ∂yᶜᶜᶜ(i, j, k, grid, pressures.pNHS) + ∂yᶜᶜᶜ(i, j, k, grid, pressures.pHY′)
#     advection = u[i, j, k] * ∂xᶜᶜᶜ(i, j, k, grid, v) + v[i, j, k] * ∂yᶜᶜᶜ(i, j, k, grid, v) + w[i, j, k] * ∂zᶜᶜᶜ(i, j, k, grid, v)
#     coriolis = f * u[i, j, k]
#     return 2 * v[i, j, k] * (-advection - pressure - coriolis)
#   end
# end

# @inline function get_∂w²∂t_nodissipation(i, j, k, grid, u, v, w, pressures, b)
#   @inbounds begin
#     pressure = ∂zᶜᶜᶜ(i, j, k, grid, pressures.pNHS)
#     advection = u[i, j, k] * ∂xᶜᶜᶜ(i, j, k, grid, w) + v[i, j, k] * ∂yᶜᶜᶜ(i, j, k, grid, w) + w[i, j, k] * ∂zᶜᶜᶜ(i, j, k, grid, w)
#     buoyancy = b[i, j, k]
#     return 2 * w[i, j, k] * (-advection - pressure + buoyancy)
#   end
# end

# @inline function get_∂b²∂t_nodissipation(i, j, k, grid, u, v, w, b)
#   @inbounds begin
#     advection = u[i, j, k] * ∂xᶜᶜᶜ(i, j, k, grid, b) + v[i, j, k] * ∂yᶜᶜᶜ(i, j, k, grid, b) + w[i, j, k] * ∂zᶜᶜᶜ(i, j, k, grid, b)
#     return 2 * b[i, j, k] * -advection
#   end
# end

# ∂u²∂t_nodissipation_op = KernelFunctionOperation{Center, Center, Center}(get_∂u²∂t_nodissipation, grid, u, v, w, model.pressures)
# ∂v²∂t_nodissipation_op = KernelFunctionOperation{Center, Center, Center}(get_∂v²∂t_nodissipation, grid, u, v, w, model.pressures)
# ∂w²∂t_nodissipation_op = KernelFunctionOperation{Center, Center, Center}(get_∂w²∂t_nodissipation, grid, u, v, w, model.pressures, b)
# ∂b²∂t_nodissipation_op = KernelFunctionOperation{Center, Center, Center}(get_∂b²∂t_nodissipation, grid, u, v, w, b)
# ∂u²∂t_nodissipation = Field(Average(Field(∂u²∂t_nodissipation_op), dims=(1, 2)))
# ∂v²∂t_nodissipation = Field(Average(Field(∂v²∂t_nodissipation_op), dims=(1, 2)))
# ∂b²∂t_nodissipation = Field(Average(Field(∂b²∂t_nodissipation_op), dims=(1, 2)))
# ∂w²∂t_nodissipation = Field(Average(Field(∂w²∂t_nodissipation_op), dims=(1, 2)))

# ∂u²∂t = ∂u²∂t_nodissipation + u_∇dotνₑ∇u
# ∂v²∂t = ∂v²∂t_nodissipation + v_∇dotνₑ∇v
# ∂b²∂t = ∂b²∂t_nodissipation + b_∇dotκₑ∇b
# ∂w²∂t = ∂w²∂t_nodissipation + w_∇dotνₑ∇w

∂u²∂t = Field(2 * (-u_udot∇u + u_∇dotνₑ∇u + fuv - u∂p∂x))
∂v²∂t = Field(2 * (-v_udot∇v + v_∇dotνₑ∇v - fuv - v∂p∂y))
∂b²∂t = Field(2 * (-b_udot∇b + b_∇dotκₑ∇b))
∂w²∂t = Field(2 * (-w_udot∇w + w_∇dotνₑ∇w - w∂p∂z + wb))

field_outputs = merge(model.velocities, model.tracers, model.pressures)
timeseries_outputs = (; ubar, vbar, bbar,
                        uw, vw, wb,
                        u∂u′w′∂z, v∂v′w′∂z, b∂w′b′∂z,
                        u′w′∂u∂z, v′w′∂v∂z, w′b′∂b∂z,
                        ∂u′²w′∂z, ∂v′²w′∂z, ∂b′²w′∂z, ∂w′³∂z,
                        u_udot∇u, v_udot∇v, b_udot∇b, w_udot∇w,
                        u_udot∇u′, v_udot∇v′, b_udot∇b′, w_udot∇w′,
                        ∂zνₑ∂u²∂z, ∂zνₑ∂v²∂z, ∂zκₑ∂b²∂z, ∂zνₑ∂w²∂z,
                        νₑ∇u∇u, νₑ∇v∇v, κₑ∇b∇b, νₑ∇w∇w,
                        u_∇dotνₑ∇u, v_∇dotνₑ∇v, b_∇dotκₑ∇b, w_∇dotνₑ∇w,
                        u_∇dotνₑ∇u′, v_∇dotνₑ∇v′, b_∇dotκₑ∇b′, w_∇dotνₑ∇w′,
                        ∂u²∂t, ∂v²∂t, ∂b²∂t, ∂w²∂t)

                        # uw, vw, wb,
                        # ∂u²∂t, ∂v²∂t, ∂w²∂t, ∂b²∂t,
                        # ∂u²∂t′, ∂v²∂t′, ∂b²∂t′, ∂w²∂t′,
                        # u∂u′w′∂z, v∂v′w′∂z, b∂w′b′∂z,
                        # u′w′∂u∂z, v′w′∂v∂z, w′b′∂b∂z,
                        # ∂u′²w′∂z, ∂v′²w′∂z, ∂b′²w′∂z, ∂w′³∂z,
                        # u_udot∇u, v_udot∇v, b_udot∇b, w_udot∇w,
                        # ∂zνₑ∂u²∂z, ∂zνₑ∂v²∂z, ∂zκₑ∂b²∂z, ∂zνₑ∂w²∂z,
                        # νₑ∇u∇u, νₑ∇v∇v, κₑ∇b∇b, νₑ∇w∇w,
                        # u_∇dotνₑ∇u, v_∇dotνₑ∇v, b_∇dotκₑ∇b, w_∇dotνₑ∇w,
                        # u′w′, v′w′, w′b′, w′²,
                        # ∂p∂x, ∂p∂y, ∂p∂z)

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

simulation.output_writers[:b] = JLD2OutputWriter(model, (; model.tracers.b),
                                                          filename = "$(FILE_DIR)/instantaneous_fields_b.jld2",
                                                          schedule = TimeInterval(1hour),
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

ubar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "ubar")
vbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "vbar")
bbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar")

uw_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "uw")
vw_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "vw")
wb_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wb")

u_udot∇u_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "u_udot∇u")
v_udot∇v_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "v_udot∇v")
b_udot∇b_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "b_udot∇b")
w_udot∇w_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "w_udot∇w")

u_udot∇u′_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "u_udot∇u′")
v_udot∇v′_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "v_udot∇v′")
b_udot∇b′_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "b_udot∇b′")
w_udot∇w′_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "w_udot∇w′")

u_∇dotνₑ∇u_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "u_∇dotνₑ∇u")
v_∇dotνₑ∇v_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "v_∇dotνₑ∇v")
b_∇dotκₑ∇b_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "b_∇dotκₑ∇b")
w_∇dotνₑ∇w_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "w_∇dotνₑ∇w")

u_∇dotνₑ∇u′_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "u_∇dotνₑ∇u′")
v_∇dotνₑ∇v′_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "v_∇dotνₑ∇v′")
b_∇dotκₑ∇b′_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "b_∇dotκₑ∇b′")
w_∇dotνₑ∇w′_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "w_∇dotνₑ∇w′")

∂u²∂t_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "∂u²∂t")
∂v²∂t_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "∂v²∂t")
∂b²∂t_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "∂b²∂t")
∂w²∂t_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "∂w²∂t")

Nt = length(bbar_data.times)

xC = bbar_data.grid.xᶜᵃᵃ[1:Nx]
yC = bbar_data.grid.yᵃᶜᵃ[1:Ny]
zC = bbar_data.grid.zᵃᵃᶜ[1:Nz]

zF = uw_data.grid.zᵃᵃᶠ[1:Nz+1]
##
fig = Figure(resolution=(2100, 2100))

axubar = Axis(fig[1, 1], title="<u>", xlabel="m s⁻¹", ylabel="z")
axvbar = Axis(fig[1, 2], title="<v>", xlabel="m s⁻¹", ylabel="z")
axbbar = Axis(fig[1, 3], title="<b>", xlabel="m s⁻²", ylabel="z")

axuw = Axis(fig[2, 1], title="uw", xlabel="m² s⁻²", ylabel="z")
axvw = Axis(fig[2, 2], title="vw", xlabel="m² s⁻²", ylabel="z")
axwb = Axis(fig[2, 3], title="wb", xlabel="m² s⁻³", ylabel="z")

axuadvection = Axis(fig[3, 1], title="u advection", ylabel="z")
axvadvection = Axis(fig[3, 3], title="v advection", ylabel="z")
axbadvection = Axis(fig[1, 4], title="b advection", ylabel="z")
axwadvection = Axis(fig[4, 1], title="w advection", ylabel="z")

axudissipation = Axis(fig[3, 2], title="u dissipation", ylabel="z")
axvdissipation = Axis(fig[3, 4], title="v dissipation", ylabel="z")
axbdissipation = Axis(fig[2, 4], title="b dissipation", ylabel="z")
axwdissipation = Axis(fig[4, 2], title="w dissipation", ylabel="z")

ax∂TKE∂t = Axis(fig[4, 3], title="0.5 ∂<u² + v² + w²>/∂t", ylabel="z")
ax∂b²∂t = Axis(fig[4, 4], title="∂<b²>/∂t", ylabel="z")

function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

ubarlim = (minimum(ubar_data), maximum(ubar_data))
vbarlim = (minimum(vbar_data), maximum(vbar_data))
bbarlim = (minimum(bbar_data), maximum(bbar_data))

startframe_lim = 30
uwlim = (minimum(uw_data[1, 1, :, startframe_lim:end]), maximum(uw_data[1, 1, :, startframe_lim:end]))
vwlim = (minimum(vw_data[1, 1, :, startframe_lim:end]), maximum(vw_data[1, 1, :, startframe_lim:end]))
wblim = (minimum(wb_data[1, 1, :, startframe_lim:end]), maximum(wb_data[1, 1, :, startframe_lim:end]))

u_udot∇ulim = (find_min(u_udot∇u_data, u_udot∇u′_data), find_max(u_udot∇u_data, u_udot∇u′_data))
v_udot∇vlim = (find_min(v_udot∇v_data, v_udot∇v′_data), find_max(v_udot∇v_data, v_udot∇v′_data))
b_udot∇blim = (find_min(b_udot∇b_data, b_udot∇b′_data), find_max(b_udot∇b_data, b_udot∇b′_data))
w_udot∇wlim = (find_min(w_udot∇w_data, w_udot∇w′_data), find_max(w_udot∇w_data, w_udot∇w′_data))

u_∇dotνₑ∇ulim = (find_min(u_∇dotνₑ∇u_data, u_∇dotνₑ∇u′_data), find_max(u_∇dotνₑ∇u_data, u_∇dotνₑ∇u′_data))
v_∇dotνₑ∇vlim = (find_min(v_∇dotνₑ∇v_data, v_∇dotνₑ∇v′_data), find_max(v_∇dotνₑ∇v_data, v_∇dotνₑ∇v′_data))
b_∇dotκₑ∇blim = (find_min(b_∇dotκₑ∇b_data, b_∇dotκₑ∇b′_data), find_max(b_∇dotκₑ∇b_data, b_∇dotκₑ∇b′_data))
w_∇dotνₑ∇wlim = (find_min(w_∇dotνₑ∇w_data, w_∇dotνₑ∇w′_data), find_max(w_∇dotνₑ∇w_data, w_∇dotνₑ∇w′_data))

∂TKE∂tlim = (minimum(0.5 .* (∂u²∂t_data .+ ∂v²∂t_data .+ ∂w²∂t_data)), 
             maximum(0.5 .* (∂u²∂t_data .+ ∂v²∂t_data .+ ∂w²∂t_data)))

∂b²∂tlim = (minimum(∂b²∂t_data), maximum(∂b²∂t_data))

n = Observable(1)

time_str = @lift "Qᵁ = $(Qᵁ), Qᴮ = $(Qᴮ), Time = $(round(bbar_data.times[$n]/24/60^2, digits=3)) days"
title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)

ubarₙ = @lift interior(ubar_data[$n], 1, 1, :)
vbarₙ = @lift interior(vbar_data[$n], 1, 1, :)
bbarₙ = @lift interior(bbar_data[$n], 1, 1, :)

uwₙ = @lift interior(uw_data[$n], 1, 1, :)
vwₙ = @lift interior(vw_data[$n], 1, 1, :)
wbₙ = @lift interior(wb_data[$n], 1, 1, :)

u_udot∇uₙ = @lift interior(u_udot∇u_data[$n], 1, 1, :)
v_udot∇vₙ = @lift interior(v_udot∇v_data[$n], 1, 1, :)
b_udot∇bₙ = @lift interior(b_udot∇b_data[$n], 1, 1, :)
w_udot∇wₙ = @lift interior(w_udot∇w_data[$n], 1, 1, :)

u_udot∇u′ₙ = @lift interior(u_udot∇u′_data[$n], 1, 1, :)
v_udot∇v′ₙ = @lift interior(v_udot∇v′_data[$n], 1, 1, :)
b_udot∇b′ₙ = @lift interior(b_udot∇b′_data[$n], 1, 1, :)
w_udot∇w′ₙ = @lift interior(w_udot∇w′_data[$n], 1, 1, :)

u_∇dotνₑ∇uₙ = @lift interior(u_∇dotνₑ∇u_data[$n], 1, 1, :)
v_∇dotνₑ∇vₙ = @lift interior(v_∇dotνₑ∇v_data[$n], 1, 1, :)
b_∇dotκₑ∇bₙ = @lift interior(b_∇dotκₑ∇b_data[$n], 1, 1, :)
w_∇dotνₑ∇wₙ = @lift interior(w_∇dotνₑ∇w_data[$n], 1, 1, :)

u_∇dotνₑ∇u′ₙ = @lift interior(u_∇dotνₑ∇u′_data[$n], 1, 1, :)
v_∇dotνₑ∇v′ₙ = @lift interior(v_∇dotνₑ∇v′_data[$n], 1, 1, :)
b_∇dotκₑ∇b′ₙ = @lift interior(b_∇dotκₑ∇b′_data[$n], 1, 1, :)
w_∇dotνₑ∇w′ₙ = @lift interior(w_∇dotνₑ∇w′_data[$n], 1, 1, :)

∂TKE∂tₙ = @lift 0.5 .* (interior(∂u²∂t_data[$n], 1, 1, :) .+ interior(∂v²∂t_data[$n], 1, 1, :) .+ interior(∂w²∂t_data[$n], 1, 1, :))
∂b²∂tₙ = @lift interior(∂b²∂t_data[$n], 1, 1, :)

lines!(axubar, ubarₙ, zC)
lines!(axvbar, vbarₙ, zC)
lines!(axbbar, bbarₙ, zC)

lines!(axuw, uwₙ, zF)
lines!(axvw, vwₙ, zF)
lines!(axwb, wbₙ, zF)

lines!(axuadvection, u_udot∇uₙ, zC, label="Total")
lines!(axuadvection, u_udot∇u′ₙ, zC, label="Sum of components")
axislegend(axuadvection, position=:rb)

lines!(axvadvection, v_udot∇vₙ, zC, label="Total")
lines!(axvadvection, v_udot∇v′ₙ, zC, label="Sum of components")
axislegend(axvadvection, position=:rb)

lines!(axbadvection, b_udot∇bₙ, zC, label="Total")
lines!(axbadvection, b_udot∇b′ₙ, zC, label="Sum of components")
axislegend(axbadvection, position=:rb)

lines!(axwadvection, w_udot∇wₙ, zC, label="Total")
lines!(axwadvection, w_udot∇w′ₙ, zC, label="Sum of components")
axislegend(axwadvection, position=:rb)

lines!(axudissipation, u_∇dotνₑ∇uₙ, zC, label="Total")
lines!(axudissipation, u_∇dotνₑ∇u′ₙ, zC, label="Sum of components")
axislegend(axudissipation, position=:rb)

lines!(axvdissipation, v_∇dotνₑ∇vₙ, zC, label="Total")
lines!(axvdissipation, v_∇dotνₑ∇v′ₙ, zC, label="Sum of components")
axislegend(axvdissipation, position=:rb)

lines!(axbdissipation, b_∇dotκₑ∇bₙ, zC, label="Total")
lines!(axbdissipation, b_∇dotκₑ∇b′ₙ, zC, label="Sum of components")
axislegend(axbdissipation, position=:rb)

lines!(axwdissipation, w_∇dotνₑ∇wₙ, zC, label="Total")
lines!(axwdissipation, w_∇dotνₑ∇w′ₙ, zC, label="Sum of components")
axislegend(axwdissipation, position=:rb)

lines!(ax∂TKE∂t, ∂TKE∂tₙ, zC)

lines!(ax∂b²∂t, ∂b²∂tₙ, zC)

xlims!(axubar, ubarlim)
xlims!(axvbar, vbarlim)
xlims!(axbbar, bbarlim)

xlims!(axuw, uwlim)
xlims!(axvw, vwlim)
xlims!(axwb, wblim)

xlims!(axuadvection, u_udot∇ulim)
xlims!(axvadvection, v_udot∇vlim)
xlims!(axbadvection, b_udot∇blim)
xlims!(axwadvection, w_udot∇wlim)

u_∇dotνₑ∇ulim != (0, 0) && xlims!(axudissipation, u_∇dotνₑ∇ulim)
v_∇dotνₑ∇vlim != (0, 0) && xlims!(axvdissipation, v_∇dotνₑ∇vlim)
b_∇dotκₑ∇blim != (0, 0) && xlims!(axbdissipation, b_∇dotκₑ∇blim)
w_∇dotνₑ∇wlim != (0, 0) && xlims!(axwdissipation, w_∇dotνₑ∇wlim)

xlims!(ax∂TKE∂t, ∂TKE∂tlim)
xlims!(ax∂b²∂t, ∂b²∂tlim)

trim!(fig.layout)
display(fig)

record(fig, "$(FILE_DIR)/$(FILE_NAME).mp4", 1:Nt, framerate=15) do nn
    n[] = nn
end

@info "Animation completed"
#%%
#=
b_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "b", backend=OnDisk())
w_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "w", backend=OnDisk())

ubar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "ubar")
vbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "vbar")
bbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar")

uw_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "uw")
vw_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "vw")
wb_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wb")

∂u²∂t_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "∂u²∂t")
∂v²∂t_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "∂v²∂t")
∂w²∂t_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "∂w²∂t")
∂b²∂t_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "∂b²∂t")

∂u²∂t′_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "∂u²∂t′")
∂v²∂t′_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "∂v²∂t′")
∂w²∂t′_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "∂w²∂t′")
∂b²∂t′_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "∂b²∂t′")

b_∇dotκₑ∇b_2_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "b_∇dotκₑ∇b_2")

Nt = length(bbar_data.times)

xC = bbar_data.grid.xᶜᵃᵃ[1:Nx]
yC = bbar_data.grid.yᵃᶜᵃ[1:Ny]
zC = bbar_data.grid.zᵃᵃᶜ[1:Nz]

zF = uw_data.grid.zᵃᵃᶠ[1:Nz+1]
##
fig = Figure(resolution=(1800, 1800))

axb = Axis3(fig[1:2, 1:2], title="b", xlabel="x", ylabel="y", zlabel="z", viewmode=:fitzoom, aspect=:data)
axw = Axis3(fig[1:2, 3:4], title="w", xlabel="x", ylabel="y", zlabel="z", viewmode=:fitzoom, aspect=:data)

axubar = Axis(fig[3, 1], title="<u>", xlabel="m s⁻¹", ylabel="z")
axvbar = Axis(fig[3, 2], title="<v>", xlabel="m s⁻¹", ylabel="z")
axbbar = Axis(fig[3, 3], title="<b>", xlabel="m s⁻²", ylabel="z")

axuw = Axis(fig[4, 1], title="uw", xlabel="m² s⁻²", ylabel="z")
axvw = Axis(fig[4, 2], title="vw", xlabel="m² s⁻²", ylabel="z")
axwb = Axis(fig[4, 3], title="wb", xlabel="m² s⁻³", ylabel="z")

ax∂TKE∂tbar = Axis(fig[3, 4], title="0.5 <∂(u² + v² + w²)/∂t>", xlabel="m² s⁻³", ylabel="z")
ax∂b²∂tbar = Axis(fig[4, 4], title="<∂b²/∂t>", xlabel="m² s⁻⁵", ylabel="z")

xCs_xy = xC
yCs_xy = yC
zCs_xy = [zC[Nz] for x in xCs_xy, y in yCs_xy]

yCs_yz = yC
xCs_yz = range(xC[1], stop=xC[1], length=length(zC))
zCs_yz = zeros(length(xCs_yz), length(yCs_yz))
for j in axes(zCs_yz, 2)
  zCs_yz[:, j] .= zC
end

xCs_xz = xC
yCs_xz = range(yC[1], stop=yC[1], length=length(zC))
zCs_xz = zeros(length(xCs_xz), length(yCs_xz))
for i in axes(zCs_xz, 1)
  zCs_xz[i, :] .= zC
end

xFs_xy = xC
yFs_xy = yC
zFs_xy = [zF[Nz+1] for x in xFs_xy, y in yFs_xy]

yFs_yz = yC
xFs_yz = range(xC[1], stop=xC[1], length=length(zF))
zFs_yz = zeros(length(xFs_yz), length(yFs_yz))
for j in axes(zFs_yz, 2)
  zFs_yz[:, j] .= zF
end

xFs_xz = xC
yFs_xz = range(yC[1], stop=yC[1], length=length(zF))
zFs_xz = zeros(length(xFs_xz), length(yFs_xz))
for i in axes(zFs_xz, 1)
  zFs_xz[i, :] .= zF
end

function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

blim = (minimum(b_data), maximum(b_data))
wlim = (minimum(w_data), maximum(w_data))

colormap = Reverse(:RdBu_10)
b_color_range = blim
w_color_range = wlim

ubarlim = (minimum(ubar_data), maximum(ubar_data))
vbarlim = (minimum(vbar_data), maximum(vbar_data))
bbarlim = (minimum(bbar_data), maximum(bbar_data))

∂b²∂tlim = (find_min(∂b²∂t_data, ∂b²∂t′_data), 
            find_max(∂b²∂t_data, ∂b²∂t′_data))
∂TKE∂tlim = (find_min(0.5 .* (∂u²∂t_data .+ ∂v²∂t_data .+ ∂w²∂t_data), 0.5 .* (∂u²∂t′_data .+ ∂v²∂t′_data .+ ∂w²∂t′_data)), 
             find_max(0.5 .* (∂u²∂t_data .+ ∂v²∂t_data .+ ∂w²∂t_data), 0.5 .* (∂u²∂t′_data .+ ∂v²∂t′_data .+ ∂w²∂t′_data)))

startframe_lim = 30
uwlim = (minimum(uw_data[1, 1, :, startframe_lim:end]), maximum(uw_data[1, 1, :, startframe_lim:end]))
vwlim = (minimum(vw_data[1, 1, :, startframe_lim:end]), maximum(vw_data[1, 1, :, startframe_lim:end]))
wblim = (minimum(wb_data[1, 1, :, startframe_lim:end]), maximum(wb_data[1, 1, :, startframe_lim:end]))

n = Observable(1)

bₙ_xy = @lift interior(b_data[$n], :, :, Nz)
bₙ_yz = @lift transpose(interior(b_data[$n], 1, :, :))
bₙ_xz = @lift interior(b_data[$n], :, 1, :)

wₙ_xy = @lift interior(w_data[$n], :, :, Nz+1)
wₙ_yz = @lift transpose(interior(w_data[$n], 1, :, :))
wₙ_xz = @lift interior(w_data[$n], :, 1, :)

time_str = @lift "Qᵁ = $(Qᵁ), Qᴮ = $(Qᴮ), Time = $(round(bbar_data.times[$n]/24/60^2, digits=3)) days"
title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)

b_xy_surface = surface!(axb, xCs_xy, yCs_xy, zCs_xy, color=bₙ_xy, colormap=colormap, colorrange = b_color_range)
b_yz_surface = surface!(axb, xCs_yz, yCs_yz, zCs_yz, color=bₙ_yz, colormap=colormap, colorrange = b_color_range)
b_xz_surface = surface!(axb, xCs_xz, yCs_xz, zCs_xz, color=bₙ_xz, colormap=colormap, colorrange = b_color_range)

w_xy_surface = surface!(axw, xFs_xy, yFs_xy, zFs_xy, color=wₙ_xy, colormap=colormap, colorrange = w_color_range)
w_yz_surface = surface!(axw, xFs_yz, yFs_yz, zFs_yz, color=wₙ_yz, colormap=colormap, colorrange = w_color_range)
w_xz_surface = surface!(axw, xFs_xz, yFs_xz, zFs_xz, color=wₙ_xz, colormap=colormap, colorrange = w_color_range)

ubarₙ = @lift interior(ubar_data[$n], 1, 1, :)
vbarₙ = @lift interior(vbar_data[$n], 1, 1, :)
bbarₙ = @lift interior(bbar_data[$n], 1, 1, :)

uwₙ = @lift interior(uw_data[$n], 1, 1, :)
vwₙ = @lift interior(vw_data[$n], 1, 1, :)
wbₙ = @lift interior(wb_data[$n], 1, 1, :)

∂TKE∂tbarₙ = @lift 0.5 .* (interior(∂u²∂t_data[$n], 1, 1, :) .+ interior(∂v²∂t_data[$n], 1, 1, :) .+ interior(∂w²∂t_data[$n], 1, 1, :))
∂TKE∂t′barₙ = @lift 0.5 .* (interior(∂u²∂t′_data[$n], 1, 1, :) .+ interior(∂v²∂t′_data[$n], 1, 1, :) .+ interior(∂w²∂t′_data[$n], 1, 1, :))

∂b²∂tbarₙ = @lift interior(∂b²∂t_data[$n], 1, 1, :)
∂b²∂t′barₙ = @lift interior(∂b²∂t′_data[$n], 1, 1, :)

lines!(axubar, ubarₙ, zC)
lines!(axvbar, vbarₙ, zC)
lines!(axbbar, bbarₙ, zC)

lines!(axuw, uwₙ, zF)
lines!(axvw, vwₙ, zF)
lines!(axwb, wbₙ, zF)

lines!(ax∂TKE∂tbar, ∂TKE∂tbarₙ, zC, label="Total")
lines!(ax∂TKE∂tbar, ∂TKE∂t′barₙ, zC, label="Sum of components")
axislegend(ax∂TKE∂tbar, position=:rb)

lines!(ax∂b²∂tbar, ∂b²∂tbarₙ, zC, label="Total")
lines!(ax∂b²∂tbar, ∂b²∂t′barₙ, zC, label="Sum of components")
axislegend(ax∂b²∂tbar, position=:rb)


xlims!(axubar, ubarlim)
xlims!(axvbar, vbarlim)
xlims!(axbbar, bbarlim)

xlims!(axuw, uwlim)
xlims!(axvw, vwlim)
xlims!(axwb, wblim)

xlims!(ax∂TKE∂tbar, ∂TKE∂tlim)
xlims!(ax∂b²∂tbar, ∂b²∂tlim)

trim!(fig.layout)

record(fig, "$(FILE_DIR)/video.mp4", 1:Nt, framerate=15) do nn
    n[] = nn
end

@info "Animation completed"
#%%
=#