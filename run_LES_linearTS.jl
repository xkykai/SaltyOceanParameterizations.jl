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
using Oceananigans.BuoyancyModels: Zᶜᶜᶠ
using SeawaterPolynomials.TEOS10
using SeawaterPolynomials
using Random
using Statistics
using ArgParse
using SeawaterPolynomials.TEOS10: ζ, r₀, r′, τ, s, R₀₀, R₀₁, R₀₂, R₀₃, R₀₄, R₀₅, r′₀, r′₁, r′₂, r′₃

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

const dTdz = args["dTdz"]
const dSdz = args["dSdz"]

const T_surface = args["T_surface"]
const S_surface = args["S_surface"]

const pickup = args["pickup"]

FILE_NAME = "linearTS_dTdz_$(dTdz)_dSdz_$(dSdz)_QU_$(Qᵁ)_QT_$(Qᵀ)_QS_$(Qˢ)_T_$(T_surface)_S_$(S_surface)_$(args["advection"])_Lxz_$(Lx)_$(Lz)_Nxz_$(Nx)_$(Nz)"
FILE_DIR = "LES/$(FILE_NAME)"
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

T_initial(x, y, z) = dTdz * z + T_surface
S_initial(x, y, z) = dSdz * z + S_surface

T_initial_noisy(x, y, z) = T_initial(x, y, z) + 1e-6 * noise(x, y, z)
S_initial_noisy(x, y, z) = S_initial(x, y, z) + 1e-6 * noise(x, y, z)

T_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵀ), bottom=GradientBoundaryCondition(dTdz))
S_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qˢ), bottom=GradientBoundaryCondition(dSdz))
u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵁ))

const eos = TEOS10EquationOfState()

damping_rate = 1/5minute

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
            boundary_conditions = (T=T_bcs, S=S_bcs, u=u_bcs)
            )

set!(model, T=T_initial_noisy, S=S_initial_noisy)

T = model.tracers.T
S = model.tracers.S
u, v, w = model.velocities

const T₀ = mean(T)
const S₀ = mean(S)
const ρ₀ = TEOS10.ρ(T₀, S₀, 0, eos)
const g = model.buoyancy.model.gravitational_acceleration
const c₀ = gsw_sound_speed(T₀, S₀, 0)

simulation = Simulation(model, Δt=args["dt"]second, stop_time=args["stop_time"]days)
# simulation = Simulation(model, Δt=args["dt"]second, stop_time=100minutes)

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
    file["metadata/parameters/coriolis_parameter"] = f
    file["metadata/parameters/momentum_flux"] = Qᵁ
    file["metadata/parameters/temperature_flux"] = Qᵀ
    file["metadata/parameters/salinity_flux"] = Qˢ
    file["metadata/parameters/surface_temperature"] = T_surface
    file["metadata/parameters/surface_salinity"] = S_surface
    file["metadata/parameters/temperature_gradient"] = dTdz
    file["metadata/parameters/salinity_gradient"] = dSdz
    file["metadata/parameters/equation_of_state"] = eos
    file["metadata/parameters/gravitational_acceleration"] = g
    return nothing
end

@inline function calculate_α(i, j, k, grid, T, S, eos)
  @inbounds return Oceananigans.BuoyancyModels.thermal_expansionᶜᶜᶜ(i, j, k, grid, eos, T, S) * eos.reference_density / ρ₀
end

@inline function calculate_β(i, j, k, grid, T, S, eos)
  @inbounds return Oceananigans.BuoyancyModels.haline_contractionᶜᶜᶜ(i, j, k, grid, eos, T, S) * eos.reference_density / ρ₀
end

@inline function calculate_α_face(i, j, k, grid, T, S, eos)
  @inbounds return Oceananigans.BuoyancyModels.thermal_expansionᶜᶜᶠ(i, j, k, grid, eos, T, S) * eos.reference_density / ρ₀
end

@inline function calculate_β_face(i, j, k, grid, T, S, eos)
  @inbounds return Oceananigans.BuoyancyModels.haline_contractionᶜᶜᶠ(i, j, k, grid, eos, T, S) * eos.reference_density / ρ₀
end

α_op = KernelFunctionOperation{Center, Center, Center}(calculate_α, grid, T, S, eos)
α = Field(α_op)
compute!(α)

β_op = KernelFunctionOperation{Center, Center, Center}(calculate_β, grid, T, S, eos)
β = Field(β_op)
compute!(β)

α_face_op = KernelFunctionOperation{Center, Center, Face}(calculate_α_face, grid, T, S, eos)
α_face = Field(α_face_op)
compute!(α_face)

β_face_op = KernelFunctionOperation{Center, Center, Face}(calculate_β_face, grid, T, S, eos)
β_face = Field(β_face_op)
compute!(β_face)

@inline function get_buoyancy(i, j, k, grid, b, C)
  T, S = Oceananigans.BuoyancyModels.get_temperature_and_salinity(b, C)
  @inbounds ρ = Oceananigans.BuoyancyModels.ρ′(i, j, k, grid, b.model.equation_of_state, T, S) + b.model.equation_of_state.reference_density
  ρ′ = ρ - ρ₀
  return -g * ρ′ / ρ₀
end

@inline function get_density(i, j, k, grid, b, C)
  T, S = Oceananigans.BuoyancyModels.get_temperature_and_salinity(b, C)
  @inbounds ρ = Oceananigans.BuoyancyModels.ρ′(i, j, k, grid, b.model.equation_of_state, T, S) + b.model.equation_of_state.reference_density
  return ρ
end

b_op = KernelFunctionOperation{Center, Center, Center}(get_buoyancy, model.grid, model.buoyancy, model.tracers)
b = Field(b_op)
compute!(b)

ρ_op = KernelFunctionOperation{Center, Center, Center}(get_density, model.grid, model.buoyancy, model.tracers)
ρ = Field(ρ_op)
compute!(ρ)

∂ρ∂z = Field(∂z(ρ))
compute!(∂ρ∂z)

# @inline function get_sound_speed(i, j, k, grid, ∂ρ∂z)
#   @inbounds √(-ρ₀*g / (∂ρ∂z[i, j, k] - eps(eltype(∂ρ∂z))))
# end

c = Field(√(-ρ₀*g / (∂ρ∂z - eps(eltype(∂ρ∂z)))))
compute!(c)

compressibility_flux = Field(w * g^2 / c^2)
compute!(compressibility_flux)

ubar = Field(Average(u, dims=(1, 2)))
vbar = Field(Average(v, dims=(1, 2)))
Tbar = Field(Average(T, dims=(1, 2)))
Sbar = Field(Average(S, dims=(1, 2)))
bbar = Field(Average(b, dims=(1, 2)))
ρbar = Field(Average(ρ, dims=(1, 2)))

Tbar_face = Field(Average(@at((Center, Center, Face), T), dims=(1, 2)))
Sbar_face = Field(Average(@at((Center, Center, Face), S), dims=(1, 2)))

uw = Field(Average(w * u, dims=(1, 2)))
vw = Field(Average(w * v, dims=(1, 2)))
wb = Field(Average(w * b, dims=(1, 2)))
wb′ = Field(Average(w * g * (α*T - β*S), dims=(1, 2)))
wT = Field(Average(w * T, dims=(1, 2)))
wS = Field(Average(w * S, dims=(1, 2)))

∂wb∂z = Field(Average(∂z(w * b), dims=(1, 2)))
∂wb′∂z = Field(Average(∂z(w * g * (α*T - β*S)), dims=(1, 2)))

∂wb′′∂z = Field(@at((Center, Center, Center), g * (α * ∂z(w*T) - β * ∂z(w*S))))
compute!(∂wb′′∂z)

∂wb′′∂z_compressibility = Field(@at((Center, Center, Center), ∂wb′′∂z + compressibility_flux))
compute!(∂wb′′∂z_compressibility)

∂wb′′∂z_bar = Field(Average(∂wb′′∂z_compressibility, dims=(1, 2)))

@inline function calculate_α_bulk(i, j, k, grid, Tbar, Sbar, eos)
  @inbounds return Oceananigans.BuoyancyModels.thermal_expansionᶜᶜᶜ(i, j, k, grid, eos, Tbar, Sbar) * eos.reference_density / ρ₀
end

@inline function calculate_β_bulk(i, j, k, grid, Tbar, Sbar, eos)
  @inbounds return Oceananigans.BuoyancyModels.haline_contractionᶜᶜᶜ(i, j, k, grid, eos, Tbar, Sbar) * eos.reference_density / ρ₀
end

α_bulk_op = KernelFunctionOperation{Nothing, Nothing, Center}(calculate_α_bulk, grid, Tbar, Sbar, eos)
α_bulk = Field(α_bulk_op)
compute!(α_bulk)

β_bulk_op = KernelFunctionOperation{Nothing, Nothing, Center}(calculate_β_bulk, grid, Tbar, Sbar, eos)
β_bulk = Field(β_bulk_op)
compute!(β_bulk)

@inline function get_bulk_density(i, j, k, grid, b, Tbar, Sbar)
  @inbounds ρ = Oceananigans.BuoyancyModels.ρ′(i, j, k, grid, b.model.equation_of_state, Tbar, Sbar) + b.model.equation_of_state.reference_density
  return ρ
end

ρ_bulk_op = KernelFunctionOperation{Nothing, Nothing, Center}(get_bulk_density, model.grid, model.buoyancy, Tbar, Sbar)
ρ_bulk = Field(ρ_bulk_op)
compute!(ρ_bulk)

∂ρ_bulk∂z = Field(∂z(ρ_bulk))
compute!(∂ρ_bulk∂z)

@inline function get_bulk_buoyancy(i, j, k, grid, b, Tbar, Sbar)
  @inbounds ρ = Oceananigans.BuoyancyModels.ρ′(i, j, k, grid, b.model.equation_of_state, Tbar, Sbar) + b.model.equation_of_state.reference_density
  ρ′ = ρ - ρ₀
  return -g * ρ′ / ρ₀
end

b_bulk_op = KernelFunctionOperation{Nothing, Nothing, Center}(get_bulk_buoyancy, model.grid, model.buoyancy, Tbar, Sbar)
b_bulk = Field(b_bulk_op)
compute!(b_bulk)
b_bulk[:]

∂b_bulk∂z = Field(∂z(b_bulk))
compute!(∂b_bulk∂z)
∂b_bulk∂z[:]

# @inline function get_sound_speed_bulk(i, j, k, grid, Tbar, Sbar)
#   @inbounds return gsw_sound_speed(Sbar[1, 1, k], Tbar[1, 1, k], -Zᶜᶜᶠ(i, j, k, grid) * ρ₀ * g * 1e-4)
# end

# c_bulk_op = KernelFunctionOperation{Nothing, Nothing, Face}(get_sound_speed_bulk, grid, Tbar, Sbar)
# c_bulk = Field(c_bulk_op)
# compute!(c_bulk)

# @inline function calculate_p_sensitivity_bulk(i, j, k, grid, c)
#   @inbounds return -ρ₀ * g / c[1, 1, k]^2
# end

# p_sensitivity_bulk_op = KernelFunctionOperation{Nothing, Nothing, Face}(calculate_p_sensitivity_bulk, grid, Tbar_face, Sbar_face)
# p_sensitivity_bulk = Field(p_sensitivity_bulk_op)
# compute!(p_sensitivity_bulk)

# wg_c² = Field(Average(w * g / c^2, dims=(1, 2)))

∂Tbar∂z = Field(Average(∂z(T), dims=(1, 2)))
∂Sbar∂z = Field(Average(∂z(S), dims=(1, 2)))
∂bbar∂z = Field(Average(∂z(b), dims=(1, 2)))
∂ρbar∂z = Field(Average(∂ρ∂z, dims=(1, 2)))

α_bulk_∂Tbar∂z = Field(∂Tbar∂z * α_bulk)
β_bulk_∂Sbar∂z = Field(∂Sbar∂z * β_bulk)

α_∂T∂z_bar = Field(Average(∂z(T) * α, dims=(1, 2)))
β_∂S∂z_bar = Field(Average(∂z(S) * β, dims=(1, 2)))

field_outputs = merge(model.velocities, model.tracers)
timeseries_outputs = (; ubar, vbar, Tbar, Sbar, bbar, ρbar,
                        Tbar_face, Sbar_face,
                        # uw, vw, wT, wS, wb, wb′, wg_c²,
                        uw, vw, wT, wS, wb, wb′,
                        ∂wb∂z, ∂wb′∂z, ∂wb′′∂z_bar,
                        ρ_bulk, ∂ρ_bulk∂z, b_bulk, ∂b_bulk∂z,
                        ∂Tbar∂z, ∂Sbar∂z, ∂bbar∂z, ∂ρbar∂z,
                        α_bulk_∂Tbar∂z, β_bulk_∂Sbar∂z,
                        α_∂T∂z_bar, β_∂S∂z_bar)
                        # p_sensitivity_bar, p_sensitivity_bulk)

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