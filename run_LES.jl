using Oceananigans
using Oceananigans.Units
# using Logging
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
using ArgParse

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
      "--f"
        help = "Coriolis parameter (s⁻¹)"
        arg_type = Float64
        default = 1e-4
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
    end
    return parse_args(s)
end

args = parse_commandline()

Random.seed!(123)
# Logging.global_logger(OceananigansLogger())

const Lz = 256meter    # depth [m]
const Lx = 512meter
const Ly = 512meter

const Nz = 128
const Nx = 256
const Ny = 256

# const Nz = 32
# const Nx = 64
# const Ny = 64

const Qᵁ = args["QU"]
const Qᵀ = args["QT"]
const Qˢ = args["QS"]

const Pr = 1

if args["advection"] == "WENO9nu1e-5"
    advection = WENO(order=9)
    closure = ScalarDiffusivity(ν=1e-5, κ=1e-5/Pr)
elseif args["advection"] == "AMD"
    advection = CenteredSecondOrder()
    closure = AnisotropicMinimumDissipation()
end

const f = args["f"]

const λᵀ = 8e-5
const λˢ = 6e-4

const T_surface = args["T_surface"]
const S_surface = args["S_surface"]

const pickup = args["pickup"]

FILE_NAME = "QU_$(Qᵁ)_QT_$(Qᵀ)_QS_$(Qˢ)_Ttop_$(T_surface)_Stop_$(S_surface)_sponge_$(args["advection"]))"
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
S_initial(x, y, z) = S_surface * exp(-λˢ*z) + 1e-6 * noise(x, y, z)

dTdz_bot = T_surface * λᵀ * exp(-λᵀ*Lz)
dSdz_bot = -S_surface * λˢ * exp(-λˢ*Lz)

T_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵀ), bottom=GradientBoundaryCondition(dTdz_bot))
S_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qˢ), bottom=GradientBoundaryCondition(dSdz_bot))
u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵁ))

const eos = TEOS10EquationOfState()

damping_rate = 1/2minute

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
            tracers = (:T, :S, :b),
            timestepper = :RungeKutta3,
            advection = advection,
            forcing = (u=uvw_sponge, v=uvw_sponge, w=uvw_sponge, T=T_sponge, S=S_sponge),
            boundary_conditions = (T=T_bcs, S=S_bcs, u=u_bcs)
            )

set!(model, T=T_initial, S=S_initial)

T = model.tracers.T
S = model.tracers.S
u, v, w = model.velocities

const T₀ = mean(T)
const S₀ = mean(S)
const ρ₀ = TEOS10.ρ(T₀, S₀, 0, eos)
const g = model.buoyancy.model.gravitational_acceleration

simulation = Simulation(model, Δt=args["dt"]second, stop_time=args["stop_time"]days)

wizard = TimeStepWizard(max_change=1.05, max_Δt=args["max_dt"]minutes, cfl=0.6)
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

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(100))

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

@inline function calculate_α(i, j, k, grid, T, S, eos)
    @inbounds return Oceananigans.BuoyancyModels.thermal_expansionᶜᶜᶜ(i, j, k, grid, eos, T, S)
    # @inbounds return Oceananigans.BuoyancyModels.thermal_expansionᶜᶜᶜ(i, j, k, grid, eos, T, S)
end

@inline function calculate_β(i, j, k, grid, T, S, eos)
    @inbounds return Oceananigans.BuoyancyModels.haline_contractionᶜᶜᶜ(i, j, k, grid, eos, T, S)
    # @inbounds return Oceananigans.BuoyancyModels.haline_contractionᶜᶜᶜ(i, j, k, grid, eos, T, S)
end

α_op = KernelFunctionOperation{Center, Center, Center}(calculate_α, grid, T, S, eos)
α = Field(α_op)
compute!(α)

β_op = KernelFunctionOperation{Center, Center, Center}(calculate_β, grid, T, S, eos)
β = Field(β_op)
compute!(β)

@inline function get_buoyancy(i, j, k, grid, b, C)
    T, S = Oceananigans.BuoyancyModels.get_temperature_and_salinity(b, C)
    @inbounds ρ = Oceananigans.BuoyancyModels.ρ′(i, j, k, grid, b.model.equation_of_state, T, S) + b.model.equation_of_state.reference_density
    ρ′ = ρ - ρ₀
    return -g * ρ′ / ρ₀
end

b_op = KernelFunctionOperation{Center, Center, Center}(get_buoyancy, model.grid, model.buoyancy, model.tracers)
b = Field(b_op)
compute!(b)

# fig = Figure(resolution=(1800, 1500))
# ax1 = Axis(fig[1, 1], title="b")
# ax2 = Axis(fig[1, 2], title="b′")

# mean(b, dims=(1, 2))
# mean(b′, dims=(1, 2))

# heatmap!(ax1, b[1, 1:64, 1:32], colormap=:thermal, colorrange=(minimum(b), maximum(b)))
# heatmap!(ax2, b′[1, 1:64, 1:32], colormap=:thermal, colorrange=(minimum(b′), maximum(b′)))
# display(fig)

ubar = Average(u, dims=(1, 2))
vbar = Average(v, dims=(1, 2))
Tbar = Average(T, dims=(1, 2))
Sbar = Average(S, dims=(1, 2))

uw = Average(w * u, dims=(1, 2))
vw = Average(w * v, dims=(1, 2))
wb = Average(w * b, dims=(1, 2))
wb′ = Average(w * g * (α*T - β*S), dims=(1, 2))
wT = Average(w * T, dims=(1, 2))
wS = Average(w * S, dims=(1, 2))

field_outputs = merge(model.velocities, model.tracers)

simulation.output_writers[:jld2] = JLD2OutputWriter(model, field_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_fields.jld2",
                                                          schedule = TimeInterval(10minutes),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!,
                                                          verbose = true)

simulation.output_writers[:timeseries] = JLD2OutputWriter(model, (; ubar, vbar, Tbar, Sbar, uw, vw, wT, wS, wb, wb′),
                                                          filename = "$(FILE_DIR)/instantaneous_timeseries.jld2",
                                                          schedule = TimeInterval(10minutes),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!,
                                                          verbose = true)

simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=TimeInterval(1day), prefix="$(FILE_DIR)/model_checkpoint")

# run!(simulation, pickup="$(FILE_DIR)/model_checkpoint_iteration97574.jld2")
# run!(simulation)

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
wb_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wb")
wb′_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wb′")

Nt = length(T_data.times)

xC = T_data.grid.xᶜᵃᵃ[1:Nx]
yC = T_data.grid.xᶜᵃᵃ[1:Ny]
zC = T_data.grid.zᵃᵃᶜ[1:Nz]

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
wblim = (minimum([minimum(wb_data), minimum(wb′_data)]), maximum([maximum(wb_data), maximum(wb′_data)]))

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
