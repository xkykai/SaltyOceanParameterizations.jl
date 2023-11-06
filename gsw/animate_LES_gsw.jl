using Oceananigans
using CairoMakie
using JLD2
using FileIO
using Statistics
using GibbsSeaWater
using SeawaterPolynomials
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
    end
    return parse_args(s)
end

args = parse_commandline()

# args["Lx"] = 128.
# args["Ly"] = 128.
# args["Nx"] = 64
# args["Ny"] = 64
# args["Nz"] = 128
# args["QU"] = -1e-3
# args["QT"] = 5e-5
# args["QS"] = -1e-4

# args["T_surface"] = 4.1
# args["S_surface"] = 0.0

# args["dTdz"] = 4 / 256 
# args["dSdz"] = -1 / 256

# Lz = args["Lz"]
# Lx = args["Lx"]
# Ly = args["Ly"]

# Nz = args["Nz"]
# Nx = args["Nx"]
# Ny = args["Ny"]

# Qᵁ = args["QU"]
# Qᵀ = args["QT"]
# Qˢ = args["QS"]

# f = args["f"]

# dTdz = args["dTdz"]
# dSdz = args["dSdz"]

# T_surface = args["T_surface"]
# S_surface = args["S_surface"]

FILE_NAME = "linearTS_dTdz_$(dTdz)_dSdz_$(dSdz)_QU_$(Qᵁ)_QT_$(Qᵀ)_QS_$(Qˢ)_T_$(T_surface)_S_$(S_surface)_$(args["advection"])_Lxz_$(Lx)_$(Lz)_Nxz_$(Nx)_$(Nz)"
FILE_DIR = "LES/$(FILE_NAME)"

parameters = jldopen("$(FILE_DIR)/instantaneous_timeseries.jld2", "r") do file
  return Dict([(key, file["metadata/parameters/$(key)"]) for key in keys(file["metadata/parameters"])])
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
ρbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "ρbar")

Tbar_face_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "Tbar_face")
Sbar_face_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "Sbar_face")

∂Tbar∂z_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "∂Tbar∂z")
∂Sbar∂z_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "∂Sbar∂z")
∂bbar∂z_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "∂bbar∂z")
∂ρbar∂z_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "∂ρbar∂z")

ρ_bulk_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "ρ_bulk")

∂ρ_bulk∂z_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "∂ρ_bulk∂z")
∂b_bulk∂z_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "∂b_bulk∂z")

α_bulk_∂Tbar∂z_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "α_bulk_∂Tbar∂z")
β_bulk_∂Sbar∂z_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "β_bulk_∂Sbar∂z")

α_∂T∂z_bar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "α_∂T∂z_bar")
β_∂S∂z_bar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "β_∂S∂z_bar")

uw_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "uw")
vw_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "vw")
wT_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wT")
wS_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wS")
wb_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wb")
wb′_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wb′")

∂wb∂z_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "∂wb∂z")
∂wb′∂z_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "∂wb′∂z")
∂wb′′∂z_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "∂wb′′∂z")

xC = T_xy_data.grid.xᶜᵃᵃ[1:Nx]
yC = T_xy_data.grid.yᵃᶜᵃ[1:Ny]
zC = T_xy_data.grid.zᵃᵃᶜ[1:Nz]

zF = uw_data.grid.zᵃᵃᶠ[1:Nz+1]
ρ₀ = mean(ρbar_data[1, 1, 1:end, 1])
g = parameters["gravitational_acceleration"]

c_bulk = zeros(size(Tbar_face_data))
for l in axes(c_bulk, 4)
  c_bulk[1, 1, :, l] .= gsw_sound_speed.(Sbar_face_data[1, 1, 1:end, l], Tbar_face_data[1, 1, 1:end, l], -zF .* ρ₀ * g * 1e-4)
end

α_bulk_∂Tbar∂z_β_bulk_∂Sbar∂z_ρ_RHS_data = ρ₀ .* (-interior(α_bulk_∂Tbar∂z_data) .+ interior(β_bulk_∂Sbar∂z_data) .- g ./ c_bulk .^2)[:, :, 2:end-1, :]
∂ρbar∂z_RHS_data = -g / ρ₀ .* interior(∂ρbar∂z_data)[:, :, 2:end-1, :]
∂ρ_bulk∂z_RHS_data = -g / ρ₀ .* interior(∂ρ_bulk∂z_data)[:, :, 2:end-1, :]

Nt = length(T_xy_data.times)

##
fig = Figure(resolution=(3000, 1800))

axubar = Axis(fig[1, 1], title="<u>", xlabel="m s⁻¹", ylabel="z")
axvbar = Axis(fig[1, 2], title="<v>", xlabel="m s⁻¹", ylabel="z")
axTbar = Axis(fig[1, 3], title="<T>", xlabel="°C", ylabel="z")
axSbar = Axis(fig[1, 4], title="<S>", xlabel="g kg⁻¹", ylabel="z")

axuw = Axis(fig[2, 1], title="uw", xlabel="m² s⁻²", ylabel="z")
axvw = Axis(fig[2, 2], title="vw", xlabel="m² s⁻²", ylabel="z")
axwT = Axis(fig[2, 3], title="wT", xlabel="m s⁻¹ °C", ylabel="z")
axwS = Axis(fig[2, 4], title="wS", xlabel="m s⁻¹ g kg⁻¹", ylabel="z")

axρ = Axis(fig[1:2, 5:6], title="ρ", xlabel="kg m⁻³", ylabel="z")
axwb = Axis(fig[3:4, 1:2], title="wb", xlabel="m² s⁻³", ylabel="z")
ax∂zwb = Axis(fig[3:4, 3:4], title="∂z(wb)", xlabel="m s⁻³", ylabel="z")
ax∂zρ = Axis(fig[3:4, 5:6], title="∂z(ρ)", xlabel="kg m⁻⁴", ylabel="z")

function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

ubarlim = (minimum(ubar_data), maximum(ubar_data))
vbarlim = (minimum(vbar_data), maximum(vbar_data))
Tbarlim = (minimum(Tbar_data), maximum(Tbar_data))
Sbarlim = (minimum(Sbar_data), maximum(Sbar_data))

startframe_lim = 30
uwlim = (minimum(uw_data[1, 1, :, startframe_lim:end]), maximum(uw_data[1, 1, :, startframe_lim:end]))
vwlim = (minimum(vw_data[1, 1, :, startframe_lim:end]), maximum(vw_data[1, 1, :, startframe_lim:end]))
wTlim = (minimum(wT_data[1, 1, :, startframe_lim:end]), maximum(wT_data[1, 1, :, startframe_lim:end]))
wSlim = (minimum(wS_data[1, 1, :, startframe_lim:end]), maximum(wS_data[1, 1, :, startframe_lim:end]))

wblim = (find_min(wb_data[1, 1, :, startframe_lim:end], wb′_data[1, 1, :, startframe_lim:end]), find_max(wb_data[1, 1, :, startframe_lim:end], wb′_data[1, 1, :, startframe_lim:end]))

ρlim = (find_min(ρbar_data, ρ_bulk_data), 
        find_max(ρbar_data, ρ_bulk_data))
∂zρlim = (find_min(∂ρbar∂z_data[1, 1, 2:end-1, startframe_lim:end], α_bulk_∂Tbar∂z_β_bulk_∂Sbar∂z_ρ_RHS_data[:, :, :, startframe_lim:end]), 
          find_max(∂ρbar∂z_data[1, 1, 2:end-1, startframe_lim:end], α_bulk_∂Tbar∂z_β_bulk_∂Sbar∂z_ρ_RHS_data[:, :, :, startframe_lim:end]))

∂zwblim = (find_min(∂wb∂z_data[1, 1, 2:end-1, startframe_lim:end], ∂wb′∂z_data[1, 1, 2:end-1, startframe_lim:end], ∂wb′′∂z_data[1, 1, 2:end-1, startframe_lim:end]), 
           find_max(∂wb∂z_data[1, 1, 2:end-1, startframe_lim:end], ∂wb′∂z_data[1, 1, 2:end-1, startframe_lim:end], ∂wb′′∂z_data[1, 1, 2:end-1, startframe_lim:end]))

n = Observable(1)

Tₙ_xy = @lift interior(T_xy_data[$n], :, :, 1)
Tₙ_yz = @lift transpose(interior(T_yz_data[$n], 1, :, :))
Tₙ_xz = @lift interior(T_xz_data[$n], :, 1, :)

Sₙ_xy = @lift interior(S_xy_data[$n], :, :, 1)
Sₙ_yz = @lift transpose(interior(S_yz_data[$n], 1, :, :))
Sₙ_xz = @lift interior(S_xz_data[$n], :, 1, :)

time_str = @lift "Qᵁ = $(Qᵁ), Qᵀ = $(Qᵀ), Qˢ = $(Qˢ), Time = $(round(T_xy_data.times[$n]/24/60^2, digits=3)) days"
title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)

ubarₙ = @lift interior(ubar_data[$n], 1, 1, :)
vbarₙ = @lift interior(vbar_data[$n], 1, 1, :)
Tbarₙ = @lift interior(Tbar_data[$n], 1, 1, :)
Sbarₙ = @lift interior(Sbar_data[$n], 1, 1, :)
ρbarₙ = @lift interior(ρbar_data[$n], 1, 1, :)

uwₙ = @lift interior(uw_data[$n], 1, 1, :)
vwₙ = @lift interior(vw_data[$n], 1, 1, :)
wTₙ = @lift interior(wT_data[$n], 1, 1, :)
wSₙ = @lift interior(wS_data[$n], 1, 1, :)
wbₙ = @lift interior(wb_data[$n], 1, 1, :)
wb′ₙ = @lift interior(wb′_data[$n], 1, 1, :)

ρ_bulkₙ = @lift interior(ρ_bulk_data[$n], 1, 1, :)
α_bulk_∂Tbar∂z_β_bulk_∂Sbar∂z_ρ_RHSₙ = @lift α_bulk_∂Tbar∂z_β_bulk_∂Sbar∂z_ρ_RHS_data[1, 1, :, $n]

∂wb∂zₙ = @lift interior(∂wb∂z_data[$n], 1, 1, :)
∂wb′∂zₙ = @lift interior(∂wb′∂z_data[$n], 1, 1, :)
∂wb′′∂zₙ = @lift interior(∂wb′′∂z_data[$n], 1, 1, :)

∂ρ_bulk∂z_RHSₙ = @lift ∂ρ_bulk∂z_RHS_data[1, 1, :, $n]
∂ρ_bulk∂zₙ = @lift interior(∂ρ_bulk∂z_data[$n], 1, 1, 2:Nz)
∂b_bulk∂zₙ = @lift interior(∂b_bulk∂z_data[$n], 1, 1, 2:Nz)
∂ρbar∂zₙ = @lift interior(∂ρbar∂z_data[$n], 1, 1, 2:Nz)

lines!(axubar, ubarₙ, zC)
lines!(axvbar, vbarₙ, zC)
lines!(axTbar, Tbarₙ, zC)
lines!(axSbar, Sbarₙ, zC)

lines!(axuw, uwₙ, zF)
lines!(axvw, vwₙ, zF)
lines!(axwT, wTₙ, zF)
lines!(axwS, wSₙ, zF)

lines!(axwb, wb′ₙ, zF, label="g * <αwT - βwS>", linewidth=8, alpha=0.5)
lines!(axwb, wbₙ, zF, label="<wb>", color=:black)
axislegend(axwb, position=:rb)

lines!(axρ, ρ_bulkₙ, zC, label="ρ(<T>, <S>)", linewidth=8, alpha=0.5)
lines!(axρ, ρbarₙ, zC, label="<ρ(T, S)>", color=:black)
axislegend(axρ, position=:rb)

lines!(ax∂zρ, α_bulk_∂Tbar∂z_β_bulk_∂Sbar∂z_ρ_RHSₙ, zF[2:end-1], label="ρ₀ * [-α(<T>, <S>)*∂z(<T>) + β(<T>, <S>)*∂z(<S>) - g/c²(<T>, <S>, ρ₀)]", linewidth=8, alpha=0.5)
# lines!(ax∂zρ, α_∂T∂z_bar_β_∂S∂z_bar_RHSₙ, zF[2:end-1], label="ρ₀ * [<-α(T, S)*∂z(T)> + <β(T, S)*∂z(S)>]", linewidth=8, alpha=0.5)
lines!(ax∂zρ, ∂ρ_bulk∂zₙ, zF[2:end-1], label="∂z(ρ(<T>, <S>))", linewidth=8, alpha=0.5)
lines!(ax∂zρ, ∂ρbar∂zₙ, zF[2:end-1], label="<∂z(ρ(T, S))>", color=:black)
axislegend(ax∂zρ, position=:rb)
# Legend(fig[4, 5], ax∂zρ)

lines!(ax∂zwb, ∂wb′∂zₙ, zC, label="g ∂z(<αwT - βwS>)", linewidth=8, alpha=0.5)
lines!(ax∂zwb, ∂wb′′∂zₙ, zC, label="g <α ∂z(wT) - β ∂z(wS)>", linewidth=8, alpha=0.5)
lines!(ax∂zwb, ∂wb∂zₙ, zC, label="g ∂z(<wb>)", color=:black)
axislegend(ax∂zwb, position=:rb)

xlims!(axubar, ubarlim)
xlims!(axvbar, vbarlim)
xlims!(axTbar, Tbarlim)
xlims!(axSbar, Sbarlim)

xlims!(axuw, uwlim)
xlims!(axvw, vwlim)
xlims!(axwT, wTlim)
xlims!(axwS, wSlim)

xlims!(axwb, wblim)
xlims!(axρ, ρlim)
xlims!(ax∂zρ, ∂zρlim)
xlims!(ax∂zwb, ∂zwblim)

trim!(fig.layout)

record(fig, "$(FILE_DIR)/video.mp4", 1:Nt, framerate=15) do nn
    n[] = nn
end

@info "Animation completed"