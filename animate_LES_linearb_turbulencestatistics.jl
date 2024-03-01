using Oceananigans
using CairoMakie
using JLD2

FILE_DIR = "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_32_64"

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

parameters = jldopen("$(FILE_DIR)/instantaneous_timeseries.jld2", "r") do file
    return Dict([(key, file["metadata/parameters/$(key)"]) for key in keys(file["metadata/parameters"])])
end 

Nx, Ny, Nz = bbar_data.grid.Nx, bbar_data.grid.Ny, bbar_data.grid.Nz
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

ax∂TKE∂tbar = Axis(fig[3, 4], title="<∂(u² + v² + w²)/∂t>", xlabel="m² s⁻³", ylabel="z")
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

Qᵁ = parameters["momentum_flux"]
Qᴮ = parameters["buoyancy_flux"]

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
display(fig)

record(fig, "$(FILE_DIR)/turbulence_statistics.mp4", 1:Nt, framerate=15) do nn
    n[] = nn
end

@info "Animation completed"