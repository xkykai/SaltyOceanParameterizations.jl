using CairoMakie
using Oceananigans
using JLD2

FILE_DIRS = [
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_32_64_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_32_64_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_32_64_f",

    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_64_128_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_64_128_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_64_128_f",

    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_128_256_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_128_256_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_128_256_f",

    # # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-5e-4_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_32_64_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-5e-4_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_32_64_f",
    # # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-5e-4_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_32_64_f",

    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-5e-4_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_64_128_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-5e-4_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_64_128_f",
    # # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-5e-4_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_64_128_f",

    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-5e-4_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_128_256_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-5e-4_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_128_256_f",
    # # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-5e-4_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_128_256_f",
]

labels = [
    # "AMD, 2m resolution",
    "WENO(9), ν = κ = 0, 2m resolution",
    # "WENO(9), ν = κ = 1e-5, 2m resolution",

    "AMD, 1m resolution",
    "WENO(9), ν = κ = 0, 1m resolution",
    # "WENO(9), ν = κ = 1e-5, 1m resolution",

    "AMD, 0.5m resolution",
    "WENO(9), ν = κ = 0, 0.5m resolution",
    # "WENO(9), ν = κ = 1e-5, 0.5m resolution",
]

parameters = jldopen("$(FILE_DIRS[1])/instantaneous_timeseries.jld2", "r") do file
    return Dict([(key, file["metadata/parameters/$(key)"]) for key in keys(file["metadata/parameters"])])
end 

Qᴮ = parameters["buoyancy_flux"]
Qᵁ = parameters["momentum_flux"]

video_name = "./Data/QU_$(Qᵁ)_QB_$(Qᴮ)_btop_0_AMD_WENO_turbulence_statistics.mp4"

∂u²∂t_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "∂u²∂t") for FILE_DIR in FILE_DIRS]
∂v²∂t_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "∂v²∂t") for FILE_DIR in FILE_DIRS]
∂w²∂t_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "∂w²∂t") for FILE_DIR in FILE_DIRS]
∂b²∂t_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "∂b²∂t") for FILE_DIR in FILE_DIRS]

∂TKE∂t_datas = [0.5 .* (interior(∂u²∂t_datas[i]) .+ interior(∂v²∂t_datas[i]) .+ interior(∂w²∂t_datas[i])) for i in eachindex(∂u²∂t_datas)]

Nxs = [size(data.grid)[1] for data in ∂b²∂t_datas]
Nys = [size(data.grid)[2] for data in ∂b²∂t_datas]
Nzs = [size(data.grid)[3] for data in ∂b²∂t_datas]
Nt = length(∂b²∂t_datas[1].times)

zCs = [data.grid.zᵃᵃᶜ[1:Nzs[i]] for (i, data) in enumerate(∂b²∂t_datas)]
zFs = [data.grid.zᵃᵃᶠ[1:Nzs[i]+1] for (i, data) in enumerate(∂b²∂t_datas)]

ubar_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "ubar") for FILE_DIR in FILE_DIRS]
vbar_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "vbar") for FILE_DIR in FILE_DIRS]
bbar_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar") for FILE_DIR in FILE_DIRS]

uw_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "uw") for FILE_DIR in FILE_DIRS]
vw_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "vw") for FILE_DIR in FILE_DIRS]
wb_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wb") for FILE_DIR in FILE_DIRS]

u_udot∇u_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "u_udot∇u") for FILE_DIR in FILE_DIRS]
v_udot∇v_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "v_udot∇v") for FILE_DIR in FILE_DIRS]
b_udot∇b_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "b_udot∇b") for FILE_DIR in FILE_DIRS]
w_udot∇w_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "w_udot∇w") for FILE_DIR in FILE_DIRS]

u_∇dotνₑ∇u_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "u_∇dotνₑ∇u") for FILE_DIR in FILE_DIRS]
v_∇dotνₑ∇v_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "v_∇dotνₑ∇v") for FILE_DIR in FILE_DIRS]
b_∇dotκₑ∇b_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "b_∇dotκₑ∇b") for FILE_DIR in FILE_DIRS]
w_∇dotνₑ∇w_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "w_∇dotνₑ∇w") for FILE_DIR in FILE_DIRS]
##
fig = Figure(size=(2100, 2100))

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
display(fig)

ubarlim = (find_min(ubar_datas...), find_max(ubar_datas...))
vbarlim = (find_min(vbar_datas...), find_max(vbar_datas...))
bbarlim = (find_min(bbar_datas...), find_max(bbar_datas...))

startframe_lim = 30
uwlim = (find_min([uw_data[1, 1, :, startframe_lim:end] for uw_data in uw_datas]...), find_max([uw_data[1, 1, :, startframe_lim:end] for uw_data in uw_datas]...))
vwlim = (find_min([vw_data[1, 1, :, startframe_lim:end] for vw_data in vw_datas]...), find_max([vw_data[1, 1, :, startframe_lim:end] for vw_data in vw_datas]...))
wblim = (find_min([wb_data[1, 1, :, startframe_lim:end] for wb_data in wb_datas]...), find_max([wb_data[1, 1, :, startframe_lim:end] for wb_data in wb_datas]...))

u_udot∇ulim = (find_min(u_udot∇u_datas...), find_max(u_udot∇u_datas...))
v_udot∇vlim = (find_min(v_udot∇v_datas...), find_max(v_udot∇v_datas...))
b_udot∇blim = (find_min(b_udot∇b_datas...), find_max(b_udot∇b_datas...))
w_udot∇wlim = (find_min(w_udot∇w_datas...), find_max(w_udot∇w_datas...))

u_∇dotνₑ∇ulim = (find_min(u_∇dotνₑ∇u_datas...), find_max(u_∇dotνₑ∇u_datas...))
v_∇dotνₑ∇vlim = (find_min(v_∇dotνₑ∇v_datas...), find_max(v_∇dotνₑ∇v_datas...))
b_∇dotκₑ∇blim = (find_min(b_∇dotκₑ∇b_datas...), find_max(b_∇dotκₑ∇b_datas...))
w_∇dotνₑ∇wlim = (find_min(w_∇dotνₑ∇w_datas...), find_max(w_∇dotνₑ∇w_datas...))

∂TKE∂tlim = (find_min(∂TKE∂t_datas...), find_max(∂TKE∂t_datas...))
∂b²∂tlim = (find_min(∂b²∂t_datas...), find_max(∂b²∂t_datas...))

n = Observable(1)

time_str = @lift "Qᵁ = $(Qᵁ), Qᴮ = $(Qᴮ), Time = $(round(∂b²∂t_datas[1].times[$n]/24/60^2, digits=3)) days"
title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)

ubarₙs = [@lift interior(data[$n], 1, 1, :) for data in ubar_datas]
vbarₙs = [@lift interior(data[$n], 1, 1, :) for data in vbar_datas]
bbarₙs = [@lift interior(data[$n], 1, 1, :) for data in bbar_datas]

uwₙs = [@lift interior(data[$n], 1, 1, :) for data in uw_datas]
vwₙs = [@lift interior(data[$n], 1, 1, :) for data in vw_datas]
wbₙs = [@lift interior(data[$n], 1, 1, :) for data in wb_datas]

u_udot∇uₙs = [@lift interior(data[$n], 1, 1, :) for data in u_udot∇u_datas]
v_udot∇vₙs = [@lift interior(data[$n], 1, 1, :) for data in v_udot∇v_datas]
b_udot∇bₙs = [@lift interior(data[$n], 1, 1, :) for data in b_udot∇b_datas]
w_udot∇wₙs = [@lift interior(data[$n], 1, 1, :) for data in w_udot∇w_datas]

u_∇dotνₑ∇uₙs = [@lift interior(data[$n], 1, 1, :) for data in u_∇dotνₑ∇u_datas]
v_∇dotνₑ∇vₙs = [@lift interior(data[$n], 1, 1, :) for data in v_∇dotνₑ∇v_datas]
b_∇dotκₑ∇bₙs = [@lift interior(data[$n], 1, 1, :) for data in b_∇dotκₑ∇b_datas]
w_∇dotνₑ∇wₙs = [@lift interior(data[$n], 1, 1, :) for data in w_∇dotνₑ∇w_datas]

∂TKE∂tₙs = [@lift data[1, 1, :, $n] for data in ∂TKE∂t_datas]
∂b²∂tₙs = [@lift interior(data[$n], 1, 1, :) for data in ∂b²∂t_datas]

for i in eachindex(ubarₙs)
    lines!(axubar, ubarₙs[i], zCs[i], label=labels[i])
    lines!(axvbar, vbarₙs[i], zCs[i], label=labels[i])
    lines!(axbbar, bbarₙs[i], zCs[i], label=labels[i])
    
    lines!(axuw, uwₙs[i], zFs[i])
    lines!(axvw, vwₙs[i], zFs[i])
    lines!(axwb, wbₙs[i], zFs[i])
    
    lines!(axuadvection, u_udot∇uₙs[i], zCs[i], label=labels[i])
    lines!(axvadvection, v_udot∇vₙs[i], zCs[i], label=labels[i])
    lines!(axbadvection, b_udot∇bₙs[i], zCs[i], label=labels[i])
    lines!(axwadvection, w_udot∇wₙs[i], zCs[i], label=labels[i])

    lines!(axudissipation, u_∇dotνₑ∇uₙs[i], zCs[i], label=labels[i])
    lines!(axvdissipation, v_∇dotνₑ∇vₙs[i], zCs[i], label=labels[i])
    lines!(axbdissipation, b_∇dotκₑ∇bₙs[i], zCs[i], label=labels[i])
    lines!(axwdissipation, w_∇dotνₑ∇wₙs[i], zCs[i], label=labels[i])

    lines!(ax∂TKE∂t, ∂TKE∂tₙs[i], zCs[i], label=labels[i])
    lines!(ax∂b²∂t, ∂b²∂tₙs[i], zCs[i], label=labels[i])
end

axislegend(axubar, position=:rb)

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

record(fig, video_name, 1:Nt, framerate=15) do nn
    n[] = nn
end

@info "Animation completed"
#%%