using CairoMakie
using Oceananigans

FILE_DIRS = [
    "./LES/QU_-0.0005_QT_5.0e-6_QS_-5.0e-5_Ttop_2.0_Stop_35.0_sponge_AMD_Lz_256.0_Lx_512.0_Ly_512.0_Nz_128_Nx_256_Ny_256",
    "./LES/QU_-0.0005_QT_5.0e-6_QS_-5.0e-5_Ttop_2.0_Stop_35.0_sponge_WENO9nu1e-5_Lz_256.0_Lx_512.0_Ly_512.0_Nz_128_Nx_256_Ny_256",
]

u_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "ubar") for FILE_DIR in FILE_DIRS]
v_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "vbar") for FILE_DIR in FILE_DIRS]
T_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "Tbar") for FILE_DIR in FILE_DIRS]
S_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "Sbar") for FILE_DIR in FILE_DIRS]

uw_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "uw") for FILE_DIR in FILE_DIRS]
vw_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "vw") for FILE_DIR in FILE_DIRS]
wT_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wT") for FILE_DIR in FILE_DIRS]
wS_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wS") for FILE_DIR in FILE_DIRS]
wb_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wb") for FILE_DIR in FILE_DIRS]
wb′_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wb′") for FILE_DIR in FILE_DIRS]

parameters = jldopen("$(FILE_DIRS[1])/instantaneous_timeseries.jld2", "r") do file
    return Dict([(key, file["metadata/parameters/$(key)"]) for key in keys(file["metadata/parameters"])])
end 

Qᵁ = parameters["momentum_flux"]
Qᵀ = parameters["temperature_flux"]
Qˢ = parameters["salinity_flux"]

# T_top = parameters["surface_temperature"]
# S_top = parameters["surface_salinity"]
Nx, Ny, Nz = size(u_datas[1].grid)
zC = u_datas[1].grid.zᵃᵃᶜ[1:Nz]
zF = uw_datas[1].grid.zᵃᵃᶠ[1:Nz+1]
T_top = 2.
S_top = 35.

labels = [
    "WENO(9), ν = κ = 1e-5 ",
    "CenteredSecondOrder, AMD"
]

#%%
fig = Figure(resolution = (1800, 600))
axu = Axis(fig[1, 1], title="u (m s⁻¹)", ylabel="z")
axv = Axis(fig[1, 2], title="v (m s⁻¹)", ylabel="z")
axT = Axis(fig[1, 3], title="T (°C)", ylabel="z")
axS = Axis(fig[1, 4], title="S (g/kg)", ylabel="z")

axuw = Axis(fig[2, 1], title="uw (m² s⁻²)", ylabel="z")
axvw = Axis(fig[2, 2], title="vw (m² s⁻²)", ylabel="z")
axwT = Axis(fig[2, 3], title="wT (m s⁻¹ °C)", ylabel="z")
axwS = Axis(fig[2, 4], title="wS (m s⁻¹ g/kg)", ylabel="z")
axwb = Axis(fig[2, 5], title="wb (m² s⁻³)", ylabel="z")

function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

ulim = (find_min(u_datas), find_max(u_datas))
vlim = (find_min(v_datas), find_max(v_datas))
Tlim = (find_min(T_datas), find_max(T_datas))
Slim = (find_min(S_datas), find_max(S_datas))

uwlim = (find_min(uw_datas), find_max(uw_datas))
vwlim = (find_min(vw_datas), find_max(vw_datas))
wTlim = (find_min(wT_datas), find_max(wT_datas))
wSlim = (find_min(wS_datas), find_max(wS_datas))
wblim = (find_min([wb_datas..., wb′_datas...]), find_max([wb_datas..., wb′_datas...]))

n = Observable(1)

times = u_datas[1].times
Nt = length(times)
time_str = @lift "Qᵁ = $(Qᵁ), Qᵀ = $(Qᵀ), Qˢ = $(Qˢ), Time = $(round(times[$n]/24/60^2, digits=3)) days"
title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)

uₙs = [@lift interior(data[$n], 1, 1, :) for data in u_datas]
vₙs = [@lift interior(data[$n], 1, 1, :) for data in v_datas]
Tₙs = [@lift interior(data[$n], 1, 1, :) for data in T_datas]
Sₙs = [@lift interior(data[$n], 1, 1, :) for data in S_datas]

uwₙs = [@lift interior(data[$n], 1, 1, :) for data in uw_datas]
vwₙs = [@lift interior(data[$n], 1, 1, :) for data in vw_datas]
wTₙs = [@lift interior(data[$n], 1, 1, :) for data in wT_datas]
wSₙs = [@lift interior(data[$n], 1, 1, :) for data in wS_datas]
wbₙs = [@lift interior(data[$n], 1, 1, :) for data in wb_datas]
wb′ₙs = [@lift interior(data[$n], 1, 1, :) for data in wb′_datas]

for i in 1:length(FILE_DIRS)
    lines!(axu, uₙs[i], zC, label=labels[i])
    lines!(axv, vₙs[i], zC, label=labels[i])
    lines!(axT, Tₙs[i], zC, label=labels[i])
    lines!(axS, Sₙs[i], zC, label=labels[i])

    lines!(axuw, uwₙs[i], zF, label=labels[i])
    lines!(axvw, vwₙs[i], zF, label=labels[i])
    lines!(axwT, wTₙs[i], zF, label=labels[i])
    lines!(axwS, wSₙs[i], zF, label=labels[i])

    lines!(axwb, wbₙs[i], zF, label="wb, $(labels[i])")
    lines!(axwb, wb′ₙs[i], zF, label="g(αwT - βwS), $(labels[i])")
end

# make a legend
Legend(fig[1, 5], axu, tellwidth=false)

xlims!(axu, ulim)
xlims!(axv, vlim)
xlims!(axT, Tlim)
xlims!(axS, Slim)

xlims!(axuw, uwlim)
xlims!(axvw, vwlim)
xlims!(axwT, wTlim)
xlims!(axwS, wSlim)
xlims!(axwb, wblim)

trim!(fig.layout)

record(fig, "./Data/QU_$(Qᵁ)_QT_$(Qᵀ)_QS_$(Qˢ)_Ttop_$(T_top)_Stop_$(S_surface)_closure.mp4", 1:Nt, framerate=15) do nn
    n[] = nn
end

@info "Animation complete"
#%%