using CairoMakie
using Oceananigans
using JLD2

FILE_DIRS = [
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_32_64_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_32_64_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_32_64_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_32_64_f",

    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_64_128_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_64_128_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_64_128_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_64_128_f",

    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_128_256_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_128_256_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_128_256_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_128_256_f",

    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_256_512_f",

    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_32_64_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_32_64_f",
    # # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_32_64_f",
    # # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_32_64_f",

    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_64_128_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_64_128_f",
    # # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_64_128_f",
    # # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_64_128_f",

    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_128_256_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_128_256_f",
    # # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_128_256_f",
    # # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_128_256_f",

    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_256_512_f",
]

labels = [
    "AMD, 2m resolution",
    # "WENO(9), ν = κ = 0, 2m resolution",
    # "WENO(9), ν = κ = 1e-5, 2m resolution",
    # "WENO(9) + AMD, 2m resolution",

    "AMD, 1m resolution",
    # "WENO(9), ν = κ = 0, 1m resolution",
    # "WENO(9), ν = κ = 1e-5, 1m resolution",
    # "WENO(9) + AMD, 1m resolution",

    "AMD, 0.5m resolution",
    # "WENO(9), ν = κ = 0, 0.5m resolution",
    # "WENO(9), ν = κ = 1e-5, 0.5m resolution",
    # "WENO(9) + AMD, 0.5m resolution",

    "AMD, 0.25m resolution"
]

parameters = jldopen("$(FILE_DIRS[1])/instantaneous_timeseries.jld2", "r") do file
    return Dict([(key, file["metadata/parameters/$(key)"]) for key in keys(file["metadata/parameters"])])
end 

Qᴮ = parameters["buoyancy_flux"]
Qᵁ = parameters["momentum_flux"]

video_name = "./Data/QU_$(Qᵁ)_QB_$(Qᴮ)_btop_0_AMD_resolution.mp4"

b_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar") for FILE_DIR in FILE_DIRS]

Nxs = [size(data.grid)[1] for data in b_datas]
Nys = [size(data.grid)[2] for data in b_datas]
Nzs = [size(data.grid)[3] for data in b_datas]

zCs = [data.grid.zᵃᵃᶜ[1:Nzs[i]] for (i, data) in enumerate(b_datas)]
zFs = [data.grid.zᵃᵃᶠ[1:Nzs[i]+1] for (i, data) in enumerate(b_datas)]

#%%
fig = Figure(size = (2400, 1200))
axb = Axis(fig[1, 1], title="b", ylabel="z")
# axΔb = Axis(fig[1, 2], title="b anomaly", ylabel="z")

function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

blim = (find_min(b_datas...), find_max(b_datas...))

n = Observable(1)

times = 0:600:2*24*60^2
Nt = length(times)
time_str = @lift "Qᵁ = $(Qᵁ), Qᴮ = $(Qᴮ), Time = $(round(times[$n]/24/60^2, digits=3)) days"
title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)

bₙs = [@lift interior(data[findfirst(x -> x≈times[$n], data.times)], 1, 1, :) for data in b_datas]

for i in 1:length(FILE_DIRS)
    lines!(axb, bₙs[i], zCs[i], label=labels[i])
end

# make a legend
Legend(fig[2, :], axb, tellwidth=false, orientation=:horizontal)

xlims!(axb, blim)

trim!(fig.layout)
# display(fig)
# save("./Data/QU_$(Qᵁ)_QB_$(Qᴮ)_btop_0_AMD_resolution.png", fig, px_per_unit=4)

record(fig, video_name, 1:Nt, framerate=15) do nn
    n[] = nn
end

@info "Animation complete"
#%%
#=
times = 0:600:2*24*60^2
Nt = length(times)
b_indices = [[findfirst(x -> x≈times[i], data.times) for i in 1:Nt] for data in b_datas]

fig = Figure(size=(2500, 500))

axb_AMD_0p5m = Axis(fig[1, 1], title="AMD, 0.5m resolution", xlabel="t", ylabel="z")
axb_WENO9nu0_0p5m = Axis(fig[1, 2], title="WENO(9), ν = κ = 0, 0.5m resolution", xlabel="t", ylabel="z")

axb_AMD_1m = Axis(fig[1, 3], title="AMD, 1m resolution", xlabel="t", ylabel="z")
axb_WENO9nu0_1m = Axis(fig[1, 4], title="WENO(9), ν = κ = 0, 1m resolution", xlabel="t", ylabel="z")

axb_AMD_2m = Axis(fig[1, 5], title="AMD, 2m resolution", xlabel="t", ylabel="z")
axb_WENO9nu0_2m = Axis(fig[1, 6], title="WENO(9), ν = κ = 0, 2m resolution", xlabel="t", ylabel="z")
Label(fig[0, :], "b", font=:bold)
clim = (find_min(b_datas...), find_max(b_datas...))

heatmap!(axb_AMD_2m, times, zCs[1], interior(b_datas[1])[1, 1, :, b_indices[1]]', colorrange=clim, colormap=Reverse(:RdBu_10))
heatmap!(axb_WENO9nu0_2m, times, zCs[2], interior(b_datas[2])[1, 1, :, b_indices[2]]', colorrange=clim, colormap=Reverse(:RdBu_10))

heatmap!(axb_AMD_1m, times, zCs[3], interior(b_datas[3])[1, 1, :, b_indices[3]]', colorrange=clim, colormap=Reverse(:RdBu_10))
heatmap!(axb_WENO9nu0_1m, times, zCs[4], interior(b_datas[4])[1, 1, :, b_indices[4]]', colorrange=clim, colormap=Reverse(:RdBu_10))

heatmap!(axb_AMD_0p5m, times, zCs[5], interior(b_datas[5])[1, 1, :, b_indices[5]]', colorrange=clim, colormap=Reverse(:RdBu_10))
heatmap!(axb_WENO9nu0_0p5m, times, zCs[6], interior(b_datas[6])[1, 1, :, b_indices[6]]', colorrange=clim, colormap=Reverse(:RdBu_10))

display(fig)

#%%

labels = [
    "AMD, 2m resolution",
    "WENO(9), ν = κ = 0, 2m resolution",
    # "WENO(9), ν = κ = 1e-5, 2m resolution",

    "AMD, 1m resolution",
    "WENO(9), ν = κ = 0, 1m resolution",
    # "WENO(9), ν = κ = 1e-5, 1m resolution",

    "AMD, 0.5m resolution",
    "WENO(9), ν = κ = 0, 0.5m resolution",
    # "WENO(9), ν = κ = 1e-5, 0.5m resolution",
]
=#