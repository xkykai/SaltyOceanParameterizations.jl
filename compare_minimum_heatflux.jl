using Oceananigans
using JLD2
using CairoMakie
using Statistics

FILE_DIRS = [
    # "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_AMD_Lxz_2.0_1.0_Nxz_256_128",
    # "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_WENO9nu0_Lxz_2.0_1.0_Nxz_256_128",
    # "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_AMD_Lxz_2.0_1.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_WENO9nu0_Lxz_2.0_1.0_Nxz_128_64",

    "./LES/linearb_experiments_dbdz_0.0001_QU_0.0_QB_4.0e-7_b_0.0_AMD_Lxz_256.0_128.0_Nxz_256_128",
    "./LES/linearb_experiments_dbdz_0.0001_QU_0.0_QB_4.0e-7_b_0.0_WENO9nu0_Lxz_256.0_128.0_Nxz_256_128",
]

labels = [
    # "Centered 2nd Order + AMD",
    # L"WENO + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$",
    # "Centered 2nd Order + AMD, 1/64 m resolution",
    # L"WENO + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 1/64 m resolution",

    "Centered 2nd Order + AMD, 1 m resolution",
    L"WENO + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 1 m resolution",
]

parameters = jldopen("$(FILE_DIRS[1])/instantaneous_timeseries.jld2", "r") do file
    return Dict([(key, file["metadata/$(key)"]) for key in keys(file["metadata"])])
end 

Qᴮ = parameters["buoyancy_flux"]
dbdz = parameters["buoyancy_gradient"]

wb_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wb") for FILE_DIR in FILE_DIRS]

Nxs = [size(data.grid)[1] for data in wb_datas]
Nys = [size(data.grid)[2] for data in wb_datas]
Nzs = [size(data.grid)[3] for data in wb_datas]

zCs = [data.grid.zᵃᵃᶜ[1:Nzs[i]] for (i, data) in enumerate(wb_datas)]
zFs = [data.grid.zᵃᵃᶠ[1:Nzs[i]+1] for (i, data) in enumerate(wb_datas)]

h_indices = [argmin(data, dims=3) for data in wb_datas]
wb_mins = [data[h_index] for (data, h_index) in zip(wb_datas, h_indices)]

hs = [[zF[h_index[1, 1, 1, i][3]] for i in axes(h_index, 4)] for (zF, h_index) in zip(zFs, h_indices)]

zs_rescaled = [[zF ./ z for z in h] for (zF, h) in zip(zFs, hs)]
wbs_rescaled = [[interior(wb[i], 1, 1, :) ./ Qᴮ for i in 1:size(wb, 4)] for wb in wb_datas]

wb_ratios = [[wb ./ Qᴮ for wb in wb_min] for wb_min in wb_mins]

mean_wb_ratios = [mean(wb_ratio[50:end]) for wb_ratio in wb_ratios]
#%%
with_theme(theme_latexfonts()) do
    fig = Figure()
    ax = Axis(fig[1, 1], xticks=-0.5:0.1:1, xlabel="wb / wb(surface)", ylabel="z/h", title="Free convection")

    for i in 50:length(wbs_rescaled[2])-100
        lines!(ax, wbs_rescaled[2][i][1:end-1], zs_rescaled[2][i][1:end-1], color=:salmon, alpha=0.1, linewidth=2)
    end

    for i in 50:length(wbs_rescaled[1])-100
        lines!(ax, wbs_rescaled[1][i][1:end-1], zs_rescaled[1][i][1:end-1], color=:royalblue, alpha=0.1, linewidth=2)
    end

    vlines!([mean_wb_ratios[1]], color=:royalblue, linewidth=2, linestyle=:dash, label="$(labels[1]), <wb / wb(0)> = $(round(mean_wb_ratios[1], digits=3))")
    vlines!([mean_wb_ratios[2]], color=:salmon, linewidth=2, label=L"WENO + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, <wb / wb(0)> = %$(round(mean_wb_ratios[2], digits=3))")

    xlims!(ax, (-0.5, 1))
    ylims!(ax, (0, 1.5))
    axislegend(ax, loc=:tr)
    display(fig)

    save("./Data/freeconvection_2.8_QB_$(Qᴮ)_dbdz_$(dbdz)_wb_ratio.png", fig, px_per_unit=8)
end
#%%

#%%