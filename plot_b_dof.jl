using Oceananigans
using CairoMakie

FILE_DIRS = [
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_256_512_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_256_512_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_128_256_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_128_256_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_64_128_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_64_128_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_32_64_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_32_64_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_128_256_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_64_128_f",

    "./LES/linearb_2layer_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_AMD_Lxz_128.0_128.0_Nxz_128_128_f",
    "./LES/linearb_2layer_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_WENO9nu0_Lxz_128.0_128.0_Nxz_64_64_f",
    "./LES/linearb_2layer_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_WENO9nu0_Lxz_128.0_128.0_Nxz_128_128_f",
    "./LES/linearb_2layer_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_AMD_Lxz_128.0_128.0_Nxz_256_256_f",
    "./LES/linearb_2layer_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_WENO9nu0_Lxz_128.0_128.0_Nxz_256_256_f",
]

labels = [
    # "Centered 2nd Order + AMD, 0.25m resolution", 
    # L"WENO + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 0.25m resolution",
    # "AMD, 0.5m resolution",
    # "WENO + AMD, 0.5m resolution",
    # "AMD, 1m resolution",
    # "WENO + AMD, 1m resolution",
    # "AMD, 2m resolution",
    # "WENO + AMD, 2m resolution",
    # L"WENO, $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 0.5m resolution",
    # L"WENO, $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 1m resolution",

    "Centered 2nd Order + AMD, 1m resolution", 
    L"WENO + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 2m resolution",

    L"WENO + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 1m resolution",

    "Centered 2nd Order + AMD, 0.5m resolution", 
    L"WENO + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 0.5m resolution",
]

bbar_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar") for FILE_DIR in FILE_DIRS]

Nzs = [data.grid.Nz for data in bbar_datas]
zCs = [bbar_data.grid.zᵃᵃᶜ[1:Nz] for (bbar_data, Nz) in zip(bbar_datas, Nzs)]

times = bbar_datas[1].times
plot_time = 2 * 24 * 3600
ns = [findfirst(x -> x ≈ plot_time, data.times) for data in bbar_datas]
#%%
with_theme(theme_latexfonts()) do
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], xlabel="Buoyancy (m s⁻²)", ylabel="z (m)")

    for (i, data) in enumerate(bbar_datas)
        lines!(ax, interior(data[ns[i]], 1, 1, :), zCs[i], label=labels[i])
    end

    axislegend(ax, position=:rb)
    # heatmap!(axw8, xCs[8], zCs[8], w_xzs[8], colormap=w_colormap)

    # Colorbar(fig[1,4], b1_xy_surface, label="Buoyancy (m² s⁻³)")
    hidedecorations!(ax, minorgrid=false, ticks=false, label=false, ticklabels=false)
    # hidedecorations!(axWENO9NU0, minorgrid=false, ticks=false, label=false, ticklabels=false)

    display(fig)
    # Label(fig[0, :], "Vertical velocity fields comparison", font=:bold)
    # save("./Data/b_dof_resolution_AMD_WENO9AMD.png", fig, px_per_unit=8)
    save("./Data/b_quarterdomain_AMD_WENO_dof.png", fig, px_per_unit=8)
end
#%%