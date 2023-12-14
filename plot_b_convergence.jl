using CairoMakie
using Oceananigans

FILE_DIR_AMD = [
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_32_64_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_64_128_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_128_256_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_256_512_f"
]

FILE_DIR_WENO9NU0 = [
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_32_64_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_64_128_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_128_256_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_256_512_f"
]

labels = ["2m resolution", "1m resolution", "0.5m resolution", "0.25m resolution"]

bbar_datas_AMD = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar") for FILE_DIR in FILE_DIR_AMD]
bbar_datas_WENO9NU0 = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar") for FILE_DIR in FILE_DIR_WENO9NU0]

Nzs = [data.grid.Nz for data in bbar_datas_AMD]
zCs_AMD = [bbar_data.grid.zᵃᵃᶜ[1:Nz] for (bbar_data, Nz) in zip(bbar_datas_AMD, Nzs)]
zCs_WENO9NU0 = [bbar_data.grid.zᵃᵃᶜ[1:Nz] for (bbar_data, Nz) in zip(bbar_datas_WENO9NU0, Nzs)]

times = bbar_datas_AMD[1].times
plot_time = 2 * 24 * 3600
ns_AMD = [findfirst(x -> x ≈ plot_time, data.times) for data in bbar_datas_AMD]
ns_WENO9NU0 = [findfirst(x -> x ≈ plot_time, data.times) for data in bbar_datas_WENO9NU0]
#%%
with_theme(theme_latexfonts()) do
    fig = Figure(size=(1100, 600))
    axAMD = Axis(fig[1, 1], xlabel="Buoyancy (m s⁻²)", ylabel="z (m)", title="AMD")
    axWENO9NU0 = Axis(fig[1, 2], xlabel="Buoyancy (m s⁻²)", ylabel="z (m)", title=L"WENO, $\nu$ = $\kappa$ = 0 m² s$^{-1}$")

    for (i, data) in enumerate(bbar_datas_AMD)
        lines!(axAMD, interior(data[ns_AMD[i]], 1, 1, :), zCs_AMD[i], label=labels[i])
    end

    for (i, data) in enumerate(bbar_datas_WENO9NU0)
        lines!(axWENO9NU0, interior(data[ns_WENO9NU0[i]], 1, 1, :), zCs_WENO9NU0[i], label=labels[i])
    end

    Legend(fig[2, :], axAMD, orientation=:horizontal)
    # heatmap!(axw8, xCs[8], zCs[8], w_xzs[8], colormap=w_colormap)

    # Colorbar(fig[1,4], b1_xy_surface, label="Buoyancy (m² s⁻³)")
    hidedecorations!(axAMD, minorgrid=false, ticks=false, label=false, ticklabels=false)
    hidedecorations!(axWENO9NU0, minorgrid=false, ticks=false, label=false, ticklabels=false)

    display(fig)
    # Label(fig[0, :], "Vertical velocity fields comparison", font=:bold)
    save("./Data/b_convergenge_AMD_WENO9nu0.png", fig, px_per_unit=8)
end
#%%