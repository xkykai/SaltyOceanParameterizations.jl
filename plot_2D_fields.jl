using Oceananigans
using CairoMakie

FILE_DIRs = [
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_32_64_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_32_64_f",

    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_64_128_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_64_128_f",

    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_128_256_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_128_256_f",

    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_256_512_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_256_512_f",
  ]

w_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_w.jld2", "w", backend=OnDisk()) for FILE_DIR in FILE_DIRs]

t = 48 * 3600

ns = [findfirst(x -> x ≈ t, data.times) for data in w_datas]

Nxs = [data.grid.Nx for data in w_datas]
Nys = [data.grid.Ny for data in w_datas]
Nzs = [data.grid.Nz for data in w_datas]

xCs = [data.grid.xᶜᵃᵃ[1:Nx] for (data, Nx) in zip(w_datas, Nxs)]
yCs = [data.grid.yᵃᶜᵃ[1:Ny] for (data, Ny) in zip(w_datas, Nys)]
zCs = [data.grid.zᵃᵃᶜ[1:Nz] for (data, Nz) in zip(w_datas, Nzs)]
zFs = [data.grid.zᵃᵃᶠ[1:Nz+1] for (data, Nz) in zip(w_datas, Nzs)]

function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

w_xzs = [interior(data[ns[i]], :, 1, :) for (i, data) in enumerate(w_datas)]

wlims = [(-maximum(abs, w_xz), maximum(abs, w_xz)) for w_xz in w_xzs]

w_colormap = :balance

with_theme(theme_latexfonts()) do
    fig = Figure(size=(2000, 2000))

    axw1 = Axis(fig[1, 4], title="AMD, 2m resolution", xlabel="x (m)", ylabel="z (m)")
    axw2 = Axis(fig[2, 4], title=L"WENO, $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 2m resolution", xlabel="x (m)", ylabel="z (m)")
    axw3 = Axis(fig[1, 3], title="AMD, 1m resolution", xlabel="x (m)", ylabel="z (m)")
    axw4 = Axis(fig[2, 3], title=L"WENO, $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 1m resolution", xlabel="x (m)", ylabel="z (m)")
    axw5 = Axis(fig[1, 2], title="AMD, 0.5m resolution", xlabel="x (m)", ylabel="z (m)")
    axw6 = Axis(fig[2, 2], title=L"WENO, $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 0.5m resolution", xlabel="x (m)", ylabel="z (m)")
    axw7 = Axis(fig[1, 1], title="AMD, 0.25m resolution", xlabel="x (m)", ylabel="z (m)")
    axw8 = Axis(fig[2, 1], title=L"WENO, $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 0.25m resolution", xlabel="x (m)", ylabel="z (m)")

    heatmap!(axw1, xCs[1], zCs[1], w_xzs[1], colormap=w_colormap, colorrange=wlims[1])
    heatmap!(axw2, xCs[2], zCs[2], w_xzs[2], colormap=w_colormap, colorrange=wlims[2])
    heatmap!(axw3, xCs[3], zCs[3], w_xzs[3], colormap=w_colormap, colorrange=wlims[3])
    heatmap!(axw4, xCs[4], zCs[4], w_xzs[4], colormap=w_colormap, colorrange=wlims[4])
    heatmap!(axw5, xCs[5], zCs[5], w_xzs[5], colormap=w_colormap, colorrange=wlims[5])
    heatmap!(axw6, xCs[6], zCs[6], w_xzs[6], colormap=w_colormap, colorrange=wlims[6])
    heatmap!(axw7, xCs[7], zCs[7], w_xzs[7], colormap=w_colormap, colorrange=wlims[7])
    heatmap!(axw8, xCs[8], zCs[8], w_xzs[8], colormap=w_colormap, colorrange=wlims[8])

    # Colorbar(fig[1,4], b1_xy_surface, label="Buoyancy (m² s⁻³)")

    display(fig)
    Label(fig[0, :], "Vertical velocity fields comparison", font=:bold)
    save("./Data/w_2D_fields.png", fig, px_per_unit=8)
end