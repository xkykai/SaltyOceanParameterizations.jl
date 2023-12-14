using JLD2
using CairoMakie
using Statistics
using Oceananigans

FILE_DIRS_2m = [
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_32_64_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_32_64_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_32_64_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_32_64_f",
]

FILE_DIRS_1m = [
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_64_128_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_64_128_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_64_128_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_64_128_f",
]

FILE_DIRS_HALFm = [
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_128_256_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_128_256_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_128_256_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_128_256_f",
]

FILE_DIRS_QUARTERm = [
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_256_512_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_256_512_f",
]

labels = [
    "AMD",
    L"WENO, $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$",
    L"WENO, $\nu$ = $\kappa$ = $1 \times 10^{-5}$ m$^{2}$ s$^{-1}$",
    "WENO + AMD",
]

parameters = jldopen("$(FILE_DIRS_2m[1])/instantaneous_timeseries.jld2", "r") do file
    return Dict([(key, file["metadata/parameters/$(key)"]) for key in keys(file["metadata/parameters"])])
end 

Qᴮ = parameters["buoyancy_flux"]
Qᵁ = parameters["momentum_flux"]

b²_spectras_2m = [load("$(FILE_DIR)/spectra.jld2", "b") for FILE_DIR in FILE_DIRS_2m]
u²_spectras_2m = [load("$(FILE_DIR)/spectra.jld2", "u") for FILE_DIR in FILE_DIRS_2m]
v²_spectras_2m = [load("$(FILE_DIR)/spectra.jld2", "v") for FILE_DIR in FILE_DIRS_2m]
w²_spectras_2m = [load("$(FILE_DIR)/spectra.jld2", "w") for FILE_DIR in FILE_DIRS_2m]

b²_spectras_1m = [load("$(FILE_DIR)/spectra.jld2", "b") for FILE_DIR in FILE_DIRS_1m]
u²_spectras_1m = [load("$(FILE_DIR)/spectra.jld2", "u") for FILE_DIR in FILE_DIRS_1m]
v²_spectras_1m = [load("$(FILE_DIR)/spectra.jld2", "v") for FILE_DIR in FILE_DIRS_1m]
w²_spectras_1m = [load("$(FILE_DIR)/spectra.jld2", "w") for FILE_DIR in FILE_DIRS_1m]

b²_spectras_halfm = [load("$(FILE_DIR)/spectra.jld2", "b") for FILE_DIR in FILE_DIRS_HALFm]
u²_spectras_halfm = [load("$(FILE_DIR)/spectra.jld2", "u") for FILE_DIR in FILE_DIRS_HALFm]
v²_spectras_halfm = [load("$(FILE_DIR)/spectra.jld2", "v") for FILE_DIR in FILE_DIRS_HALFm]
w²_spectras_halfm = [load("$(FILE_DIR)/spectra.jld2", "w") for FILE_DIR in FILE_DIRS_HALFm]

b²_spectras_quarterm = [load("$(FILE_DIR)/spectra.jld2", "b") for FILE_DIR in FILE_DIRS_QUARTERm]
u²_spectras_quarterm = [load("$(FILE_DIR)/spectra.jld2", "u") for FILE_DIR in FILE_DIRS_QUARTERm]
v²_spectras_quarterm = [load("$(FILE_DIR)/spectra.jld2", "v") for FILE_DIR in FILE_DIRS_QUARTERm]
w²_spectras_quarterm = [load("$(FILE_DIR)/spectra.jld2", "w") for FILE_DIR in FILE_DIRS_QUARTERm]

b_datas_2m = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_b.jld2", "b", backend=OnDisk()) for FILE_DIR in FILE_DIRS_2m]
u_datas_2m = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_u.jld2", "u", backend=OnDisk()) for FILE_DIR in FILE_DIRS_2m]
v_datas_2m = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_v.jld2", "v", backend=OnDisk()) for FILE_DIR in FILE_DIRS_2m]
w_datas_2m = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_w.jld2", "w", backend=OnDisk()) for FILE_DIR in FILE_DIRS_2m]

b_datas_1m = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_b.jld2", "b", backend=OnDisk()) for FILE_DIR in FILE_DIRS_1m]
u_datas_1m = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_u.jld2", "u", backend=OnDisk()) for FILE_DIR in FILE_DIRS_1m]
v_datas_1m = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_v.jld2", "v", backend=OnDisk()) for FILE_DIR in FILE_DIRS_1m]
w_datas_1m = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_w.jld2", "w", backend=OnDisk()) for FILE_DIR in FILE_DIRS_1m]

b_datas_halfm = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_b.jld2", "b", backend=OnDisk()) for FILE_DIR in FILE_DIRS_HALFm]
u_datas_halfm = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_u.jld2", "u", backend=OnDisk()) for FILE_DIR in FILE_DIRS_HALFm]
v_datas_halfm = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_v.jld2", "v", backend=OnDisk()) for FILE_DIR in FILE_DIRS_HALFm]
w_datas_halfm = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_w.jld2", "w", backend=OnDisk()) for FILE_DIR in FILE_DIRS_HALFm]

b_datas_quarterm = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_b.jld2", "b", backend=OnDisk()) for FILE_DIR in FILE_DIRS_QUARTERm]
u_datas_quarterm = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_u.jld2", "u", backend=OnDisk()) for FILE_DIR in FILE_DIRS_QUARTERm]
v_datas_quarterm = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_v.jld2", "v", backend=OnDisk()) for FILE_DIR in FILE_DIRS_QUARTERm]
w_datas_quarterm = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_w.jld2", "w", backend=OnDisk()) for FILE_DIR in FILE_DIRS_QUARTERm]

times_2m = [b_datas_2m[i].times for i in eachindex(b_datas_2m)]
times_1m = [b_datas_1m[i].times for i in eachindex(b_datas_1m)]
times_halfm = [b_datas_halfm[i].times for i in eachindex(b_datas_halfm)]
times_quarterm = [b_datas_quarterm[i].times for i in eachindex(b_datas_quarterm)]

zC_2m = b_datas_2m[1].grid.zᵃᵃᶜ[1:64]
zC_1m = b_datas_1m[1].grid.zᵃᵃᶜ[1:128]
zC_halfm = b_datas_halfm[1].grid.zᵃᵃᶜ[1:256]
zC_quarterm = b_datas_quarterm[1].grid.zᵃᵃᶜ[1:512]

n_start = 29
n_window = n_start:48

times = (0:60^2:48*60^2) ./ (24 * 60^2)

zrange = (-50, -2)

z_levels_2m = findfirst(z -> z>zrange[1], zC_2m):findlast(z -> z<zrange[2], zC_2m)
z_levels_1m = findfirst(z -> z>zrange[1], zC_1m):findlast(z -> z<zrange[2], zC_1m)
z_levels_halfm = findfirst(z -> z>zrange[1], zC_halfm):findlast(z -> z<zrange[2], zC_halfm)
z_levels_quarterm = findfirst(z -> z>zrange[1], zC_quarterm):findlast(z -> z<zrange[2], zC_quarterm)

TKE_kxs_2m = [0.5 .* (u.spectra²_kx .+ v.spectra²_kx .+ w.spectra²_kx) for (u, v, w) in zip(u²_spectras_2m, v²_spectras_2m, w²_spectras_2m)]
TKE_kys_2m = [0.5 .* (u.spectra²_ky .+ v.spectra²_ky .+ w.spectra²_ky) for (u, v, w) in zip(u²_spectras_2m, v²_spectras_2m, w²_spectras_2m)]

TKE_kxs_1m = [0.5 .* (u.spectra²_kx .+ v.spectra²_kx .+ w.spectra²_kx) for (u, v, w) in zip(u²_spectras_1m, v²_spectras_1m, w²_spectras_1m)]
TKE_kys_1m = [0.5 .* (u.spectra²_ky .+ v.spectra²_ky .+ w.spectra²_ky) for (u, v, w) in zip(u²_spectras_1m, v²_spectras_1m, w²_spectras_1m)]

TKE_kxs_halfm = [0.5 .* (u.spectra²_kx .+ v.spectra²_kx .+ w.spectra²_kx) for (u, v, w) in zip(u²_spectras_halfm, v²_spectras_halfm, w²_spectras_halfm)]
TKE_kys_halfm = [0.5 .* (u.spectra²_ky .+ v.spectra²_ky .+ w.spectra²_ky) for (u, v, w) in zip(u²_spectras_halfm, v²_spectras_halfm, w²_spectras_halfm)]

TKE_kxs_quarterm = [0.5 .* (u.spectra²_kx .+ v.spectra²_kx .+ w.spectra²_kx) for (u, v, w) in zip(u²_spectras_quarterm, v²_spectras_quarterm, w²_spectras_quarterm)]
TKE_kys_quarterm = [0.5 .* (u.spectra²_ky .+ v.spectra²_ky .+ w.spectra²_ky) for (u, v, w) in zip(u²_spectras_quarterm, v²_spectras_quarterm, w²_spectras_quarterm)]

#%%
with_theme(theme_latexfonts()) do
    fig = Figure(size=(1200, 1200))

    axkx_b_2m = Axis(fig[1, 1], xscale=log10, yscale=log10)
    axky_b_2m = Axis(fig[3, 1], xscale=log10, yscale=log10)
    axkx_TKE_2m = Axis(fig[5, 1], xscale=log10, yscale=log10)
    axky_TKE_2m = Axis(fig[7, 1], xscale=log10, yscale=log10)
    
    axkx_b_1m = Axis(fig[1, 2], xscale=log10, yscale=log10)
    axky_b_1m = Axis(fig[3, 2], xscale=log10, yscale=log10)
    axkx_TKE_1m = Axis(fig[5, 2], xscale=log10, yscale=log10)
    axky_TKE_1m = Axis(fig[7, 2], xscale=log10, yscale=log10)

    axkx_b_halfm = Axis(fig[1, 3], xscale=log10, yscale=log10)
    axky_b_halfm = Axis(fig[3, 3], xscale=log10, yscale=log10)
    axkx_TKE_halfm = Axis(fig[5, 3], xscale=log10, yscale=log10)
    axky_TKE_halfm = Axis(fig[7, 3], xscale=log10, yscale=log10)

    axkx_b_quarterm = Axis(fig[1, 4], xscale=log10, yscale=log10)
    axky_b_quarterm = Axis(fig[3, 4], xscale=log10, yscale=log10)
    axkx_TKE_quarterm = Axis(fig[5, 4], xscale=log10, yscale=log10)
    axky_TKE_quarterm = Axis(fig[7, 4], xscale=log10, yscale=log10)
    
    Label(fig[0, 1], "Resolution = 2m", tellwidth=false, font=:bold)
    Label(fig[0, 2], "Resolution = 1m", tellwidth=false, font=:bold)
    Label(fig[0, 3], "Resolution = 0.5m", tellwidth=false, font=:bold)
    Label(fig[0, 4], "Resolution = 0.25m", tellwidth=false, font=:bold)

    Label(fig[1:4, 0], "Buoyancy variance spectra (m³ s⁻⁴)", rotation=π/2, font=:bold)
    Label(fig[5:8, 0], "Turbulent kinetic energy spectra (m³ s⁻²)", rotation=π/2, font=:bold)

    Label(fig[2, 1:4], L"k_x (\mathrm{m}^{-1})", tellwidth=false)
    Label(fig[4, 1:4], L"k_y (\mathrm{m}^{-1})", tellwidth=false)
    Label(fig[6, 1:4], L"k_x (\mathrm{m}^{-1})", tellwidth=false)
    Label(fig[8, 1:4], L"k_y (\mathrm{m}^{-1})", tellwidth=false)

    Label(fig[1:2, -1], "Averaged in y", rotation=π/2, tellheight=false, font=:bold)
    Label(fig[3:4, -1], "Averaged in x", rotation=π/2, tellheight=false, font=:bold)
    Label(fig[5:6, -1], "Averaged in y", rotation=π/2, tellheight=false, font=:bold)
    Label(fig[7:8, -1], "Averaged in x", rotation=π/2, tellheight=false, font=:bold)

    for i in eachindex(b²_spectras_2m)
        lines!(axkx_b_2m, b²_spectras_2m[i].kx[2:end], mean(b²_spectras_2m[i].spectra²_kx[2:end, :, z_levels_2m, n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(b²_spectras_2m)
        lines!(axky_b_2m, b²_spectras_2m[i].ky[2:end], mean(b²_spectras_2m[i].spectra²_ky[:, 2:end, z_levels_2m, n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(TKE_kxs_2m)
        lines!(axkx_TKE_2m, b²_spectras_2m[i].kx[2:end], mean(TKE_kxs_2m[i][2:end, :, z_levels_2m, n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(TKE_kys_2m)
        lines!(axky_TKE_2m, b²_spectras_2m[i].ky[2:end], mean(TKE_kys_2m[i][:, 2:end, z_levels_2m, n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(b²_spectras_1m)
        lines!(axkx_b_1m, b²_spectras_1m[i].kx[2:end], mean(b²_spectras_1m[i].spectra²_kx[2:end, :, z_levels_1m, n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(b²_spectras_1m)
        lines!(axky_b_1m, b²_spectras_1m[i].ky[2:end], mean(b²_spectras_1m[i].spectra²_ky[:, 2:end, z_levels_1m, n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(TKE_kxs_1m)
        lines!(axkx_TKE_1m, b²_spectras_1m[i].kx[2:end], mean(TKE_kxs_1m[i][2:end, :, z_levels_1m, n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(TKE_kys_1m)
        lines!(axky_TKE_1m, b²_spectras_1m[i].ky[2:end], mean(TKE_kys_1m[i][:, 2:end, z_levels_1m, n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(b²_spectras_halfm)
        lines!(axkx_b_halfm, b²_spectras_halfm[i].kx[2:end], mean(b²_spectras_halfm[i].spectra²_kx[2:end, :, z_levels_halfm, n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(b²_spectras_halfm)
        lines!(axky_b_halfm, b²_spectras_halfm[i].ky[2:end], mean(b²_spectras_halfm[i].spectra²_ky[:, 2:end, z_levels_halfm, n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(TKE_kxs_halfm)
        lines!(axkx_TKE_halfm, b²_spectras_halfm[i].kx[2:end], mean(TKE_kxs_halfm[i][2:end, :, z_levels_halfm, n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(TKE_kys_halfm)
        lines!(axky_TKE_halfm, b²_spectras_halfm[i].ky[2:end], mean(TKE_kys_halfm[i][:, 2:end, z_levels_halfm, n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(b²_spectras_quarterm)
        lines!(axkx_b_quarterm, b²_spectras_quarterm[i].kx[2:end], mean(b²_spectras_quarterm[i].spectra²_kx[2:end, :, z_levels_quarterm, n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(b²_spectras_quarterm)
        lines!(axky_b_quarterm, b²_spectras_quarterm[i].ky[2:end], mean(b²_spectras_quarterm[i].spectra²_ky[:, 2:end, z_levels_quarterm, n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(TKE_kxs_quarterm)
        lines!(axkx_TKE_quarterm, b²_spectras_quarterm[i].kx[2:end], mean(TKE_kxs_quarterm[i][2:end, :, z_levels_quarterm, n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(TKE_kys_quarterm)
        lines!(axky_TKE_quarterm, b²_spectras_quarterm[i].ky[2:end], mean(TKE_kys_quarterm[i][:, 2:end, z_levels_quarterm, n_window], dims=(3, 4))[:], label=labels[i])
    end


    hidedecorations!(axkx_b_2m, ticks=false, ticklabels=false)
    hidedecorations!(axky_b_2m, ticks=false, ticklabels=false)
    hidedecorations!(axkx_TKE_2m, ticks=false, ticklabels=false)
    hidedecorations!(axky_TKE_2m, ticks=false, ticklabels=false)

    hidedecorations!(axkx_b_1m, ticks=false, ticklabels=false)
    hidedecorations!(axky_b_1m, ticks=false, ticklabels=false)
    hidedecorations!(axkx_TKE_1m, ticks=false, ticklabels=false)
    hidedecorations!(axky_TKE_1m, ticks=false, ticklabels=false)

    hidedecorations!(axkx_b_halfm, ticks=false, ticklabels=false)
    hidedecorations!(axky_b_halfm, ticks=false, ticklabels=false)
    hidedecorations!(axkx_TKE_halfm, ticks=false, ticklabels=false)
    hidedecorations!(axky_TKE_halfm, ticks=false, ticklabels=false)

    hidedecorations!(axkx_b_quarterm, ticks=false, ticklabels=false)
    hidedecorations!(axky_b_quarterm, ticks=false, ticklabels=false)
    hidedecorations!(axkx_TKE_quarterm, ticks=false, ticklabels=false)
    hidedecorations!(axky_TKE_quarterm, ticks=false, ticklabels=false)

    Legend(fig[9, :], axkx_b_2m, orientation=:horizontal)

    display(fig)
    save("./Data/spectra_resolution_closure.png", fig, px_per_unit=8)
end
#%%





