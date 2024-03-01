using JLD2
using CairoMakie
using Statistics
using Oceananigans

FILE_DIRS_AMD = [
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_32_64_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_64_128_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_128_256_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_256_512_f",
    

]

FILE_DIRS_WENO9nu0 = [
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_32_64_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_64_128_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_128_256_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_256_512_f",
]

FILE_DIRS_WENO9nu1eminus5 = [
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_32_64_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_64_128_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_128_256_f",
]

FILE_DIRS_WENO9AMD = [
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_32_64_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_64_128_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_128_256_f",
]

labels = [
    # "Centered 2nd Order + AMD",
    # L"WENO, $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$",
    # L"WENO, $\nu$ = $\kappa$ = $1 \times 10^{-5}$ m$^{2}$ s$^{-1}$",
    # "WENO + AMD",
    "2m resolution",
    "1m resolution",
    "0.5m resolution",
    "0.25m resolution"
]

parameters = jldopen("$(FILE_DIRS_AMD[1])/instantaneous_timeseries.jld2", "r") do file
    return Dict([(key, file["metadata/parameters/$(key)"]) for key in keys(file["metadata/parameters"])])
end 

Qᴮ = parameters["buoyancy_flux"]
Qᵁ = parameters["momentum_flux"]

b²_spectras_AMD = [load("$(FILE_DIR)/spectra.jld2", "b") for FILE_DIR in FILE_DIRS_AMD]
u²_spectras_AMD = [load("$(FILE_DIR)/spectra.jld2", "u") for FILE_DIR in FILE_DIRS_AMD]
v²_spectras_AMD = [load("$(FILE_DIR)/spectra.jld2", "v") for FILE_DIR in FILE_DIRS_AMD]
w²_spectras_AMD = [load("$(FILE_DIR)/spectra.jld2", "w") for FILE_DIR in FILE_DIRS_AMD]

b²_spectras_WENO9nu0 = [load("$(FILE_DIR)/spectra.jld2", "b") for FILE_DIR in FILE_DIRS_WENO9nu0]
u²_spectras_WENO9nu0 = [load("$(FILE_DIR)/spectra.jld2", "u") for FILE_DIR in FILE_DIRS_WENO9nu0]
v²_spectras_WENO9nu0 = [load("$(FILE_DIR)/spectra.jld2", "v") for FILE_DIR in FILE_DIRS_WENO9nu0]
w²_spectras_WENO9nu0 = [load("$(FILE_DIR)/spectra.jld2", "w") for FILE_DIR in FILE_DIRS_WENO9nu0]

b²_spectras_WENO9nu1eminus5 = [load("$(FILE_DIR)/spectra.jld2", "b") for FILE_DIR in FILE_DIRS_WENO9nu1eminus5]
u²_spectras_WENO9nu1eminus5 = [load("$(FILE_DIR)/spectra.jld2", "u") for FILE_DIR in FILE_DIRS_WENO9nu1eminus5]
v²_spectras_WENO9nu1eminus5 = [load("$(FILE_DIR)/spectra.jld2", "v") for FILE_DIR in FILE_DIRS_WENO9nu1eminus5]
w²_spectras_WENO9nu1eminus5 = [load("$(FILE_DIR)/spectra.jld2", "w") for FILE_DIR in FILE_DIRS_WENO9nu1eminus5]

b²_spectras_WENO9AMD = [load("$(FILE_DIR)/spectra.jld2", "b") for FILE_DIR in FILE_DIRS_WENO9AMD]
u²_spectras_WENO9AMD = [load("$(FILE_DIR)/spectra.jld2", "u") for FILE_DIR in FILE_DIRS_WENO9AMD]
v²_spectras_WENO9AMD = [load("$(FILE_DIR)/spectra.jld2", "v") for FILE_DIR in FILE_DIRS_WENO9AMD]
w²_spectras_WENO9AMD = [load("$(FILE_DIR)/spectra.jld2", "w") for FILE_DIR in FILE_DIRS_WENO9AMD]

b_datas_AMD = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_b.jld2", "b", backend=OnDisk()) for FILE_DIR in FILE_DIRS_AMD]
u_datas_AMD = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_u.jld2", "u", backend=OnDisk()) for FILE_DIR in FILE_DIRS_AMD]
v_datas_AMD = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_v.jld2", "v", backend=OnDisk()) for FILE_DIR in FILE_DIRS_AMD]
w_datas_AMD = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_w.jld2", "w", backend=OnDisk()) for FILE_DIR in FILE_DIRS_AMD]

b_datas_WENO9nu0 = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_b.jld2", "b", backend=OnDisk()) for FILE_DIR in FILE_DIRS_WENO9nu0]
u_datas_WENO9nu0 = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_u.jld2", "u", backend=OnDisk()) for FILE_DIR in FILE_DIRS_WENO9nu0]
v_datas_WENO9nu0 = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_v.jld2", "v", backend=OnDisk()) for FILE_DIR in FILE_DIRS_WENO9nu0]
w_datas_WENO9nu0 = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_w.jld2", "w", backend=OnDisk()) for FILE_DIR in FILE_DIRS_WENO9nu0]

b_datas_WENO9nu1eminus5 = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_b.jld2", "b", backend=OnDisk()) for FILE_DIR in FILE_DIRS_WENO9nu1eminus5]
u_datas_WENO9nu1eminus5 = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_u.jld2", "u", backend=OnDisk()) for FILE_DIR in FILE_DIRS_WENO9nu1eminus5]
v_datas_WENO9nu1eminus5 = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_v.jld2", "v", backend=OnDisk()) for FILE_DIR in FILE_DIRS_WENO9nu1eminus5]
w_datas_WENO9nu1eminus5 = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_w.jld2", "w", backend=OnDisk()) for FILE_DIR in FILE_DIRS_WENO9nu1eminus5]

b_datas_WENO9AMD = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_b.jld2", "b", backend=OnDisk()) for FILE_DIR in FILE_DIRS_WENO9AMD]
u_datas_WENO9AMD = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_u.jld2", "u", backend=OnDisk()) for FILE_DIR in FILE_DIRS_WENO9AMD]
v_datas_WENO9AMD = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_v.jld2", "v", backend=OnDisk()) for FILE_DIR in FILE_DIRS_WENO9AMD]
w_datas_WENO9AMD = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_w.jld2", "w", backend=OnDisk()) for FILE_DIR in FILE_DIRS_WENO9AMD]

times_AMD = [b_datas_AMD[i].times for i in eachindex(b_datas_AMD)]
times_WENO9nu0 = [b_datas_WENO9nu0[i].times for i in eachindex(b_datas_WENO9nu0)]
times_WENO9nu1eminus5 = [b_datas_WENO9nu1eminus5[i].times for i in eachindex(b_datas_WENO9nu1eminus5)]
times_WENO9AMD = [b_datas_WENO9AMD[i].times for i in eachindex(b_datas_WENO9AMD)]

Nzs_AMD = [b_datas_AMD[i].grid.Nz for i in eachindex(b_datas_AMD)]
Nzs_WENO9nu0 = [b_datas_WENO9nu0[i].grid.Nz for i in eachindex(b_datas_WENO9nu0)]
Nzs_WENO9nu1eminus5 = [b_datas_WENO9nu1eminus5[i].grid.Nz for i in eachindex(b_datas_WENO9nu1eminus5)]
Nzs_WENO9AMD = [b_datas_WENO9AMD[i].grid.Nz for i in eachindex(b_datas_WENO9AMD)]

zCs_AMD = [b_datas_AMD[i].grid.zᵃᵃᶜ[1:Nzs_AMD[i]] for i in eachindex(b_datas_AMD)]
zCs_WENO9nu0 = [b_datas_WENO9nu0[i].grid.zᵃᵃᶜ[1:Nzs_WENO9nu0[i]] for i in eachindex(b_datas_WENO9nu0)]
zCs_WENO9nu1eminus5 = [b_datas_WENO9nu1eminus5[i].grid.zᵃᵃᶜ[1:Nzs_WENO9nu1eminus5[i]] for i in eachindex(b_datas_WENO9nu1eminus5)]
zCs_WENO9AMD = [b_datas_WENO9AMD[i].grid.zᵃᵃᶜ[1:Nzs_WENO9AMD[i]] for i in eachindex(b_datas_WENO9AMD)]

n_start = 29
n_window = n_start:48

times = (0:60^2:48*60^2) ./ (24 * 60^2)

zrange = (-50, -2)

z_levels_AMD = [findfirst(z -> z>zrange[1], zC_AMD):findlast(z -> z<zrange[2], zC_AMD) for zC_AMD in zCs_AMD]
z_levels_WENO9nu0 = [findfirst(z -> z>zrange[1], zC_WENO9nu0):findlast(z -> z<zrange[2], zC_WENO9nu0) for zC_WENO9nu0 in zCs_WENO9nu0]
z_levels_WENO9nu1eminus5 = [findfirst(z -> z>zrange[1], zC_WENO9nu1eminus5):findlast(z -> z<zrange[2], zC_WENO9nu1eminus5) for zC_WENO9nu1eminus5 in zCs_WENO9nu1eminus5]
z_levels_WENO9AMD = [findfirst(z -> z>zrange[1], zC_WENO9AMD):findlast(z -> z<zrange[2], zC_WENO9AMD) for zC_WENO9AMD in zCs_WENO9AMD]

TKE_kxs_AMD = [0.5 .* (u.spectra²_kx .+ v.spectra²_kx .+ w.spectra²_kx) for (u, v, w) in zip(u²_spectras_AMD, v²_spectras_AMD, w²_spectras_AMD)]
TKE_kys_AMD = [0.5 .* (u.spectra²_ky .+ v.spectra²_ky .+ w.spectra²_ky) for (u, v, w) in zip(u²_spectras_AMD, v²_spectras_AMD, w²_spectras_AMD)]

TKE_kxs_WENO9nu0 = [0.5 .* (u.spectra²_kx .+ v.spectra²_kx .+ w.spectra²_kx) for (u, v, w) in zip(u²_spectras_WENO9nu0, v²_spectras_WENO9nu0, w²_spectras_WENO9nu0)]
TKE_kys_WENO9nu0 = [0.5 .* (u.spectra²_ky .+ v.spectra²_ky .+ w.spectra²_ky) for (u, v, w) in zip(u²_spectras_WENO9nu0, v²_spectras_WENO9nu0, w²_spectras_WENO9nu0)]

TKE_kxs_WENO9nu1eminus5 = [0.5 .* (u.spectra²_kx .+ v.spectra²_kx .+ w.spectra²_kx) for (u, v, w) in zip(u²_spectras_WENO9nu1eminus5, v²_spectras_WENO9nu1eminus5, w²_spectras_WENO9nu1eminus5)]
TKE_kys_WENO9nu1eminus5 = [0.5 .* (u.spectra²_ky .+ v.spectra²_ky .+ w.spectra²_ky) for (u, v, w) in zip(u²_spectras_WENO9nu1eminus5, v²_spectras_WENO9nu1eminus5, w²_spectras_WENO9nu1eminus5)]

TKE_kxs_WENO9AMD = [0.5 .* (u.spectra²_kx .+ v.spectra²_kx .+ w.spectra²_kx) for (u, v, w) in zip(u²_spectras_WENO9AMD, v²_spectras_WENO9AMD, w²_spectras_WENO9AMD)]
TKE_kys_WENO9AMD = [0.5 .* (u.spectra²_ky .+ v.spectra²_ky .+ w.spectra²_ky) for (u, v, w) in zip(u²_spectras_WENO9AMD, v²_spectras_WENO9AMD, w²_spectras_WENO9AMD)]

#%%
with_theme(theme_latexfonts()) do
    fig = Figure(size=(1200, 1200))

    axkx_b_AMD = Axis(fig[1, 1], xscale=log10, yscale=log10)
    axky_b_AMD = Axis(fig[3, 1], xscale=log10, yscale=log10)
    axkx_TKE_AMD = Axis(fig[5, 1], xscale=log10, yscale=log10)
    axky_TKE_AMD = Axis(fig[7, 1], xscale=log10, yscale=log10)
    
    axkx_b_WENO9nu0 = Axis(fig[1, 2], xscale=log10, yscale=log10)
    axky_b_WENO9nu0 = Axis(fig[3, 2], xscale=log10, yscale=log10)
    axkx_TKE_WENO9nu0 = Axis(fig[5, 2], xscale=log10, yscale=log10)
    axky_TKE_WENO9nu0 = Axis(fig[7, 2], xscale=log10, yscale=log10)

    axkx_b_WENO91eminus5 = Axis(fig[1, 3], xscale=log10, yscale=log10)
    axky_b_WENO91eminus5 = Axis(fig[3, 3], xscale=log10, yscale=log10)
    axkx_TKE_WENO91eminus5 = Axis(fig[5, 3], xscale=log10, yscale=log10)
    axky_TKE_WENO91eminus5 = Axis(fig[7, 3], xscale=log10, yscale=log10)

    axkx_b_WENO9AMD = Axis(fig[1, 4], xscale=log10, yscale=log10)
    axky_b_WENO9AMD = Axis(fig[3, 4], xscale=log10, yscale=log10)
    axkx_TKE_WENO9AMD = Axis(fig[5, 4], xscale=log10, yscale=log10)
    axky_TKE_WENO9AMD = Axis(fig[7, 4], xscale=log10, yscale=log10)
    
    Label(fig[0, 1], "Centered 2nd Order + AMD", tellwidth=false, font=:bold)
    Label(fig[0, 2], L"WENO, $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$", tellwidth=false, font=:bold)
    Label(fig[0, 3], L"WENO, $\nu$ = $\kappa$ = $1 \times 10^{-5}$ m$^{2}$ s$^{-1}$", tellwidth=false, font=:bold)
    Label(fig[0, 4], "WENO + AMD", tellwidth=false, font=:bold)

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

    for i in eachindex(b²_spectras_AMD)
        lines!(axkx_b_AMD, b²_spectras_AMD[i].kx[2:end], mean(b²_spectras_AMD[i].spectra²_kx[2:end, :, z_levels_AMD[i], n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(b²_spectras_AMD)
        lines!(axky_b_AMD, b²_spectras_AMD[i].ky[2:end], mean(b²_spectras_AMD[i].spectra²_ky[:, 2:end, z_levels_AMD[i], n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(TKE_kxs_AMD)
        lines!(axkx_TKE_AMD, b²_spectras_AMD[i].kx[2:end], mean(TKE_kxs_AMD[i][2:end, :, z_levels_AMD[i], n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(TKE_kys_AMD)
        lines!(axky_TKE_AMD, b²_spectras_AMD[i].ky[2:end], mean(TKE_kys_AMD[i][:, 2:end, z_levels_AMD[i], n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(b²_spectras_WENO9nu0)
        lines!(axkx_b_WENO9nu0, b²_spectras_WENO9nu0[i].kx[2:end], mean(b²_spectras_WENO9nu0[i].spectra²_kx[2:end, :, z_levels_WENO9nu0[i], n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(b²_spectras_WENO9nu0)
        lines!(axky_b_WENO9nu0, b²_spectras_WENO9nu0[i].ky[2:end], mean(b²_spectras_WENO9nu0[i].spectra²_ky[:, 2:end, z_levels_WENO9nu0[i], n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(TKE_kxs_WENO9nu0)
        lines!(axkx_TKE_WENO9nu0, b²_spectras_WENO9nu0[i].kx[2:end], mean(TKE_kxs_WENO9nu0[i][2:end, :, z_levels_WENO9nu0[i], n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(TKE_kys_WENO9nu0)
        lines!(axky_TKE_WENO9nu0, b²_spectras_WENO9nu0[i].ky[2:end], mean(TKE_kys_WENO9nu0[i][:, 2:end, z_levels_WENO9nu0[i], n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(b²_spectras_WENO9nu1eminus5)
        lines!(axkx_b_WENO91eminus5, b²_spectras_WENO9nu1eminus5[i].kx[2:end], mean(b²_spectras_WENO9nu1eminus5[i].spectra²_kx[2:end, :, z_levels_WENO9nu1eminus5[i], n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(b²_spectras_WENO9nu1eminus5)
        lines!(axky_b_WENO91eminus5, b²_spectras_WENO9nu1eminus5[i].ky[2:end], mean(b²_spectras_WENO9nu1eminus5[i].spectra²_ky[:, 2:end, z_levels_WENO9nu1eminus5[i], n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(TKE_kxs_WENO9nu1eminus5)
        lines!(axkx_TKE_WENO91eminus5, b²_spectras_WENO9nu1eminus5[i].kx[2:end], mean(TKE_kxs_WENO9nu1eminus5[i][2:end, :, z_levels_WENO9nu1eminus5[i], n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(TKE_kys_WENO9nu1eminus5)
        lines!(axky_TKE_WENO91eminus5, b²_spectras_WENO9nu1eminus5[i].ky[2:end], mean(TKE_kys_WENO9nu1eminus5[i][:, 2:end, z_levels_WENO9nu1eminus5[i], n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(b²_spectras_WENO9AMD)
        lines!(axkx_b_WENO9AMD, b²_spectras_WENO9AMD[i].kx[2:end], mean(b²_spectras_WENO9AMD[i].spectra²_kx[2:end, :, z_levels_WENO9AMD[i], n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(b²_spectras_WENO9AMD)
        lines!(axky_b_WENO9AMD, b²_spectras_WENO9AMD[i].ky[2:end], mean(b²_spectras_WENO9AMD[i].spectra²_ky[:, 2:end, z_levels_WENO9AMD[i], n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(TKE_kxs_WENO9AMD)
        lines!(axkx_TKE_WENO9AMD, b²_spectras_WENO9AMD[i].kx[2:end], mean(TKE_kxs_WENO9AMD[i][2:end, :, z_levels_WENO9AMD[i], n_window], dims=(3, 4))[:], label=labels[i])
    end

    for i in eachindex(TKE_kys_WENO9AMD)
        lines!(axky_TKE_WENO9AMD, b²_spectras_WENO9AMD[i].ky[2:end], mean(TKE_kys_WENO9AMD[i][:, 2:end, z_levels_WENO9AMD[i], n_window], dims=(3, 4))[:], label=labels[i])
    end


    hidedecorations!(axkx_b_AMD, ticks=false, ticklabels=false)
    hidedecorations!(axky_b_AMD, ticks=false, ticklabels=false)
    hidedecorations!(axkx_TKE_AMD, ticks=false, ticklabels=false)
    hidedecorations!(axky_TKE_AMD, ticks=false, ticklabels=false)

    hidedecorations!(axkx_b_WENO9nu0, ticks=false, ticklabels=false)
    hidedecorations!(axky_b_WENO9nu0, ticks=false, ticklabels=false)
    hidedecorations!(axkx_TKE_WENO9nu0, ticks=false, ticklabels=false)
    hidedecorations!(axky_TKE_WENO9nu0, ticks=false, ticklabels=false)

    hidedecorations!(axkx_b_WENO91eminus5, ticks=false, ticklabels=false)
    hidedecorations!(axky_b_WENO91eminus5, ticks=false, ticklabels=false)
    hidedecorations!(axkx_TKE_WENO91eminus5, ticks=false, ticklabels=false)
    hidedecorations!(axky_TKE_WENO91eminus5, ticks=false, ticklabels=false)

    hidedecorations!(axkx_b_WENO9AMD, ticks=false, ticklabels=false)
    hidedecorations!(axky_b_WENO9AMD, ticks=false, ticklabels=false)
    hidedecorations!(axkx_TKE_WENO9AMD, ticks=false, ticklabels=false)
    hidedecorations!(axky_TKE_WENO9AMD, ticks=false, ticklabels=false)

    linkyaxes!(axkx_b_AMD, axkx_b_WENO9nu0, axkx_b_WENO91eminus5, axkx_b_WENO9AMD)
    linkyaxes!(axky_b_AMD, axky_b_WENO9nu0, axky_b_WENO91eminus5, axky_b_WENO9AMD)
    linkyaxes!(axkx_TKE_AMD, axkx_TKE_WENO9nu0, axkx_TKE_WENO91eminus5, axkx_TKE_WENO9AMD)
    linkyaxes!(axky_TKE_AMD, axky_TKE_WENO9nu0, axky_TKE_WENO91eminus5, axky_TKE_WENO9AMD)

    linkxaxes!(axkx_b_AMD, axkx_b_WENO9nu0, axkx_b_WENO91eminus5, axkx_b_WENO9AMD)
    linkxaxes!(axky_b_AMD, axky_b_WENO9nu0, axky_b_WENO91eminus5, axky_b_WENO9AMD)
    linkxaxes!(axkx_TKE_AMD, axkx_TKE_WENO9nu0, axkx_TKE_WENO91eminus5, axkx_TKE_WENO9AMD)
    linkxaxes!(axky_TKE_AMD, axky_TKE_WENO9nu0, axky_TKE_WENO91eminus5, axky_TKE_WENO9AMD)

    Legend(fig[9, :], axkx_b_AMD, orientation=:horizontal)

    display(fig)
    save("./Data/spectra_resolution_closure_2.png", fig, px_per_unit=8)
end
#%%





