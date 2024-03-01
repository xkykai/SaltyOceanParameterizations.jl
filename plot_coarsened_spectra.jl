using JLD2
using CairoMakie
using Statistics

FILE_DIRS = [
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_32_64_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_32_64_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_32_64_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_32_64_f",

    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_64_128_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_64_128_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_64_128_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_64_128_f",
]

FILE_DIR_FINE = "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_256_512_f"

labels = [
    # "AMD, 2m resolution",
    # "WENO(9), ν = κ = 0, 2m resolution",
    # "WENO(9), ν = κ = 1e-5, 2m resolution",
    # "WENO(9) + AMD, 2m resolution",

    "AMD, 1m resolution",
    "WENO(9), ν = κ = 0, 1m resolution",
    "WENO(9), ν = κ = 1e-5, 1m resolution",
    "WENO(9) + AMD, 1m resolution",

    "AMD, 0.25m resolution"
]

scaling = 4

parameters = jldopen("$(FILE_DIRS[1])/instantaneous_timeseries.jld2", "r") do file
    return Dict([(key, file["metadata/parameters/$(key)"]) for key in keys(file["metadata/parameters"])])
end 

Qᴮ = parameters["buoyancy_flux"]
Qᵁ = parameters["momentum_flux"]

bbar_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar") for FILE_DIR in FILE_DIRS]
bbar_fine = FieldTimeSeries("$(FILE_DIR_FINE)/instantaneous_timeseries.jld2", "bbar")

b_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_b.jld2", "b", backend=OnDisk()) for FILE_DIR in FILE_DIRS]

b²_spectras = [load("$(FILE_DIR)/spectra.jld2", "b") for FILE_DIR in FILE_DIRS]
u²_spectras = [load("$(FILE_DIR)/spectra.jld2", "u") for FILE_DIR in FILE_DIRS]
v²_spectras = [load("$(FILE_DIR)/spectra.jld2", "v") for FILE_DIR in FILE_DIRS]
w²_spectras = [load("$(FILE_DIR)/spectra.jld2", "w") for FILE_DIR in FILE_DIRS]

b²_spectra_fine = load("$(FILE_DIR_FINE)/spectra.jld2", "b")
u²_spectra_fine = load("$(FILE_DIR_FINE)/spectra.jld2", "u")
v²_spectra_fine = load("$(FILE_DIR_FINE)/spectra.jld2", "v")
w²_spectra_fine = load("$(FILE_DIR_FINE)/spectra.jld2", "w")

b²_spectra_coarsened = load("$(FILE_DIR_FINE)/coarsened_spectra_$(scaling).jld2", "b")
u²_spectra_coarsened = load("$(FILE_DIR_FINE)/coarsened_spectra_$(scaling).jld2", "u")
v²_spectra_coarsened = load("$(FILE_DIR_FINE)/coarsened_spectra_$(scaling).jld2", "v")
w²_spectra_coarsened = load("$(FILE_DIR_FINE)/coarsened_spectra_$(scaling).jld2", "w")

Nxs = [size(data.grid)[1] for data in bbar_datas]
Nys = [size(data.grid)[2] for data in bbar_datas]
Nzs = [size(data.grid)[3] for data in bbar_datas]

Δxs = [data.grid.Δxᶜᵃᵃ for data in bbar_datas]
Δys = [data.grid.Δyᵃᶜᵃ for data in bbar_datas]

zCs = [data.grid.zᵃᵃᶜ[1:Nzs[i]] for (i, data) in enumerate(bbar_datas)]
zFs = [data.grid.zᵃᵃᶠ[1:Nzs[i]+1] for (i, data) in enumerate(bbar_datas)]

Nz_fine = size(bbar_fine.grid, 3)
zCs_fine = bbar_fine.grid.zᵃᵃᶜ[1:Nz_fine]

# times_coarse = b_datas[1].times[1:end-2]
# times_fine = bbar_datas[1].times[1:end-2]

times_coarse = b_datas[1].times
times_fine = bbar_datas[1].times

n_start = 29
n_window = n_start:length(times_coarse)

n_start_fine = findfirst(x -> x ≈ times_coarse[n_start], times_fine)
n_end_fine = findfirst(x -> x ≈ times_coarse[end], times_fine)

times = (0:60^2:48*60^2) ./ (24 * 60^2)

zrange = (-50, -2)

z_levels = [findfirst(z -> z>zrange[1], zC):findlast(z -> z<zrange[2], zC) for zC in zCs]
z_levels_fine = findfirst(z -> z>zrange[1], zCs_fine):findlast(z -> z<zrange[2], zCs_fine)

TKE_kxs = [0.5 .* (u²_spectras[i].spectra²_kx .+ v²_spectras[i].spectra²_kx .+ w²_spectras[i].spectra²_kx) for i in eachindex(u²_spectras)]
TKE_kys = [0.5 .* (u²_spectras[i].spectra²_ky .+ v²_spectras[i].spectra²_ky .+ w²_spectras[i].spectra²_ky) for i in eachindex(u²_spectras)]

TKE_kx_fine = 0.5 .* (u²_spectra_fine.spectra²_kx .+ v²_spectra_fine.spectra²_kx .+ w²_spectra_fine.spectra²_kx)
TKE_ky_fine = 0.5 .* (u²_spectra_fine.spectra²_ky .+ v²_spectra_fine.spectra²_ky .+ w²_spectra_fine.spectra²_ky)

TKE_kx_coarsened = 0.5 .* (u²_spectra_coarsened.spectra²_kx .+ v²_spectra_coarsened.spectra²_kx .+ w²_spectra_coarsened.spectra²_kx)
TKE_ky_coarsened = 0.5 .* (u²_spectra_coarsened.spectra²_ky .+ v²_spectra_coarsened.spectra²_ky .+ w²_spectra_coarsened.spectra²_ky)
#%%
with_theme(theme_latexfonts()) do
    fig = Figure(size=(1200, 800))
    axb1 = Axis(fig[1, 1], xlabel="Buoyancy (m s⁻²)", ylabel="z (m)", title="Time = $(round(times[1], digits=3)) days")
    axb2 = Axis(fig[2, 1], xlabel="Buoyancy (m s⁻²)", ylabel="z (m)", title="Time = $(round(times[end], digits=3)) days")

    axkx_b = Axis(fig[1, 2], xlabel=L"k_x (\mathrm{m}^{-1})", ylabel="Buoyancy spectra (m³ s⁻⁴)", title="Buoyancy variance, averaged in y", xscale=log10, yscale=log10)
    axky_b = Axis(fig[2, 2], xlabel=L"k_y (\mathrm{m}^{-1})", ylabel="Buoyancy spectra (m³ s⁻⁴)", title="Buoyancy variance, averaged in x", xscale=log10, yscale=log10)

    axkx_TKE = Axis(fig[1, 3], xlabel=L"k_x (\mathrm{m}^{-1})", ylabel="TKE spectra (m³ s⁻²)", title="Turbulent kinetic energy, averaged in y", xscale=log10, yscale=log10)
    axky_TKE = Axis(fig[2, 3], xlabel=L"k_y (\mathrm{m}^{-1})", ylabel="TKE spectra (m³ s⁻²)", title="Turbulent kinetic energy, averaged in x", xscale=log10, yscale=log10)
    
    for i in eachindex(bbar_datas)
        lines!(axb1, interior(bbar_datas[i][n_start_fine], 1, 1, :), zCs[i], label=labels[i])
        lines!(axb2, interior(bbar_datas[i][n_end_fine], 1, 1, :), zCs[i], label=labels[i])
    
        lines!(axkx_b, b²_spectras[i].kx[2:end], mean(b²_spectras[i].spectra²_kx[2:end, :, z_levels[i], n_window], dims=(3, 4))[:], label=labels[i])
        lines!(axky_b, b²_spectras[i].ky[2:end], mean(b²_spectras[i].spectra²_ky[:, 2:end, z_levels[i], n_window], dims=(3, 4))[:], label=labels[i])
    
        lines!(axkx_TKE, b²_spectras[i].kx[2:end], mean(TKE_kxs[i][2:end, :, z_levels[i], n_window], dims=(3, 4))[:], label=labels[i])
        lines!(axky_TKE, b²_spectras[i].ky[2:end], mean(TKE_kys[i][:, 2:end, z_levels[i], n_window], dims=(3, 4))[:], label=labels[i])
    end

    
    # lines!(axkx_b, b²_spectra_fine.kx[2:end], mean(b²_spectra_fine.spectra²_kx[2:end, :, z_levels_fine, n_window], dims=(3, 4))[:], label="Fine")
    # lines!(axky_b, b²_spectra_fine.ky[2:end], mean(b²_spectra_fine.spectra²_ky[:, 2:end, z_levels_fine, n_window], dims=(3, 4))[:], label="Fine")
    
    # lines!(axkx_TKE, b²_spectra_fine.kx[2:end], mean(TKE_kx_fine[2:end, :, z_levels_fine, n_window], dims=(3, 4))[:], label="Fine")
    # lines!(axky_TKE, b²_spectra_fine.ky[2:end], mean(TKE_ky_fine[:, 2:end, z_levels_fine, n_window], dims=(3, 4))[:], label="Fine")
    
    lines!(axkx_b, b²_spectra_coarsened.kx[2:end], mean(b²_spectra_coarsened.spectra²_kx[2:end, :, z_levels[1], n_window], dims=(3, 4))[:], label="Coarsened")
    lines!(axky_b, b²_spectra_coarsened.ky[2:end], mean(b²_spectra_coarsened.spectra²_ky[:, 2:end, z_levels[1], n_window], dims=(3, 4))[:], label="Coarsened")
    
    lines!(axkx_TKE, b²_spectra_coarsened.kx[2:end], mean(TKE_kx_coarsened[2:end, :, z_levels[1], n_window], dims=(3, 4))[:], label="Coarsened")
    lines!(axky_TKE, b²_spectra_coarsened.ky[2:end], mean(TKE_ky_coarsened[:, 2:end, z_levels[1], n_window], dims=(3, 4))[:], label="Coarsened")
    Legend(fig[3, :], axkx_b, orientation=:horizontal)

    hlines!(axb1, [zCs[1][z_levels[1][1]], zCs[1][z_levels[1][end]]], color=:black, linestyle=:dash)
    hlines!(axb2, [zCs[1][z_levels[1][1]], zCs[1][z_levels[1][end]]], color=:black, linestyle=:dash)
    
    hidedecorations!(axb1, minorgrid=false, ticks=false, label=false, ticklabels=false)
    hidedecorations!(axb2, minorgrid=false, ticks=false, label=false, ticklabels=false)
    hidedecorations!(axkx_b, minorgrid=false, ticks=false, label=false, ticklabels=false)
    hidedecorations!(axky_b, minorgrid=false, ticks=false, label=false, ticklabels=false)
    hidedecorations!(axkx_TKE, minorgrid=false, ticks=false, label=false, ticklabels=false)
    hidedecorations!(axky_TKE, minorgrid=false, ticks=false, label=false, ticklabels=false)

    display(fig)
    save("./Data/coarsened_spectra_$(scaling).pdf", fig)
end
#%%





