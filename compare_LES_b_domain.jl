using CairoMakie
using Oceananigans
using JLD2

FILE_DIRS = [
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_32_64_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_128.0_128.0_Nxz_64_64_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_256.0_128.0_Nxz_128_64_f",

    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_64_128_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_128.0_128.0_Nxz_128_128_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_256.0_128.0_Nxz_256_128_f",
]

labels = [
    "Lx = Ly = 64m",
    "Lx = Ly = 128m",
    "Lx = Ly = 256m",
]

parameters = jldopen("$(FILE_DIRS[1])/instantaneous_timeseries.jld2", "r") do file
    return Dict([(key, file["metadata/parameters/$(key)"]) for key in keys(file["metadata/parameters"])])
end 

Qᴮ = parameters["buoyancy_flux"]
Qᵁ = parameters["momentum_flux"]

@info "Loading b data..."
b_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar") for FILE_DIR in FILE_DIRS]

Nzs = [size(data.grid)[3] for data in b_datas]

zCs = [data.grid.zᵃᵃᶜ[1:Nzs[i]] for (i, data) in enumerate(b_datas)]
zFs = [data.grid.zᵃᵃᶠ[1:Nzs[i]+1] for (i, data) in enumerate(b_datas)]

t = 48 * 3600
ns = [findfirst(x -> x ≈ t, data.times) for data in b_datas]
@info ns

#%%
function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

@info "Plotting..."
with_theme(theme_latexfonts()) do
    fig = Figure(size=(1200, 600))
    ax2m = Axis(fig[1, 1], title="2m resolution", ylabel="z (m)", xlabel="Buoyancy (m s⁻²)")
    ax1m = Axis(fig[1, 2], title="1m resolution", ylabel="z (m)", xlabel="Buoyancy (m s⁻²)")

    for (data, n, zC, label) in zip(b_datas[1:3], ns[1:3], zCs[1:3], labels)
        lines!(ax2m, interior(data[n], 1, 1, :), zC, label=label)
    end

    for (data, n, zC, label) in zip(b_datas[4:6], ns[4:6], zCs[4:6], labels)
        lines!(ax1m, interior(data[n], 1, 1, :), zC, label=label)
    end

    Label(fig[0, :], L"WENO, $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$", tellwidth=false)
    Legend(fig[2, :], ax2m, orientation=:horizontal)
    # Legend(fig[5, :], axzF_AMD, orientation=:horizontal)
    # hidedecorations!(axΔb_AMD, ticks=false, label=false, ticklabels=false)
    # hidedecorations!(axΔb_WENO9nu0, ticks=false, label=false, ticklabels=false)
    # hidedecorations!(axzF_AMD, ticks=false, label=false, ticklabels=false)
    # hidedecorations!(axzF_WENO9nu0, ticks=false, label=false, ticklabels=false)

    save("./Data/domain_WENO9nu0_esolution.png", fig, px_per_unit=8)
end

@info "completed"