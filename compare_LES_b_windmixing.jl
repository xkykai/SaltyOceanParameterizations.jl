using CairoMakie
using Oceananigans
using JLD2

FILE_DIRS = [
    # "./LES/linearb_experiments_dbdz_5.0_QU_-5.0e-5_QB_0.0_b_0.0_AMD_Lxz_2.0_1.0_Nxz_256_128",
    # "./LES/linearb_experiments_dbdz_5.0_QU_-5.0e-5_QB_0.0_b_0.0_WENO9nu0_Lxz_2.0_1.0_Nxz_256_128",
    "./LES/linearb_experiments_dbdz_5.0_QU_-5.0e-5_QB_0.0_b_0.0_AMD_Lxz_2.0_1.0_Nxz_128_64",
    "./LES/linearb_experiments_dbdz_5.0_QU_-5.0e-5_QB_0.0_b_0.0_WENO9nu0_Lxz_2.0_1.0_Nxz_128_64",
]

labels = [
    # "Centered 2nd Order + AMD, 1/128 m resolution",
    # L"WENO + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 1/128 m resolution",
    "Centered 2nd Order + AMD, 1/64 m resolution",
    L"WENO + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 1/64 m resolution",
]

parameters = jldopen("$(FILE_DIRS[1])/instantaneous_timeseries.jld2", "r") do file
    return Dict([(key, file["metadata/parameters/$(key)"]) for key in keys(file["metadata/parameters"])])
end 

Qᵁ = parameters["momentum_flux"]
dbdz = parameters["buoyancy_gradient"]

video_name = "./Data/windmixing_QU_$(Qᵁ)_dbdz_$(dbdz).mp4"

b_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar") for FILE_DIR in FILE_DIRS]

Nxs = [size(data.grid)[1] for data in b_datas]
Nys = [size(data.grid)[2] for data in b_datas]
Nzs = [size(data.grid)[3] for data in b_datas]

zCs = [data.grid.zᵃᵃᶜ[1:Nzs[i]] for (i, data) in enumerate(b_datas)]
zFs = [data.grid.zᵃᵃᶠ[1:Nzs[i]+1] for (i, data) in enumerate(b_datas)]

#%%
function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

blim = (find_min(b_datas...), find_max(b_datas...))
times = b_datas[1].times
Nt = length(times)

function wind_mixing_MLD_scaling(Qᵁ, dbdz, t)
    ustar = √(Qᵁ)
    return -ustar * (15*t / dbdz)^(1/3)
end

MLD_theory = [wind_mixing_MLD_scaling(Qᵁ, dbdz, t) for t in times]

#%%
with_theme(theme_latexfonts()) do
    fig = Figure(size = (1200, 900))
    axb = Axis(fig[1, 1], title="b", ylabel="z")

    n = Observable(1)
    
    time_str = @lift "Wind Mixing, Qᵁ = $(Qᵁ), Time = $(round(times[$n], digits=1)) s"
    title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)
    
    bₙs = [@lift interior(data[findfirst(x -> x≈times[$n], data.times)], 1, 1, :) for data in b_datas]

    MLD_theoryₙ = @lift [MLD_theory[$n]]
    
    for i in 1:length(FILE_DIRS)
        lines!(axb, bₙs[i], zCs[i], label=labels[i])
    end

    hlines!(axb, MLD_theoryₙ, color=:black, linestyle=:dash, linewidth=2, label="Theoretical MLD")
    
    # make a legend
    Legend(fig[2, :], axb, tellwidth=false, orientation=:horizontal, nbanks=2)
    
    xlims!(axb, blim)
    
    trim!(fig.layout)
    
    record(fig, video_name, 1:Nt, framerate=15) do nn
        n[] = nn
    end
    
    @info "Animation complete"
end