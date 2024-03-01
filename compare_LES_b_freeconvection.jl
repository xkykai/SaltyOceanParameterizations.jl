using CairoMakie
using Oceananigans
using JLD2
using Statistics

FILE_DIRS = [
    # "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_AMD_Lxz_2.0_1.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_WENO9nu0_Lxz_2.0_1.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_AMD_Lxz_2.0_1.0_Nxz_256_128",
    # "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_WENO9nu0_Lxz_2.0_1.0_Nxz_256_128",
    # "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_AMD_Lxz_2.0_1.0_Nxz_512_256",
    # "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_WENO9nu0_Lxz_2.0_1.0_Nxz_512_256",

    # "./LES/linearb_experiments_dbdz_1.0e-5_QU_0.0_QB_4.0e-7_b_0.0_AMD_Lxz_256.0_128.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_1.0e-5_QU_0.0_QB_4.0e-7_b_0.0_WENO9nu0_Lxz_256.0_128.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_1.0e-5_QU_0.0_QB_4.0e-7_b_0.0_AMD_Lxz_256.0_128.0_Nxz_256_128",
    # "./LES/linearb_experiments_dbdz_1.0e-5_QU_0.0_QB_4.0e-7_b_0.0_WENO9nu0_Lxz_256.0_128.0_Nxz_256_128",
    # "./LES/linearb_experiments_dbdz_1.0e-5_QU_0.0_QB_4.0e-7_b_0.0_AMD_Lxz_256.0_128.0_Nxz_512_256",
    # "./LES/linearb_experiments_dbdz_1.0e-5_QU_0.0_QB_4.0e-7_b_0.0_WENO9nu0_Lxz_256.0_128.0_Nxz_512_256",

#     "./LES/linearb_experiments_dbdz_1.0e-5_QU_0.0_QB_4.0e-8_b_0.0_AMD_Lxz_256.0_128.0_Nxz_128_64",
#     "./LES/linearb_experiments_dbdz_1.0e-5_QU_0.0_QB_4.0e-8_b_0.0_WENO9nu0_Lxz_256.0_128.0_Nxz_128_64",
#     # "./LES/linearb_experiments_dbdz_1.0e-5_QU_0.0_QB_4.0e-8_b_0.0_AMD_Lxz_512.0_128.0_Nxz_256_64",
#     # "./LES/linearb_experiments_dbdz_1.0e-5_QU_0.0_QB_4.0e-8_b_0.0_WENO9nu0_Lxz_512.0_128.0_Nxz_256_64",
#     "./LES/linearb_experiments_dbdz_1.0e-5_QU_0.0_QB_4.0e-8_b_0.0_AMD_Lxz_256.0_128.0_Nxz_256_128",
#     "./LES/linearb_experiments_dbdz_1.0e-5_QU_0.0_QB_4.0e-8_b_0.0_WENO9nu0_Lxz_256.0_128.0_Nxz_256_128",
#     "./LES/linearb_experiments_dbdz_1.0e-5_QU_0.0_QB_4.0e-8_b_0.0_AMD_Lxz_256.0_128.0_Nxz_512_256",
#     "./LES/linearb_experiments_dbdz_1.0e-5_QU_0.0_QB_4.0e-8_b_0.0_WENO9nu0_Lxz_256.0_128.0_Nxz_512_256",

    # "./LES/linearb_experiments_dbdz_0.0001_QU_0.0_QB_4.0e-7_b_0.0_AMD_Lxz_256.0_128.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_0.0001_QU_0.0_QB_4.0e-7_b_0.0_WENO9nu0_Lxz_256.0_128.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_0.0001_QU_0.0_QB_4.0e-7_b_0.0_WENO9AMD_Lxz_256.0_128.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_0.0001_QU_0.0_QB_4.0e-7_b_0.0_AMD_Lxz_256.0_128.0_Nxz_256_128",
    # "./LES/linearb_experiments_dbdz_0.0001_QU_0.0_QB_4.0e-7_b_0.0_WENO9nu0_Lxz_256.0_128.0_Nxz_256_128",
    # "./LES/linearb_experiments_dbdz_0.0001_QU_0.0_QB_4.0e-7_b_0.0_WENO9AMD_Lxz_256.0_128.0_Nxz_256_128",
    # "./LES/linearb_experiments_dbdz_0.0001_QU_0.0_QB_4.0e-7_b_0.0_AMD_Lxz_256.0_128.0_Nxz_512_256",
    # "./LES/linearb_experiments_dbdz_0.0001_QU_0.0_QB_4.0e-7_b_0.0_WENO9nu0_Lxz_256.0_128.0_Nxz_512_256",
    # "./LES/linearb_experiments_dbdz_0.0001_QU_0.0_QB_4.0e-7_b_0.0_WENO9AMD_Lxz_256.0_128.0_Nxz_512_256",

    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_0.0_QB_4.0e-7_b_0.0_f_0.0_AMD_Lxz_256.0_128.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_0.0_QB_4.0e-7_b_0.0_f_0.0_WENO9nu0_Lxz_256.0_128.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_0.0_QB_4.0e-7_b_0.0_f_0.0_AMD_Lxz_256.0_128.0_Nxz_256_128",
    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_0.0_QB_4.0e-7_b_0.0_f_0.0_WENO9nu0_Lxz_256.0_128.0_Nxz_256_128",

    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_0.0_QB_4.0e-7_b_0.0_f_0.0001_AMD_Lxz_256.0_128.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_0.0_QB_4.0e-7_b_0.0_f_0.0001_WENO9nu0_Lxz_256.0_128.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_0.0_QB_4.0e-7_b_0.0_f_0.0001_AMD_Lxz_256.0_128.0_Nxz_256_128",
    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_0.0_QB_4.0e-7_b_0.0_f_0.0001_WENO9nu0_Lxz_256.0_128.0_Nxz_256_128",

    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_-5.0e-5_QB_4.0e-7_b_0.0_f_0.0001_AMD_Lxz_256.0_128.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_-5.0e-5_QB_4.0e-7_b_0.0_f_0.0001_WENO9nu0_Lxz_256.0_128.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_-5.0e-5_QB_4.0e-7_b_0.0_f_0.0001_AMD_Lxz_256.0_128.0_Nxz_256_128",
    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_-5.0e-5_QB_4.0e-7_b_0.0_f_0.0001_WENO9nu0_Lxz_256.0_128.0_Nxz_256_128",

    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_0.0_QB_4.0e-8_b_0.0_f_0.0_AMD_Lxz_256.0_128.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_0.0_QB_4.0e-8_b_0.0_f_0.0_WENO9nu0_Lxz_256.0_128.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_0.0_QB_4.0e-8_b_0.0_f_0.0_WENO9AMD_Lxz_256.0_128.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_0.0_QB_4.0e-8_b_0.0_f_0.0_AMD_Lxz_256.0_128.0_Nxz_256_128",
    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_0.0_QB_4.0e-8_b_0.0_f_0.0_WENO9nu0_Lxz_256.0_128.0_Nxz_256_128",
    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_0.0_QB_4.0e-8_b_0.0_f_0.0_WENO9AMD_Lxz_256.0_128.0_Nxz_256_128",

    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_-5.0e-5_QB_4.0e-8_b_0.0_f_0.0_AMD_Lxz_256.0_128.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_-5.0e-5_QB_4.0e-8_b_0.0_f_0.0_WENO9nu0_Lxz_256.0_128.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_-5.0e-5_QB_4.0e-8_b_0.0_f_0.0_WENO9AMD_Lxz_256.0_128.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_-5.0e-5_QB_4.0e-8_b_0.0_f_0.0_AMD_Lxz_256.0_128.0_Nxz_256_128",
    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_-5.0e-5_QB_4.0e-8_b_0.0_f_0.0_WENO9nu0_Lxz_256.0_128.0_Nxz_256_128",
    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_-5.0e-5_QB_4.0e-8_b_0.0_f_0.0_WENO9AMD_Lxz_256.0_128.0_Nxz_256_128",

    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_-5.0e-5_QB_4.0e-8_b_0.0_f_0.0001_AMD_Lxz_256.0_128.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_-5.0e-5_QB_4.0e-8_b_0.0_f_0.0001_WENO9nu0_Lxz_256.0_128.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_5.0e-6_QU_-5.0e-5_QB_4.0e-8_b_0.0_f_0.0001_WENO9AMD_Lxz_256.0_128.0_Nxz_128_64",
    
    # "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_f_0.0_AMD_C2_0.08333333333333333_Lxz_2.0_1.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_f_0.0_AMD_C2_0.25_Lxz_2.0_1.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_f_0.0_AMD_C2_0.041666666666666664_Lxz_2.0_1.0_Nxz_128_64"

    "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_f_0.0_WENO9nu0_Lxz_2.0_1.0_Nxz_128_64",
    "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_f_0.0_WENO9nu0_Lxz_2.0_1.0_Nxz_256_128",
    "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_f_0.0_WENO9nu0_Lxz_2.0_1.0_Nxz_512_256",

    "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_f_0.0_WENO5nu0_Lxz_2.0_1.0_Nxz_128_64",
    "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_f_0.0_WENO5nu0_Lxz_2.0_1.0_Nxz_256_128",
    "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_f_0.0_WENO5nu0_Lxz_2.0_1.0_Nxz_512_256",
]

labels = [
    # "Centered 2nd Order + AMD, 1/64 m resolution",
    # L"WENO + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 1/64 m resolution",
    # "Centered 2nd Order + AMD, 1/128 m resolution",
    # L"WENO + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 1/128 m resolution",
    # "Centered 2nd Order + AMD, 1/256 m resolution",
    # L"WENO + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 1/256 m resolution",

    # "Centered 2nd Order + AMD, 2 m resolution",
    # L"WENO + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 2 m resolution",
    # L"WENO + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$ + AMD, 2 m resolution",
    # "Centered 2nd Order + AMD, 2 m resolution, 2x horizontal domain",
    # L"WENO + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 2 m resolution, 2x horizontal domain",
    # "Centered 2nd Order + AMD, 1 m resolution",
    # L"WENO + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 1 m resolution",
    # L"WENO + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$ + AMD, 1 m resolution",
    # "Centered 2nd Order + AMD, 0.5 m resolution",
    # L"WENO + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 0.5 m resolution",
    # L"WENO + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$ + AMD, 0.5 m resolution",

    # "Centered 2nd Order + AMD, 2 m resolution, C² = 1/12",
    # "Centered 2nd Order + AMD, 2 m resolution, C² = 1/4",
    # "Centered 2nd Order + AMD, 2 m resolution, C² = 1/24"

    L"WENO(9) + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 2 m resolution",
    L"WENO(9) + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 1 m resolution",
    L"WENO(9) + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 0.5 m resolution",

    L"WENO(5) + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 2 m resolution",
    L"WENO(5) + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 1 m resolution",
    L"WENO(5) + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 0.5 m resolution",
]

parameters = jldopen("$(FILE_DIRS[1])/instantaneous_timeseries.jld2", "r") do file
    # return Dict([(key, file["metadata/parameters/$(key)"]) for key in keys(file["metadata/parameters"])])
    return Dict([(key, file["metadata/$(key)"]) for key in keys(file["metadata"])])
end 

Qᵁ = parameters["momentum_flux"]
Qᴮ = parameters["buoyancy_flux"]
dbdz = parameters["buoyancy_gradient"]
f = parameters["coriolis_parameter"]

video_name = "./Data/freeconvection_WENO_2.8_QB_$(Qᴮ)_QU_$(Qᵁ)_dbdz_$(dbdz)_f_$(f)_convergence.mp4"

b_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar") for FILE_DIR in FILE_DIRS]
wb_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wb") for FILE_DIR in FILE_DIRS]

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

startframe_lim = 30
wblim = (find_min([wb_data[1, 1, :, startframe_lim:end] for wb_data in wb_datas]...), find_max([wb_data[1, 1, :, startframe_lim:end] for wb_data in wb_datas]...))

times = b_datas[1].times
Nt = length(times)

function free_convection_MLD_scaling(Qᴮ, dbdz, t)
    # return -√(3 * Qᴮ / dbdz * t)
    return -√(2.8 * Qᴮ / dbdz * t)
end

MLD_theory = [free_convection_MLD_scaling(Qᴮ, dbdz, t) for t in times]

# h_indices = [argmin(data, dims=3) for data in wb_datas]
# wb_mins = [data[h_index] for (data, h_index) in zip(wb_datas, h_indices)]

# hs = [[zF[h_index[1, 1, 1, i][3]] for i in axes(h_index, 4)] for (zF, h_index) in zip(zFs, h_indices)]
# rmse_hs = [sqrt(mean((h[2:end] .- MLD_theory[2:end]).^2)) for h in hs]

#%%
with_theme(theme_latexfonts()) do
    fig = Figure(size = (1200, 800))
    axb = Axis(fig[1, 1], title="b", ylabel="z")
    axwb = Axis(fig[1, 2], title="wb", ylabel="z")

    n = Observable(1)
    
    time_str = @lift "Free convection, Qᵁ = $(Qᵁ) m² s⁻², Qᴮ = $(Qᴮ) m² s⁻³, f = $(f) s⁻², Time = $(round(times[$n], digits=1)) s"
    title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)
    
    bₙs = [@lift interior(data[findfirst(x -> x≈times[$n], data.times)], 1, 1, :) for data in b_datas]
    wbₙs = [@lift interior(data[findfirst(x -> x≈times[$n], data.times)], 1, 1, :) for data in wb_datas]

    MLD_theoryₙ = @lift [MLD_theory[$n]]
    
    for i in 1:length(FILE_DIRS)
        lines!(axb, bₙs[i], zCs[i], label=labels[i])
        lines!(axwb, wbₙs[i], zFs[i], label=labels[i])
    end

    hlines!(axb, MLD_theoryₙ, color=:black, linestyle=:dash, linewidth=2, label="Theoretical MLD")
    hlines!(axwb, MLD_theoryₙ, color=:black, linestyle=:dash, linewidth=2)
    # lines!(axb, MLD_theoryₙ, label="Theoretical MLD")
    
    # make a legend
    Legend(fig[2, :], axb, tellwidth=false, orientation=:horizontal, nbanks=3)
    # Label(fig[3, :], "RMSE(MLD) [AMD, WENO] = $(rmse_hs)", tellwidth=false)
    
    # xlims!(axb, blim)
    xlims!(axwb, wblim)
    
    trim!(fig.layout)
    # display(fig)
    # save("./Data/QU_$(Qᵁ)_QB_$(Qᴮ)_btop_0_AMD_resolution.png", fig, px_per_unit=4)
    
    record(fig, video_name, 1:Nt, framerate=15) do nn
        n[] = nn
        xlims!(axb, (nothing, nothing))
    end
    
    @info "Animation complete"
end