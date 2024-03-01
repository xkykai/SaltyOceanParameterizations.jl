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
    # "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_f_0.0_AMD_C2_0.041666666666666664_Lxz_2.0_1.0_Nxz_128_64",

    # "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_f_0.0_AMD_C2_0.08333333333333333_Lxz_2.0_1.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_f_0.0_AMD_C2_0.08333333333333333_Lxz_2.0_1.0_Nxz_256_128",

    # "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_f_0.0_AMDC4O_C2_0.08333333333333333_Lxz_2.0_1.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_f_0.0_AMDC4O_C2_0.08333333333333333_Lxz_2.0_1.0_Nxz_256_128",
    # "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_f_0.0_AMDC4O_C2_0.08333333333333333_Lxz_2.0_1.0_Nxz_512_256",

    # "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_f_0.0_SmagorinskyLilly_Lxz_2.0_1.0_Nxz_128_64",
    # "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_f_0.0_SmagorinskyLilly_Lxz_2.0_1.0_Nxz_256_128",
    # "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_f_0.0_SmagorinskyLilly_Lxz_2.0_1.0_Nxz_512_256",

    "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_f_0.0_WENO9nu0_Lxz_2.0_1.0_Nxz_128_64",
    "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_f_0.0_WENO9nu0_Lxz_2.0_1.0_Nxz_256_128",

    "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_f_0.0_WENO5nu0_Lxz_2.0_1.0_Nxz_128_64",
    "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_f_0.0_WENO5nu0_Lxz_2.0_1.0_Nxz_256_128",
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
    # "Centered 2nd Order + AMD, 2 m resolution, C² = 1/24",
    
    # "Centered 2nd Order + AMD, 2 m resolution",
    # "Centered 2nd Order + AMD, 1 m resolution",
    # "Centered 4th Order + AMD, 2 m resolution",
    # "Centered 4th Order + AMD, 1 m resolution",
    # "Centered 4th Order + AMD, 0.5 m resolution",

    # "Centered 2nd Order + AMD, 1 m resolution",
    # "Centered 2nd Order + Smagorinsky-Lilly, 2 m resolution",
    # "Centered 2nd Order + Smagorinsky-Lilly, 1 m resolution",
    # "Centered 2nd Order + Smagorinsky-Lilly, 0.5 m resolution",

    L"WENO(9) + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 2 m resolution",
    L"WENO(9) + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 1 m resolution",
    L"WENO(5) + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 2 m resolution",
    L"WENO(5) + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 1 m resolution",
]

index_explicit = []

parameters = jldopen("$(FILE_DIRS[1])/instantaneous_timeseries.jld2", "r") do file
    # return Dict([(key, file["metadata/parameters/$(key)"]) for key in keys(file["metadata/parameters"])])
    return Dict([(key, file["metadata/$(key)"]) for key in keys(file["metadata"])])
end 

Qᵁ = parameters["momentum_flux"]
Qᴮ = parameters["buoyancy_flux"]
dbdz = parameters["buoyancy_gradient"]
f = parameters["coriolis_parameter"]

video_name = "./Data/freeconvection_WENO9_WENO5_2.8_QB_$(Qᴮ)_QU_$(Qᵁ)_dbdz_$(dbdz)_f_$(f)_dissipation.mp4"

b_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar") for FILE_DIR in FILE_DIRS]
wb_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "wb") for FILE_DIR in FILE_DIRS]
χᵁ_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "χᵁbar") for FILE_DIR in FILE_DIRS]
χⱽ_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "χⱽbar") for FILE_DIR in FILE_DIRS]
χᵂ_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "χᵂbar") for FILE_DIR in FILE_DIRS]

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
χᵁlim = (find_min(χᵁ_datas...), find_max(χᵁ_datas...))
χⱽlim = (find_min(χⱽ_datas...), find_max(χⱽ_datas...))
χᵂlim = (find_min(χᵂ_datas...), find_max(χᵂ_datas...))

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

colors = Makie.wong_colors()

#%%
with_theme(theme_latexfonts()) do
    fig = Figure(size = (1600, 1100))
    axb = Axis(fig[1, 1], title="Buoyancy", xlabel="b (m s⁻²)", ylabel="z")
    axwb = Axis(fig[1, 2], title="Buoyancy Flux", xlabel="wb (m² s⁻³)", ylabel="z")
    axχᵁ = Axis(fig[2, 1], title="Dissipation in x-direction", xlabel="<κ (∂x(b))²> (m² s⁻⁵)", ylabel="z")
    axχⱽ = Axis(fig[2, 2], title="Dissipation in y-direction", xlabel="<κ (∂y(b))²> (m² s⁻⁵)", ylabel="z")
    axχᵂ = Axis(fig[2, 3], title="Dissipation in z-direction", xlabel="<κ (∂z(b))²> (m² s⁻⁵)", ylabel="z")

    n = Observable(1)
    
    time_str = @lift "Free convection, Qᵁ = $(Qᵁ) m² s⁻², Qᴮ = $(Qᴮ) m² s⁻³, f = $(f) s⁻², Time = $(round(times[$n], digits=1)) s"
    title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)
    
    bₙs = [@lift interior(data[findfirst(x -> x≈times[$n], data.times)], 1, 1, :) for data in b_datas]
    wbₙs = [@lift interior(data[findfirst(x -> x≈times[$n], data.times)], 1, 1, :) for data in wb_datas]

    χᵁₙs = []
    χⱽₙs = []
    χᵂₙs = []

    for i in eachindex(χᵁ_datas)
        if i in index_explicit
            push!(χᵁₙs, @lift 2 .* interior(χᵁ_datas[i][findfirst(x -> x≈times[$n], χᵁ_datas[i].times)], 1, 1, :))
            push!(χⱽₙs, @lift 2 .* interior(χⱽ_datas[i][findfirst(x -> x≈times[$n], χⱽ_datas[i].times)], 1, 1, :))
            push!(χᵂₙs, @lift 2 .* interior(χᵂ_datas[i][findfirst(x -> x≈times[$n], χᵂ_datas[i].times)], 1, 1, :))
        else
            push!(χᵁₙs, @lift -1 .* interior(χᵁ_datas[i][findfirst(x -> x≈times[$n], χᵁ_datas[i].times)], 1, 1, :))
            push!(χⱽₙs, @lift -1 .* interior(χⱽ_datas[i][findfirst(x -> x≈times[$n], χⱽ_datas[i].times)], 1, 1, :))
            push!(χᵂₙs, @lift -1 .* interior(χᵂ_datas[i][findfirst(x -> x≈times[$n], χᵂ_datas[i].times)], 1, 1, :))
        end
    end

    MLD_theoryₙ = @lift [MLD_theory[$n]]
    
    for i in 1:length(FILE_DIRS)
        lines!(axb, bₙs[i], zCs[i], label=labels[i])
        lines!(axwb, wbₙs[i], zFs[i], label=labels[i])
        lines!(axχᵁ, χᵁₙs[i], zCs[i], label=labels[i])
        lines!(axχⱽ, χⱽₙs[i], zCs[i], label=labels[i])
        lines!(axχᵂ, χᵂₙs[i], zCs[i], label=labels[i])
    end

    hlines!(axb, MLD_theoryₙ, color=:black, linestyle=:dash, linewidth=2, label="Theoretical MLD")
    hlines!(axwb, MLD_theoryₙ, color=:black, linestyle=:dash, linewidth=2)
    
    # make a legend
    Legend(fig[3, :], axb, tellwidth=false, orientation=:horizontal, nbanks=2)
    
    # xlims!(axb, blim)
    xlims!(axwb, wblim)
    # xlims!(axχᵁ, χᵁlim)
    # xlims!(axχⱽ, χⱽlim)
    # xlims!(axχᵂ, χᵂlim)
    
    trim!(fig.layout)
    # display(fig)
    
    record(fig, video_name, 1:Nt, framerate=15) do nn
        n[] = nn
        xlims!(axb, (nothing, nothing))
        xlims!(axχᵁ, (nothing, nothing))
        xlims!(axχⱽ, (nothing, nothing))
        xlims!(axχᵂ, (nothing, nothing))
    end
    
    @info "Animation complete"
end