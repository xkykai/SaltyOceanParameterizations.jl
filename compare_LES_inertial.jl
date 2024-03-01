using CairoMakie
using Oceananigans
using JLD2
using Statistics

FILE_DIRS = [
    # "./LES/linearb_2layer_inertial_f_0.0001_U0_0.1_LzML_32.0_dbdz_0.001_AMD_Lxz_256.0_128.0_Nxz_128_64",
    # "./LES/linearb_2layer_inertial_f_0.0001_U0_0.1_LzML_32.0_dbdz_0.001_WENO9nu0_Lxz_256.0_128.0_Nxz_128_64",

    # "./LES/linearb_2layer_inertial_f_0.0001_U0_0.1_LzML_32.0_dbdz_0.01_AMD_Lxz_256.0_128.0_Nxz_128_64",
    # "./LES/linearb_2layer_inertial_f_0.0001_U0_0.1_LzML_32.0_dbdz_0.01_WENO9nu0_Lxz_256.0_128.0_Nxz_128_64",

    # "./LES/linearb_1layer_inertial_f_0.0001_U0_0.1_LzML_32.0_dbdz_0.01_AMD_Lxz_256.0_128.0_Nxz_128_64",
    # "./LES/linearb_1layer_inertial_f_0.0001_U0_0.1_LzML_32.0_dbdz_0.01_WENO9nu0_Lxz_256.0_128.0_Nxz_128_64",

    # "./LES/linearb_1layer_inertial_f_0.0001_U0_0.1_LzML_32.0_dbdz_0.01_AMD_Lxz_256.0_128.0_Nxz_256_128",
    # "./LES/linearb_1layer_inertial_f_0.0001_U0_0.1_LzML_32.0_dbdz_0.01_WENO9nu0_Lxz_256.0_128.0_Nxz_256_128",

    # "./LES/linearb_2layer_inertial_f_0.0001_U0_0.1_LzML_32.0_dbdz_0.01_AMD_Lxz_256.0_128.0_Nxz_128_64",
    # "./LES/linearb_2layer_inertial_f_0.0001_U0_0.1_LzML_32.0_dbdz_0.01_WENO9nu0_Lxz_256.0_128.0_Nxz_128_64",

    "./LES/linearb_2layer_inertial_f_0.0001_U0_0.1_LzML_32.0_dbdz_0.01_AMD_Lxz_256.0_128.0_Nxz_256_128",
    "./LES/linearb_2layer_inertial_f_0.0001_U0_0.1_LzML_32.0_dbdz_0.01_WENO9nu0_Lxz_256.0_128.0_Nxz_256_128",
]

labels = [
    "Centered 2nd Order + AMD, 1 m resolution",
    L"WENO + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 1 m resolution",
]

parameters = jldopen("$(FILE_DIRS[1])/instantaneous_timeseries.jld2", "r") do file
    return Dict([(key, file["metadata/parameters/$(key)"]) for key in keys(file["metadata/parameters"])])
end 

dbdz = parameters["buoyancy_gradient"]
f = parameters["coriolis_parameter"]

video_name = "./Data/2layer_1mresolution_inertial_dbdz_$(dbdz)_f_$(f).mp4"

u_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "ubar") for FILE_DIR in FILE_DIRS]
v_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "vbar") for FILE_DIR in FILE_DIRS]
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

ulim = (find_min(u_datas...), find_max(u_datas...))
vlim = (find_min(v_datas...), find_max(v_datas...))
blim = (find_min(b_datas...), find_max(b_datas...))

times = b_datas[1].times
Nt = length(times)

#%%
with_theme(theme_latexfonts()) do
    fig = Figure(size=(2100, 700))

    axubar = Axis(fig[1, 1], title="<u>", xlabel="m s⁻¹", ylabel="z")
    axvbar = Axis(fig[1, 2], title="<v>", xlabel="m s⁻¹", ylabel="z")
    axbbar = Axis(fig[1, 3], title="<b>", xlabel="m s⁻²", ylabel="z")

    n = Observable(1)

    time_str = @lift "f = $(f) s⁻², dbdz = $(dbdz) s⁻², Time = $(round(b_datas[1].times[$n]/24/60^2, digits=3)) days"
    title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)

    uₙs = [@lift interior(data[$n], 1, 1, :) for data in u_datas]
    vₙs = [@lift interior(data[$n], 1, 1, :) for data in v_datas]
    bₙs = [@lift interior(data[$n], 1, 1, :) for data in b_datas]

    for i in 1:length(FILE_DIRS)
        lines!(axubar, uₙs[i], zCs[i], label=labels[i])
        lines!(axvbar, vₙs[i], zCs[i], label=labels[i])
        lines!(axbbar, bₙs[i], zCs[i], label=labels[i])
    end

    xlims!(axubar, ulim)
    xlims!(axvbar, vlim)
    xlims!(axbbar, blim)
    Legend(fig[2, :], axbbar, tellwidth=false, orientation=:horizontal, nbanks=1)

    trim!(fig.layout)

    record(fig, video_name, 1:Nt, framerate=15) do nn
        n[] = nn
    end
    # fig = Figure(size = (1200, 800))
    # axb = Axis(fig[1, 1], title="b", ylabel="z")
    # axwb = Axis(fig[1, 2], title="wb", ylabel="z")

    # n = Observable(1)
    
    # time_str = @lift "Free convection, Qᴮ = $(Qᴮ), Time = $(round(times[$n], digits=1)) s"
    # title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)
    
    # bₙs = [@lift interior(data[findfirst(x -> x≈times[$n], data.times)], 1, 1, :) for data in b_datas]
    # wbₙs = [@lift interior(data[findfirst(x -> x≈times[$n], data.times)], 1, 1, :) for data in wb_datas]

    # MLD_theoryₙ = @lift [MLD_theory[$n]]
    
    # for i in 1:length(FILE_DIRS)
    #     lines!(axb, bₙs[i], zCs[i], label=labels[i])
    #     lines!(axwb, wbₙs[i], zFs[i], label=labels[i])
    # end

    # hlines!(axb, MLD_theoryₙ, color=:black, linestyle=:dash, linewidth=2, label="Theoretical MLD")
    # hlines!(axwb, MLD_theoryₙ, color=:black, linestyle=:dash, linewidth=2)
    # # lines!(axb, MLD_theoryₙ, label="Theoretical MLD")
    
    # # make a legend
    # Legend(fig[2, :], axb, tellwidth=false, orientation=:horizontal, nbanks=2)
    # # Label(fig[3, :], "RMSE(MLD) [AMD, WENO] = $(rmse_hs)", tellwidth=false)
    
    # # xlims!(axb, blim)
    # xlims!(axwb, wblim)
    
    # trim!(fig.layout)
    # # display(fig)
    # # save("./Data/QU_$(Qᵁ)_QB_$(Qᴮ)_btop_0_AMD_resolution.png", fig, px_per_unit=4)
    
    # record(fig, video_name, 1:Nt, framerate=15) do nn
    #     n[] = nn
    #     xlims!(axb, (nothing, nothing))
    # end
    
    @info "Animation complete"
end