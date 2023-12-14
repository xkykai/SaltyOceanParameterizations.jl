using Oceananigans
using CairoMakie
using Loess

FILE_DIRS_AMD = [
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_32_64_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_64_128_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_128_256_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_256_512_f",
]

FILE_DIRS_WENO9NU0 = [
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_32_64_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_64_128_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_128_256_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_256_512_f",
]

labels = [
    "2m resolution",
    "1m resolution",
    "0.5m resolution",
    "0.25m resolution"
]

bbar_datas_AMD = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar") for FILE_DIR in FILE_DIRS_AMD]
bbar_datas_WENO9nu0 = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar") for FILE_DIR in FILE_DIRS_WENO9NU0]

Δzs_AMD = [bbar_data.grid.Δzᵃᵃᶜ for bbar_data in bbar_datas_AMD]
Δzs_WENO9nu0 = [bbar_data.grid.Δzᵃᵃᶜ for bbar_data in bbar_datas_WENO9nu0]

Nzs_AMD = [bbar_data.grid.Nz for bbar_data in bbar_datas_AMD]
Nzs_WENO9nu0 = [bbar_data.grid.Nz for bbar_data in bbar_datas_WENO9nu0]

times_AMD = [bbar_data.times for bbar_data in bbar_datas_AMD]
times_WENO9nu0 = [bbar_data.times for bbar_data in bbar_datas_WENO9nu0]

zFs_AMD = [bbar_data.grid.zᵃᵃᶠ[1:Nz+1] for (bbar_data, Nz) in zip(bbar_datas_AMD, Nzs_AMD)]
zFs_WENO9nu0 = [bbar_data.grid.zᵃᵃᶠ[1:Nz+1] for (bbar_data, Nz) in zip(bbar_datas_WENO9nu0, Nzs_WENO9nu0)]

zFs_Δb_AMD = [zF[2:end-1] for zF in zFs_AMD]
zFs_Δb_WENO9nu0 = [zF[2:end-1] for zF in zFs_WENO9nu0]

Δbs_AMD = [(interior(data)[1, 1, 2:end, :] .- interior(data)[1, 1, 1:end-1, :]) ./ Δz for (data, Δz) in zip(bbar_datas_AMD, Δzs_AMD)]
Δbs_WENO9nu0 = [(interior(data)[1, 1, 2:end, :] .- interior(data)[1, 1, 1:end-1, :]) ./ Δz for (data, Δz) in zip(bbar_datas_WENO9nu0, Δzs_WENO9nu0)]

Δbs_argmax_AMD = [argmax(Δb, dims=1)[:] for Δb in Δbs_AMD]
Δbs_argmax_WENO9nu0 = [argmax(Δb, dims=1)[:] for Δb in Δbs_WENO9nu0]

Δbs_max_AMD = [Δb[argmax] for (Δb, argmax) in zip(Δbs_AMD, Δbs_argmax_AMD)]
Δbs_max_WENO9nu0 = [Δb[argmax] for (Δb, argmax) in zip(Δbs_WENO9nu0, Δbs_argmax_WENO9nu0)]

zFs_max_AMD = [[zF[argmax[1]] for argmax in Δb_argmax] for (zF, Δb_argmax) in zip(zFs_Δb_AMD, Δbs_argmax_AMD)]
zFs_max_WENO9nu0 = [[zF[argmax[1]] for argmax in Δb_argmax] for (zF, Δb_argmax) in zip(zFs_Δb_WENO9nu0, Δbs_argmax_WENO9nu0)]

times_fit_AMD = [range(extrema(time)..., step=maximum(time)/1000) for time in times_AMD]
times_fit_WENO9nu0 = [range(extrema(time)..., step=maximum(time)/1000) for time in times_WENO9nu0]

loess_model_Δbs_AMD = [loess(time, Δb_max, span=0.15) for (time, Δb_max) in zip(times_AMD, Δbs_max_AMD)]
loess_model_Δbs_WENO9nu0 = [loess(time, Δb_max, span=0.15) for (time, Δb_max) in zip(times_WENO9nu0, Δbs_max_WENO9nu0)]

Δbs_fit_AMD = [predict(model, time) for (model, time) in zip(loess_model_Δbs_AMD, times_fit_AMD)]
Δbs_fit_WENO9nu0 = [predict(model, time) for (model, time) in zip(loess_model_Δbs_WENO9nu0, times_fit_WENO9nu0)]

loess_model_zFs_AMD = [loess(time, zF_max) for (time, zF_max) in zip(times_AMD, zFs_max_AMD)]
loess_model_zFs_WENO9nu0 = [loess(time, zF_max) for (time, zF_max) in zip(times_WENO9nu0, zFs_max_WENO9nu0)]

zFs_fit_AMD = [predict(model, time) for (model, time) in zip(loess_model_zFs_AMD, times_fit_AMD)]
zFs_fit_WENO9nu0 = [predict(model, time) for (model, time) in zip(loess_model_zFs_WENO9nu0, times_fit_WENO9nu0)]

with_theme(theme_latexfonts()) do
    fig = Figure(size=(1200, 1200))
    axΔb_AMD = Axis(fig[1, 1])
    axΔb_WENO9nu0 = Axis(fig[1, 2])
    axzF_AMD = Axis(fig[3, 1])
    axzF_WENO9nu0 = Axis(fig[3, 2])

    Label(fig[0, 1], "AMD", tellwidth=false)
    Label(fig[0, 2], L"WENO, $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$", tellwidth=false)

    Label(fig[1, 0], "Maximum buoyancy gradient in the vertical (s⁻²)", rotation=π/2, font=:bold, tellheight=false)
    Label(fig[3, 0], "z-location of maximum buoyancy gradient (m)", rotation=π/2, font=:bold, tellheight=false)

    Label(fig[2, 1:2], "Time (days)", font=:bold, tellwidth=false)
    Label(fig[4, 1:2], "Time (days)", font=:bold, tellwidth=false)

    for (Δb_fit, label, time) in zip(Δbs_fit_AMD, labels, times_fit_AMD)
        lines!(axΔb_AMD, time ./ (24 * 60^2), Δb_fit, label=label)
    end

    for (zF_fit, label, time) in zip(zFs_fit_AMD, labels, times_fit_AMD)
        lines!(axzF_AMD, time ./ (24 * 60^2), zF_fit, label=label)
    end

    for (Δb_fit, label, time) in zip(Δbs_fit_WENO9nu0, labels, times_fit_WENO9nu0)
        lines!(axΔb_WENO9nu0, time ./ (24 * 60^2), Δb_fit, label=label)
    end

    for (zF_fit, label, time) in zip(zFs_fit_WENO9nu0, labels, times_fit_WENO9nu0)
        lines!(axzF_WENO9nu0, time ./ (24 * 60^2), zF_fit, label=label)
    end

    Legend(fig[5, :], axzF_AMD, orientation=:horizontal)
    linkyaxes!(axΔb_AMD, axΔb_WENO9nu0)
    linkyaxes!(axzF_AMD, axzF_WENO9nu0)
    hidedecorations!(axΔb_AMD, ticks=false, label=false, ticklabels=false)
    hidedecorations!(axΔb_WENO9nu0, ticks=false, label=false, ticklabels=false)
    hidedecorations!(axzF_AMD, ticks=false, label=false, ticklabels=false)
    hidedecorations!(axzF_WENO9nu0, ticks=false, label=false, ticklabels=false)

    display(fig)

    save("./Data/MLD_AMD_WENO9nu0.png", fig, px_per_unit=8)
end