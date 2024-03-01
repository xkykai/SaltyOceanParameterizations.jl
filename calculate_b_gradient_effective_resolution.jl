using Oceananigans
using CairoMakie
using Loess

FILE_DIRS = [
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_32_64_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_64_128_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_64_128_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_32_64_f",

    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_128_256_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_128_256_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_64_128_f",

    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_256_512_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_256_512_f",

    "./LES/linearb_2layer_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_AMD_Lxz_128.0_128.0_Nxz_64_64_f",
    "./LES/linearb_2layer_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_AMD_Lxz_128.0_128.0_Nxz_128_128_f",
    "./LES/linearb_2layer_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_AMD_Lxz_128.0_128.0_Nxz_256_256_f",
    "./LES/linearb_2layer_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_WENO9nu0_Lxz_128.0_128.0_Nxz_64_64_f",
    "./LES/linearb_2layer_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_WENO9nu0_Lxz_128.0_128.0_Nxz_128_128_f",
    "./LES/linearb_2layer_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_WENO9nu0_Lxz_128.0_128.0_Nxz_256_256_f",
]

labels = [
    # "Centered 2nd Order + AMD, 1m resolution",
    # "WENO + AMD, 1m resolution",
    # "WENO + No explicit closure, 2m resolution",

    # "Centered 2nd Order + AMD, 0.5m resolution",
    # "WENO + AMD, 0.5m resolution",
    # "WENO + No explicit closure, 1m resolution",

    # "Centered 2nd Order + AMD, 0.25m resolution",
    # "WENO, 0.25m resolution"

    "Centered 2nd Order + AMD, 2m resolution",
    "Centered 2nd Order + AMD, 1m resolution",
    "Centered 2nd Order + AMD, 0.5m resolution",

    L"WENO + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 2m resolution",
    L"WENO + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 1m resolution",
    L"WENO + $\nu$ = $\kappa$ = 0 m$^{2}$ s$^{-1}$, 0.5m resolution",

]

bbar_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar") for FILE_DIR in FILE_DIRS]

Δzs = [bbar_data.grid.Δzᵃᵃᶜ for bbar_data in bbar_datas]

Nzs = [bbar_data.grid.Nz for bbar_data in bbar_datas]

times = [bbar_data.times for bbar_data in bbar_datas]

zFs = [bbar_data.grid.zᵃᵃᶠ[1:Nz+1] for (bbar_data, Nz) in zip(bbar_datas, Nzs)]

zFs_Δb = [zF[2:end-1] for zF in zFs]

Δbs = [(interior(data)[1, 1, 2:end, :] .- interior(data)[1, 1, 1:end-1, :]) ./ Δz for (data, Δz) in zip(bbar_datas, Δzs)]

Δbs_argmax = [argmax(Δb, dims=1)[:] for Δb in Δbs]

Δbs_max = [Δb[argmax] for (Δb, argmax) in zip(Δbs, Δbs_argmax)]

zFs_max = [[zF[argmax[1]] for argmax in Δb_argmax] for (zF, Δb_argmax) in zip(zFs_Δb, Δbs_argmax)]

times_fit = [range(extrema(time)..., step=maximum(time)/1000) for time in times]

loess_model_Δbs = [loess(time, Δb_max, span=0.5) for (time, Δb_max) in zip(times, Δbs_max)]

Δbs_fit = [predict(model, time) for (model, time) in zip(loess_model_Δbs, times_fit)]

loess_model_zFs = [loess(time, zF_max) for (time, zF_max) in zip(times, zFs_max)]

zFs_fit = [predict(model, time) for (model, time) in zip(loess_model_zFs, times_fit)]

with_theme(theme_latexfonts()) do
    fig = Figure(size=(1200, 700))
    axΔb = Axis(fig[1, 1], ylabel="Maximum buoyancy gradient in the vertical (s⁻²)", xlabel="Time (days)")
    axzF = Axis(fig[1, 2], ylabel="z-location of maximum buoyancy gradient (m)", xlabel="Time (days)")

    for (Δb_fit, time, label) in zip(Δbs_fit, times_fit, labels)
        lines!(axΔb, time ./ 86400, Δb_fit, label=label)
    end

    for (zF_fit, time, label) in zip(zFs_fit, times_fit, labels)
        lines!(axzF, time ./ 86400, zF_fit, label=label)
    end

    Legend(fig[2, :], axzF, orientation=:horizontal, nbanks=3)
    hidedecorations!(axΔb, ticks=false, label=false, ticklabels=false)
    hidedecorations!(axzF, ticks=false, label=false, ticklabels=false)

    display(fig)

    save("./Data/MLD_quarterdomain_AMD_WENO9nu0_effectiveresolution.png", fig, px_per_unit=8)
end