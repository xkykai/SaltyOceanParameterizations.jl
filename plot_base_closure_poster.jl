using CairoMakie
using SaltyOceanParameterizations: local_Ri_ν_convectivetanh_shearlinear, local_Ri_κ_convectivetanh_shearlinear
using JLD2
using ComponentArrays

BASECLOSURE_FILE_DIR = "./training_output/localbaseclosure_convectivetanh_shearlinear_TSrho_EKI/training_results.jld2"
ps_baseclosure = jldopen(BASECLOSURE_FILE_DIR, "r")["u"]

Ris = -0.5:0.001:1
νs = [local_Ri_ν_convectivetanh_shearlinear(Ri, ps_baseclosure.ν_conv, ps_baseclosure.ν_shear, ps_baseclosure.m, ps_baseclosure.ΔRi) for Ri in Ris]
Ric = (1e-5 - ps_baseclosure.ν_shear) / ps_baseclosure.m
#%%
with_theme(theme_latexfonts()) do
    fig = Figure(size=(800, 300), fontsize=22)
    ax = CairoMakie.Axis(fig[1, 1], xlabel="Ri", ylabel="Diffusivity (m² s⁻¹)")
    lines!(ax, Ris, νs, label="Viscosity", linewidth=5)
    lines!(ax, Ris, νs ./ ps_baseclosure.Pr, label="Diffusivity", linewidth=5)
    axislegend(ax, position=:rt, labelfont=:bold)
    hidedecorations!(ax, ticks=false, ticklabels=false, label=false)
    vlines!(ax, [0, Ric], color=:black, linestyle=:dash, linewidth=2)
    text!(ax, -0.3, 0.25, text="Convection-driven\nmixing", align=(:center, :center), font=:bold)
    text!(ax, Ric/2, 0.25, text="Shear-driven\nmixing", align=(:center, :center), font=:bold)

    display(fig)
    save("./poster_figures/localbaseclosure.png", fig, px_per_unit=16)
end
#%%