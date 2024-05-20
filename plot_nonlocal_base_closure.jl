using CairoMakie
using SaltyOceanParameterizations: local_Ri_ν_convectivetanh_shearlinear, local_Ri_κ_convectivetanh_shearlinear
using JLD2
using ComponentArrays

BASECLOSURE_FILE_DIR = "./training_output/nonlocalbaseclosure_1.0Sscaling_convectivetanh_shearlinear_TSrho_EKI_smallrho_2/training_results_mean.jld2"

ps_baseclosure = jldopen(BASECLOSURE_FILE_DIR, "r")["u"]

C_en = ps_baseclosure.C_en
x₀ = ps_baseclosure.x₀
Δx = ps_baseclosure.Δx
ν_conv = ps_baseclosure.ν_conv

xs = 0:0.001:0.16
ν_ens =  C_en * ν_conv * 0.5 .* (tanh.((xs .- x₀) ./ Δx) .+ 1)
κ_ens = ν_ens ./ ps_baseclosure.Pr
#%%
with_theme(theme_latexfonts()) do
    fig = Figure(size=(800, 400))
    ax = CairoMakie.Axis(fig[1, 1], xlabel=L"$q = \frac{Q_b}{N^2}$ (m$^{2}$ s$^{-1}$)", ylabel=L"Entrainment Diffusivity (m$^{2}$ s$^{-1}$)")
    lines!(ax, xs, ν_ens, label="Viscosity")
    lines!(ax, xs, κ_ens, label="Diffusivity")

    axislegend(ax, position=:rb)
    hidedecorations!(ax, ticks=false, ticklabels=false, label=false)
    display(fig)
    save("./figures/nonlocalbaseclosure_entrainment.png", fig, px_per_unit=8)
end
#%%