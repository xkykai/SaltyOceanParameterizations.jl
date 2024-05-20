using CairoMakie
using SeawaterPolynomials.TEOS10
using SeawaterPolynomials

Ts = -1.5:0.1:25
Ss = 34:0.1:37
eos = TEOS10EquationOfState()

ρs = [TEOS10.ρ(T, S, 0, eos) for T in Ts, S in Ss]
αs = [SeawaterPolynomials.thermal_expansion(T, S, 0, eos) for T in Ts, S in Ss]
βs = [SeawaterPolynomials.haline_contraction(T, S, 0, eos) for T in Ts, S in Ss]
#%%
with_theme(theme_latexfonts()) do
    fig = Figure(size=(1300, 400))
    axρ = CairoMakie.Axis(fig[1, 1], xlabel="Temperature (°C)", ylabel="Salinity (g kg⁻¹)", title="Potential Density (kg m⁻³)")
    axα = CairoMakie.Axis(fig[1, 3], xlabel="Temperature (°C)", ylabel="Salinity (g kg⁻¹)", title="Thermal expansion coefficient (°C⁻¹)")
    axβ = CairoMakie.Axis(fig[1, 5], xlabel="Temperature (°C)", ylabel="Salinity (g kg⁻¹)", title="Haline contraction coefficient (g⁻¹ kg)")

    ρ_plot = contourf!(axρ, Ts, Ss, ρs, levels=20, colormap=:lipari100)
    α_plot = contourf!(axα, Ts, Ss, αs, levels=20, colormap=:lipari100)
    β_plot = contourf!(axβ, Ts, Ss, βs, levels=20, colormap=:lipari100)

    Colorbar(fig[1, 2], ρ_plot)
    Colorbar(fig[1, 4], α_plot)
    Colorbar(fig[1, 6], β_plot)
    display(fig)
    save("./figures/density_surface.png", fig, px_per_unit=8)
end
#%%