using SeawaterPolynomials.TEOS10
using NNlib: tanh_fast

function calculate_Ri(u, v, T, S, Dᶠ, g, ρ₀; clamp_lims=(-Inf, Inf), ϵ=1e-11)
    # ϵ = 1e-7
    eos = TEOS10EquationOfState()
    ρ′ = TEOS10.ρ′.(T, S, 0, Ref(eos))
    ∂ρ∂z = Dᶠ * ρ′
    ∂u∂z = Dᶠ * u
    ∂v∂z = Dᶠ * v
    ∂b∂z = -g ./ ρ₀ .* ∂ρ∂z

    return clamp.(∂b∂z ./ (∂u∂z.^2 .+ ∂v∂z.^2 .+ ϵ), clamp_lims[1], clamp_lims[2])
end

function calculate_Ri(u, v, ρ, Dᶠ, g, ρ₀; clamp_lims=(-Inf, Inf), ϵ=1e-11)
    # ϵ = 1e-7
    ∂ρ∂z = Dᶠ * ρ
    ∂u∂z = Dᶠ * u
    ∂v∂z = Dᶠ * v
    ∂b∂z = -g ./ ρ₀ .* ∂ρ∂z

    return clamp.(∂b∂z ./ (∂u∂z.^2 .+ ∂v∂z.^2 .+ ϵ), clamp_lims[1], clamp_lims[2])
end

function calculate_Ri!(Ri, u, v, ρ, ∂b∂z, ∂u∂z, ∂v∂z, g, ρ₀, Δ; clamp_lims=(-Inf, Inf), ϵ=1e-11)
    # ϵ = 1e-7
    Dᶠ!(∂u∂z, u, Δ)
    Dᶠ!(∂v∂z, v, Δ)
    
    Dᶠ!(∂b∂z, ρ, Δ)
    @. ∂b∂z = -g / ρ₀ * ∂b∂z

    @. Ri = clamp(∂b∂z / (∂u∂z^2 + ∂v∂z^2 + ϵ), clamp_lims[1], clamp_lims[2])
end

function local_Ri_diffusivity(Ri, ν₁, Riᶜ, ΔRi, Pr)
    ν₀ = 1e-5
    ν_conv = ν₁ / 2 * (1 - tanh((Ri - Riᶜ)/ΔRi))
    ν = ν₀ + ν_conv
    κ = ν / Pr
    return ν, κ
end

function local_Ri_ν_piecewise_linear(Ri, ν₁, m)
    m > 0 && return NaN
    ν₀ = 1e-5

    ν = clamp(m * Ri + ν₁ + ν₀, ν₀, ν₁)
    return ν
    
end

function local_Ri_κ_piecewise_linear(Ri, ν₁, m, Pr)
    ν = local_Ri_ν_piecewise_linear(Ri, ν₁, m)
    return ν / Pr
end

function local_Ri_κ_piecewise_linear(ν, Pr)
    return ν / Pr
end

function nonlocal_Ri_diffusivity(ν, D⁺, ν₁_en, Δν_enᶜ, ΔΔν_en, Pr)
    ν⁺ = D⁺ * ν
    Δν = ν⁺ .- ν

    @. ν_en = ν₁_en / 2 * (tanh((Δν - Δν_enᶜ)/ΔΔν_en) + 1)
    κ_en = ν_en ./ Pr
    return ν_en, κ_en
end

function local_Ri_ν_convectivestep_shearlinear(Ri, ν_conv, ν_shear, m)
    m > 0 && return NaN
    ν₀ = 1e-5

    # ν = ifelse(Ri < 0, ν_shear, clamp(m * Ri + ν_conv + ν₀, ν₀, ν_conv))
    # ν = ifelse(Ri < 0, ν_conv, ifelse(Ri >= (ν₀ - ν_shear) / m, ν₀, m * Ri + ν_shear))
    # ν = Ri < 0 ? ν_conv : Ri >= (ν₀ - ν_shear) / m ? ν₀ : m * Ri + ν_shear

    # ν = (-tanh_fast(1000 * Ri) + 1) * ν_conv / 2 + ν₀
    # ν = clamp(m * Ri + ν_conv + ν₀, ν₀, ν_conv)

    ν = clamp(ν_conv / 2 * (-tanh_fast(1000 * Ri) + 1) + m * Ri + ν_shear, ν₀, ν_conv)

    return ν
    # return ν₀
end

function local_Ri_κ_convectivestep_shearlinear(Ri, ν_conv, ν_shear, m, Pr)
    ν = local_Ri_ν_convectivestep_shearlinear(Ri, ν_conv, ν_shear, m)
    return ν / Pr
end

function local_Ri_κ_convectivestep_shearlinear(ν, Pr)
    return ν / Pr
end

function local_Ri_ν_convectivetanh_shearlinear(Ri, ν_conv, ν_shear, m, ΔRi)
    m > 0 && return NaN
    ν₀ = 1e-5

    if Ri >= 0
        ν = clamp(m * Ri + ν_shear + ν₀, ν₀, ν_shear)
    else
        ν = (ν_conv - ν_shear) / 2 * -tanh_fast(Ri / ΔRi) + ν_shear
    end

    return ν
end

function local_Ri_κ_convectivetanh_shearlinear(Ri, ν_conv, ν_shear, m, ΔRi, Pr)
    ν = local_Ri_ν_convectivetanh_shearlinear(Ri, ν_conv, ν_shear, m, ΔRi)
    return ν / Pr
end

function local_Ri_κ_convectivetanh_shearlinear(ν, Pr)
    return ν / Pr
end

function nonlocal_Ri_ν_convectivetanh_shearlinear(Ri, Ri_above, ∂ρ∂z, Qρ, ν_conv, ν_shear, m, ΔRi, C_en, x₀, Δx, ϵ=1e-11)
    ν_local = local_Ri_ν_convectivetanh_shearlinear(Ri, ν_conv, ν_shear, m, ΔRi)
    entrainment = Ri > 0 && Ri_above < 0 && Qρ < 0
    x = Qρ / (∂ρ∂z + ϵ)
    ν_nonlocal = entrainment * C_en * ν_conv * 0.5 * (tanh((x - x₀) / Δx) + 1)

    return ν_local + ν_nonlocal
end

function nonlocal_Ri_κ_convectivetanh_shearlinear(Ri, Ri_above, ∂ρ∂z, Qρ, ν_conv, ν_shear, m, ΔRi, C_en, x₀, Δx, Pr, ϵ=1e-11)
    ν = nonlocal_Ri_ν_convectivetanh_shearlinear(Ri, Ri_above, ∂ρ∂z, Qρ, ν_conv, ν_shear, m, ΔRi, C_en, x₀, Δx, ϵ)
    return ν / Pr
end

function nonlocal_Ri_κ_convectivetanh_shearlinear(ν, Pr)
    return ν / Pr
end