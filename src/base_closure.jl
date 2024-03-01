using SeawaterPolynomials.TEOS10

function calculate_Ri(u, v, T, S, z, Dᶠ, g, ρ₀)
    eos = TEOS10EquationOfState()
    ρ′ = TEOS10.ρ′(T, S, z, eos)
    ∂ρ∂z = Dᶠ * ρ′
    ∂u∂z = Dᶠ * u
    ∂v∂z = Dᶠ * v
    ∂b∂z = -g / ρ₀ * ∂ρ∂z

    return ∂b∂z / (∂u∂z^2 + ∂v∂z^2)
end

function local_Ri_diffusivity(Ri, ν₀, ν₁, Riᶜ, ΔRi, Pr)
    ν_conv = ν₁ / 2 * (1 - tanh((Ri - Riᶜ)/ΔRi))
    ν = ν₀ + ν_conv
    κ = ν / Pr
    return ν, κ
end

function nonlocal_Ri_diffusivity(ν, D⁺, ν₁_en, Δν_enᶜ, ΔΔν_en, Pr)
    ν⁺ = D⁺ * ν
    Δν = ν⁺ .- ν

    @. ν_en = ν₁_en / 2 * (tanh((Δν - Δν_enᶜ)/ΔΔν_en) + 1)
    κ_en = ν_en ./ Pr
    return ν_en, κ_en
end