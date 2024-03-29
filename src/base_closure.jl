using SeawaterPolynomials.TEOS10

function calculate_Ri(u, v, T, S, Dᶠ, g, ρ₀; clamp_lims=(-Inf, Inf))
    ϵ = 1e-7
    eos = TEOS10EquationOfState()
    ρ′ = TEOS10.ρ′.(T, S, 0, Ref(eos))
    ∂ρ∂z = Dᶠ * ρ′
    ∂u∂z = Dᶠ * u
    ∂v∂z = Dᶠ * v
    ∂b∂z = -g ./ ρ₀ .* ∂ρ∂z

    return clamp.(∂b∂z ./ (∂u∂z.^2 .+ ∂v∂z.^2 .+ ϵ), clamp_lims[1], clamp_lims[2])
end

function calculate_Ri(u, v, ρ, Dᶠ, g, ρ₀; clamp_lims=(-Inf, Inf))
    ϵ = 1e-7
    ∂ρ∂z = Dᶠ * ρ
    ∂u∂z = Dᶠ * u
    ∂v∂z = Dᶠ * v
    ∂b∂z = -g ./ ρ₀ .* ∂ρ∂z

    return clamp.(∂b∂z ./ (∂u∂z.^2 .+ ∂v∂z.^2 .+ ϵ), clamp_lims[1], clamp_lims[2])
end

function calculate_Ri!(Ri, u, v, ρ, ∂b∂z, ∂u∂z, ∂v∂z, g, ρ₀, Δ; clamp_lims=(-Inf, Inf))
    ϵ = 1e-7
    
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