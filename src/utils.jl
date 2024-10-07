using SeawaterPolynomials.TEOS10

function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

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