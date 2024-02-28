using Oceananigans
using Oceananigans.Advection: _advective_tracer_flux_x, _advective_tracer_flux_y, _advective_tracer_flux_z
using Oceananigans.Operators
using Oceananigans: architecture
using Oceananigans.Utils: launch!
using KernelAbstractions

function compute_χ_values(simulation)
    model = simulation.model
    advection = model.advection
    grid = model.grid
    arch = architecture(grid)
    b = model.tracers.b
    χᵁ, χⱽ, χᵂ, bⁿ⁻¹, Uⁿ⁻¹, Vⁿ⁻¹, Wⁿ⁻¹ = simulation.model.auxiliary_fields

    launch!(arch, grid, :xyz, _compute_dissipation!, χᵁ, χⱽ, χᵂ, grid, advection, 
            Uⁿ⁻¹, Vⁿ⁻¹, Wⁿ⁻¹, b, bⁿ⁻¹)

    return nothing
end

function update_previous_values(simulation)
    u, v, w = simulation.model.velocities
    b = simulation.model.tracers.b

    set!(simulation.model.auxiliary_fields.bⁿ⁻¹, b)
    set!(simulation.model.auxiliary_fields.Uⁿ⁻¹, u)
    set!(simulation.model.auxiliary_fields.Vⁿ⁻¹, v)
    set!(simulation.model.auxiliary_fields.Wⁿ⁻¹, w)

    return nothing
end


@kernel function _compute_dissipation!(χᵁ, χⱽ, χᵂ, grid, advection, Uⁿ⁻¹, Vⁿ⁻¹, Wⁿ⁻¹, b, bⁿ⁻¹)
    i, j, k = @index(Global, NTuple)

    @inbounds χᵁ[i, j, k] = compute_χᵁ(i, j, k, grid, advection, Uⁿ⁻¹, b, bⁿ⁻¹)
    @inbounds χⱽ[i, j, k] = compute_χⱽ(i, j, k, grid, advection, Vⁿ⁻¹, b, bⁿ⁻¹)
    @inbounds χᵂ[i, j, k] = compute_χᵂ(i, j, k, grid, advection, Wⁿ⁻¹, b, bⁿ⁻¹)
end

@inline b★(i, j, k, grid, bⁿ, bⁿ⁻¹) = @inbounds (bⁿ[i, j, k] + bⁿ⁻¹[i, j, k]) / 2
@inline b²(i, j, k, grid, b₁, b₂)   = @inbounds (b₁[i, j, k] * b₂[i, j, k])

@inline function compute_χᵁ(i, j, k, grid, advection, U, bⁿ, bⁿ⁻¹)
   
    δˣb★ = δxᶠᶜᶜ(i, j, k, grid, b★, bⁿ, bⁿ⁻¹)
    δˣb² = δxᶠᶜᶜ(i, j, k, grid, b², bⁿ, bⁿ⁻¹)

    𝒜x = _advective_tracer_flux_x(i, j, k, grid, advection, U, bⁿ⁻¹)
    𝒟x = @inbounds Axᶠᶜᶜ(i, j, k, grid) * U[i, j, k] * δˣb²

    return (𝒜x * 2 * δˣb★ - 𝒟x) / Vᶠᶜᶜ(i, j, k, grid)
end

@inline function compute_χⱽ(i, j, k, grid, advection, V, bⁿ, bⁿ⁻¹)
   
    δʸb★ = δyᶜᶠᶜ(i, j, k, grid, b★, bⁿ, bⁿ⁻¹)
    δʸb² = δyᶜᶠᶜ(i, j, k, grid, b², bⁿ, bⁿ⁻¹)

    𝒜y = _advective_tracer_flux_y(i, j, k, grid, advection, V, bⁿ⁻¹)
    𝒟y = @inbounds Ayᶜᶠᶜ(i, j, k, grid) * V[i, j, k] * δʸb²

    return (𝒜y * 2 * δʸb★ - 𝒟y) / Vᶜᶠᶜ(i, j, k, grid)
end

@inline function compute_χᵂ(i, j, k, grid, advection, W, bⁿ, bⁿ⁻¹)
   
    δᶻb★ = δzᶜᶜᶠ(i, j, k, grid, b★, bⁿ, bⁿ⁻¹)
    δᶻb² = δzᶜᶜᶠ(i, j, k, grid, b², bⁿ, bⁿ⁻¹)

    𝒜z = _advective_tracer_flux_z(i, j, k, grid, advection, W, bⁿ⁻¹)
    𝒟z = @inbounds Azᶜᶜᶠ(i, j, k, grid) * W[i, j, k] * δᶻb²

    return (𝒜z * 2 * δᶻb★ - 𝒟z) / Vᶜᶜᶠ(i, j, k, grid)
end

function compute_average_implicit_diffusivity(b::FieldTimeSeries, χᵁ::FieldTimeSeries, χⱽ::FieldTimeSeries, χᵂ::FieldTimeSeries)
    κx = FieldTimeSeries{Nothing, Nothing, Center}(b.grid, b.times)
    κy = FieldTimeSeries{Nothing, Nothing, Center}(b.grid, b.times)
    κz = FieldTimeSeries{Nothing, Nothing, Center}(b.grid, b.times)

    for i in eachindex(b.times)
        @info i
        _κx = Field(Average(χᵁ[i] / ∂x(b[i])^2, dims=(1, 2)))
        _κy = Field(Average(χⱽ[i] / ∂y(b[i])^2, dims=(1, 2)))
        _κz = @at (Nothing, Nothing, Center) Field(Average(χᵂ[i] / ∂z(b[i])^2, dims=(1, 2)))
        
        compute!(_κx)
        compute!(_κy)
        compute!(_κz)

        set!(κx[i], _κx)
        set!(κy[i], _κy)
        set!(κz[i], _κz)
    end
    return κx, κy, κz
end

function compute_average_implicit_dissipation(χᵁ::FieldTimeSeries, χⱽ::FieldTimeSeries, χᵂ::FieldTimeSeries)
    χx = FieldTimeSeries{Nothing, Nothing, Center}(χᵁ.grid, χᵁ.times)
    χy = FieldTimeSeries{Nothing, Nothing, Center}(χᵁ.grid, χᵁ.times)
    χz = FieldTimeSeries{Nothing, Nothing, Center}(χᵁ.grid, χᵁ.times)

    for i in eachindex(χᵁ.times)
        @info i
        _χx = Field(Average(χᵁ[i], dims=(1, 2)))
        _χy = Field(Average(χⱽ[i], dims=(1, 2)))
        _χz = Field(Average(@at((Center, Center, Center), χᵂ[i]), dims=(1, 2)))
        
        compute!(_χx)
        compute!(_χy)
        compute!(_χz)

        set!(χx[i], _χx)
        set!(χy[i], _χy)
        set!(χz[i], _χz)
    end
    return χx, χy, χz
end

function compute_average_explicit_dissipation(νₑ, χⱽ::FieldTimeSeries, χᵂ::FieldTimeSeries)
    χx = FieldTimeSeries{Nothing, Nothing, Center}(χᵁ.grid, χᵁ.times)
    χy = FieldTimeSeries{Nothing, Nothing, Center}(χᵁ.grid, χᵁ.times)
    χz = FieldTimeSeries{Nothing, Nothing, Center}(χᵁ.grid, χᵁ.times)

    for i in eachindex(χᵁ.times)
        @info i
        _χx = Field(Average(χᵁ[i], dims=(1, 2)))
        _χy = Field(Average(χⱽ[i], dims=(1, 2)))
        _χz = Field(Average(@at((Center, Center, Center), χᵂ[i]), dims=(1, 2)))
        
        compute!(_χx)
        compute!(_χy)
        compute!(_χz)

        set!(χx[i], _χx)
        set!(χy[i], _χy)
        set!(χz[i], _χz)
    end
    return χx, χy, χz
end