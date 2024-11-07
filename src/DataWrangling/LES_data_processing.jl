using Oceananigans: FieldDataset, interior, Face, Center
using SaltyOceanParameterizations.Operators: Dᶠ
using SeawaterPolynomials
using SeawaterPolynomials.TEOS10
struct LESDataset{D}
    data :: D
end

abstract type AbstractProfile end

struct Profile{SC, S, U} <: AbstractProfile
    scaling :: SC
    scaled :: S
    unscaled :: U
end

function Profile(data, scaling::AbstractFeatureScaling)
    scaled = scaling(data)
    return Profile(scaling, scaled, data)
end

struct Profiles{UU, VV, TT, SS, ΡΡ, DU, DV, DT, DS, DΡ}
       u :: UU
       v :: VV
       T :: TT
       S :: SS
       ρ :: ΡΡ
    ∂u∂z :: DU
    ∂v∂z :: DV
    ∂T∂z :: DT
    ∂S∂z :: DS
    ∂ρ∂z :: DΡ
end

function Profiles(u_data, v_data, T_data, S_data, ρ_data, 
                  ∂u∂z_data, ∂v∂z_data, ∂T∂z_data, ∂S∂z_data, ∂ρ∂z_data,
                  u_scaling::AbstractFeatureScaling, 
                  v_scaling::AbstractFeatureScaling, 
                  T_scaling::AbstractFeatureScaling, 
                  S_scaling::AbstractFeatureScaling, 
                  ρ_scaling::AbstractFeatureScaling,
                  ∂u∂z_scaling::AbstractFeatureScaling,
                  ∂v∂z_scaling::AbstractFeatureScaling,
                  ∂T∂z_scaling::AbstractFeatureScaling,
                  ∂S∂z_scaling::AbstractFeatureScaling,
                  ∂ρ∂z_scaling::AbstractFeatureScaling)
    u = Profile(u_data, u_scaling)
    v = Profile(v_data, v_scaling)
    T = Profile(T_data, T_scaling)
    S = Profile(S_data, S_scaling)
    ρ = Profile(ρ_data, ρ_scaling)
    ∂u∂z = Profile(∂u∂z_data, ∂u∂z_scaling)
    ∂v∂z = Profile(∂v∂z_data, ∂v∂z_scaling)
    ∂T∂z = Profile(∂T∂z_data, ∂T∂z_scaling)
    ∂S∂z = Profile(∂S∂z_data, ∂S∂z_scaling)
    ∂ρ∂z = Profile(∂ρ∂z_data, ∂ρ∂z_scaling)

    return Profiles(u, v, T, S, ρ, ∂u∂z, ∂v∂z, ∂T∂z, ∂S∂z, ∂ρ∂z)
end

struct ProfilesB{UU, VV, BB, ΡΡ, DU, DV, DB, DΡ}
       u :: UU
       v :: VV
       b :: BB
       ρ :: ΡΡ
    ∂u∂z :: DU
    ∂v∂z :: DV
    ∂b∂z :: DB
    ∂ρ∂z :: DΡ
end

function ProfilesB(u_data, v_data, b_data, ρ_data, 
                  ∂u∂z_data, ∂v∂z_data, ∂b∂z_data, ∂ρ∂z_data,
                  u_scaling::AbstractFeatureScaling, 
                  v_scaling::AbstractFeatureScaling, 
                  b_scaling::AbstractFeatureScaling, 
                  ρ_scaling::AbstractFeatureScaling,
                  ∂u∂z_scaling::AbstractFeatureScaling,
                  ∂v∂z_scaling::AbstractFeatureScaling,
                  ∂b∂z_scaling::AbstractFeatureScaling,
                  ∂ρ∂z_scaling::AbstractFeatureScaling)
    u = Profile(u_data, u_scaling)
    v = Profile(v_data, v_scaling)
    b = Profile(b_data, b_scaling)
    ρ = Profile(ρ_data, ρ_scaling)
    ∂u∂z = Profile(∂u∂z_data, ∂u∂z_scaling)
    ∂v∂z = Profile(∂v∂z_data, ∂v∂z_scaling)
    ∂b∂z = Profile(∂b∂z_data, ∂b∂z_scaling)
    ∂ρ∂z = Profile(∂ρ∂z_data, ∂ρ∂z_scaling)

    return ProfilesB(u, v, b, ρ, ∂u∂z, ∂v∂z, ∂b∂z, ∂ρ∂z)
end

abstract type AbstractColumnFlux end

struct ColumnFlux{S, U} <: AbstractColumnFlux
    scaled :: S
    unscaled :: U
end

function ColumnFlux(unscaled_data, scaling::AbstractFeatureScaling)
    scaled = scaling(unscaled_data)
    return ColumnFlux(scaled, unscaled_data)
end

abstract type AbstractBoundaryFlux end

struct BoundaryFlux{S, U} <: AbstractBoundaryFlux
    scaled :: S
    unscaled :: U
end

function BoundaryFlux(unscaled_data, scaling::AbstractFeatureScaling)
    scaled = scaling(unscaled_data)
    return BoundaryFlux(scaled, unscaled_data)
end

abstract type AbstractFlux end

struct Flux{SC, S, B, C} <: AbstractFlux
    scaling :: SC
    surface :: S
    bottom :: B
    column :: C
end

function Flux(data, scaling, surface_flux::Number, bottom_flux::Number)
    return Flux(scaling, BoundaryFlux(surface_flux, scaling), BoundaryFlux(bottom_flux, scaling), ColumnFlux(data, scaling))
end

struct Fluxes{UW, VW, WT, WS}
    uw :: UW
    vw :: VW
    wT :: WT
    wS :: WS
end

struct FluxesB{UW, VW, WB, WΡ}
    uw :: UW
    vw :: VW
    wb :: WB
    wρ :: WΡ
end

function Fluxes(uw_data, vw_data, wT_data, wS_data, 
                uw_surface, vw_surface, wT_surface, wS_surface, 
                uw_bottom, vw_bottom, wT_bottom, wS_bottom,
                uw_scaling::AbstractFeatureScaling, vw_scaling::AbstractFeatureScaling, wT_scaling::AbstractFeatureScaling, wS_scaling::AbstractFeatureScaling)
    uw = Flux(uw_data, uw_scaling, uw_surface, uw_bottom)
    vw = Flux(vw_data, vw_scaling, vw_surface, vw_bottom)
    wT = Flux(wT_data, wT_scaling, wT_surface, wT_bottom)
    wS = Flux(wS_data, wS_scaling, wS_surface, wS_bottom)
    return Fluxes(uw, vw, wT, wS)
end

function FluxesB(uw_data, vw_data, wb_data, wρ_data, 
                 uw_surface, vw_surface, wb_surface, wρ_surface, 
                 uw_bottom, vw_bottom, wb_bottom, wρ_bottom,
                 uw_scaling::AbstractFeatureScaling, vw_scaling::AbstractFeatureScaling, wb_scaling::AbstractFeatureScaling, wρ_scaling::AbstractFeatureScaling)
    uw = Flux(uw_data, uw_scaling, uw_surface, uw_bottom)
    vw = Flux(vw_data, vw_scaling, vw_surface, vw_bottom)
    wb = Flux(wb_data, wb_scaling, wb_surface, wb_bottom)
    wρ = Flux(wρ_data, wρ_scaling, wρ_surface, wρ_bottom)
    return FluxesB(uw, vw, wb, wρ)
end

struct LESData{M, T, P, F, C}
    metadata :: M
    times :: T
    profile :: P
    flux :: F
    coriolis :: C
end

get_surface_fluxes(surface_flux::Number) = surface_flux

struct CoriolisParameter{SC, S, U}
     scaling :: SC
      scaled :: S
    unscaled :: U
end

function CoriolisParameter(data, scaling::AbstractFeatureScaling)
    scaled = scaling(data)
    return CoriolisParameter(scaling, scaled, data)
end

function LESData(data::FieldDataset, scalings::NamedTuple, timeframes, coarse_size=32)
    u = hcat([coarse_grain(interior(data["ubar"][i], 1, 1, :), coarse_size, Center) for i in timeframes]...)
    v = hcat([coarse_grain(interior(data["vbar"][i], 1, 1, :), coarse_size, Center) for i in timeframes]...)
    T = hcat([coarse_grain(interior(data["Tbar"][i], 1, 1, :), coarse_size, Center) for i in timeframes]...)
    S = hcat([coarse_grain(interior(data["Sbar"][i], 1, 1, :), coarse_size, Center) for i in timeframes]...)
    ρ = hcat([coarse_grain(interior(data["ρbar"][i], 1, 1, :), coarse_size, Center) for i in timeframes]...)
    
    uw = hcat([coarse_grain(interior(data["uw"][i], 1, 1, :), coarse_size+1, Face) for i in timeframes]...)
    vw = hcat([coarse_grain(interior(data["vw"][i], 1, 1, :), coarse_size+1, Face) for i in timeframes]...)
    wT = hcat([coarse_grain(interior(data["wT"][i], 1, 1, :), coarse_size+1, Face) for i in timeframes]...)
    wS = hcat([coarse_grain(interior(data["wS"][i], 1, 1, :), coarse_size+1, Face) for i in timeframes]...)

    if data.metadata["momentum_flux"] == 0
        u .= 0
        v .= 0
        uw .= 0
        vw .= 0
    end

    zC = coarse_grain(data["ubar"].grid.zᵃᵃᶜ[1:data["ubar"].grid.Nz], coarse_size, Center)
    D = Dᶠ(coarse_size, zC[2] - zC[1])

    ∂u∂z = hcat([D * u[:, i] for i in axes(u, 2)]...)
    ∂v∂z = hcat([D * v[:, i] for i in axes(v, 2)]...)
    ∂T∂z = hcat([D * T[:, i] for i in axes(T, 2)]...)
    ∂S∂z = hcat([D * S[:, i] for i in axes(S, 2)]...)
    ∂ρ∂z = hcat([D * ρ[:, i] for i in axes(ρ, 2)]...)

    f = data.metadata["coriolis_parameter"]

    uw_surface = get_surface_fluxes(data.metadata["momentum_flux"])
    wT_surface = get_surface_fluxes(data.metadata["temperature_flux"])
    wS_surface = get_surface_fluxes(data.metadata["salinity_flux"])
    
    ## TODO: Ensure this is correct
    # uw_surface = 0
    # wT_surface = 0
    # wS_surface = 0
    vw_surface = 0

    uw_bottom = 0
    vw_bottom = 0
    wT_bottom = 0
    wS_bottom = 0

    profile = Profiles(u, v, T, S, ρ, 
                       ∂u∂z, ∂v∂z, ∂T∂z, ∂S∂z, ∂ρ∂z, 
                       scalings.u, scalings.v, scalings.T, scalings.S, scalings.ρ, 
                       scalings.∂u∂z, scalings.∂v∂z, scalings.∂T∂z, scalings.∂S∂z, scalings.∂ρ∂z)
    flux = Fluxes(uw, vw, wT, wS, 
                  uw_surface, vw_surface, wT_surface, wS_surface, 
                  uw_bottom, vw_bottom, wT_bottom, wS_bottom, scalings.uw, 
                  scalings.vw, scalings.wT, scalings.wS)

    coriolis = CoriolisParameter(f, scalings.f)

    metadata = data.metadata
    metadata["original_grid"] = data["ubar"].grid
    metadata["Nz"] = coarse_size
    metadata["zC"] = coarse_grain(data["ubar"].grid.zᵃᵃᶜ[1:data["ubar"].grid.Nz], coarse_size, Center)
    metadata["zF"] = coarse_grain_downsampling(data["ubar"].grid.zᵃᵃᶠ[1:data["ubar"].grid.Nz+1], coarse_size+1, Face)
    metadata["original_times"] = data["ubar"].times

    return LESData(metadata, data["ubar"].times[timeframes], profile, flux, coriolis)
end

function LESDataB(data::FieldDataset, scalings::NamedTuple, timeframes, coarse_size=32)
    u = hcat([coarse_grain(interior(data["ubar"][i], 1, 1, :), coarse_size, Center) for i in timeframes]...)
    v = hcat([coarse_grain(interior(data["vbar"][i], 1, 1, :), coarse_size, Center) for i in timeframes]...)
    b = hcat([coarse_grain(interior(data["bbar"][i], 1, 1, :), coarse_size, Center) for i in timeframes]...)
    ρ = hcat([coarse_grain(interior(data["ρbar"][i], 1, 1, :), coarse_size, Center) for i in timeframes]...)
    
    uw = hcat([coarse_grain(interior(data["uw"][i], 1, 1, :), coarse_size+1, Face) for i in timeframes]...)
    vw = hcat([coarse_grain(interior(data["vw"][i], 1, 1, :), coarse_size+1, Face) for i in timeframes]...)
    wb = hcat([coarse_grain(interior(data["wb"][i], 1, 1, :), coarse_size+1, Face) for i in timeframes]...)
    wρ = hcat([coarse_grain(interior(data["wρ"][i], 1, 1, :), coarse_size+1, Face) for i in timeframes]...)

    if data.metadata["momentum_flux"] == 0
        u .= 0
        v .= 0
        uw .= 0
        vw .= 0
    end

    zC = coarse_grain(data["ubar"].grid.zᵃᵃᶜ[1:data["ubar"].grid.Nz], coarse_size, Center)
    D = Dᶠ(coarse_size, zC[2] - zC[1])

    ∂u∂z = hcat([D * u[:, i] for i in axes(u, 2)]...)
    ∂v∂z = hcat([D * v[:, i] for i in axes(v, 2)]...)
    ∂b∂z = hcat([D * b[:, i] for i in axes(b, 2)]...)
    ∂ρ∂z = hcat([D * ρ[:, i] for i in axes(ρ, 2)]...)

    f = data.metadata["coriolis_parameter"]

    uw_surface = get_surface_fluxes(data.metadata["momentum_flux"])
    wb_surface = get_surface_fluxes(data.metadata["buoyancy_flux"])
    wρ_surface = get_surface_fluxes(data.metadata["density_flux"])
    
    ## TODO: Ensure this is correct
    vw_surface = 0

    uw_bottom = 0
    vw_bottom = 0
    wb_bottom = 0
    wρ_bottom = 0

    profile = ProfilesB(u, v, b, ρ, 
                       ∂u∂z, ∂v∂z, ∂b∂z, ∂ρ∂z, 
                       scalings.u, scalings.v, scalings.b, scalings.ρ, 
                       scalings.∂u∂z, scalings.∂v∂z, scalings.∂b∂z, scalings.∂ρ∂z)

    flux = FluxesB(uw, vw, wb, wρ, 
                  uw_surface, vw_surface, wb_surface, wρ_surface, 
                  uw_bottom, vw_bottom, wb_bottom, wρ_bottom,
                  scalings.uw, scalings.vw, scalings.wb, scalings.wρ)

    coriolis = CoriolisParameter(f, scalings.f)

    metadata = data.metadata
    metadata["original_grid"] = data["ubar"].grid
    metadata["Nz"] = coarse_size
    metadata["zC"] = coarse_grain(data["ubar"].grid.zᵃᵃᶜ[1:data["ubar"].grid.Nz], coarse_size, Center)
    metadata["zF"] = coarse_grain_downsampling(data["ubar"].grid.zᵃᵃᶠ[1:data["ubar"].grid.Nz+1], coarse_size+1, Face)
    metadata["original_times"] = data["ubar"].times

    return LESData(metadata, data["ubar"].times[timeframes], profile, flux, coriolis)
end

struct LESDatasets{D, S}
    data :: D
    scaling :: S
end

"""
    Construct a `LESDatasets` object from a list of `LESData` objects. `scaling` is applied to all `LESData` objects in `datasets`. `tim`
"""
function LESDatasets(datasets::Vector, scaling::Type{<:AbstractFeatureScaling}, timeframes::Vector, coarse_size=32; abs_f=false)
    eos = TEOS10EquationOfState()
    u = [hcat([coarse_grain(interior(data["ubar"][i], 1, 1, :), coarse_size, Center) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    v = [hcat([coarse_grain(interior(data["vbar"][i], 1, 1, :), coarse_size, Center) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    T = [hcat([coarse_grain(interior(data["Tbar"][i], 1, 1, :), coarse_size, Center) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    S = [hcat([coarse_grain(interior(data["Sbar"][i], 1, 1, :), coarse_size, Center) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    ρ = [hcat([coarse_grain(interior(data["ρbar"][i], 1, 1, :), coarse_size, Center) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    
    uw = [hcat([coarse_grain(interior(data["uw"][i], 1, 1, :), coarse_size+1, Face) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    vw = [hcat([coarse_grain(interior(data["vw"][i], 1, 1, :), coarse_size+1, Face) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    wT = [hcat([coarse_grain(interior(data["wT"][i], 1, 1, :), coarse_size+1, Face) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    wS = [hcat([coarse_grain(interior(data["wS"][i], 1, 1, :), coarse_size+1, Face) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]

    αs = [zeros(size(wT_data, 2)) for wT_data in wT]
    βs = [zeros(size(wT_data, 2)) for wT_data in wT]

    for (α, β, T_center, S_center) in zip(αs, βs, T, S)
        α .= SeawaterPolynomials.thermal_expansion.(T_center[end, :], S_center[end, :], 0, Ref(eos))
        β .= SeawaterPolynomials.haline_contraction.(T_center[end, :], S_center[end, :], 0, Ref(eos))
    end

    wbs = [zeros(size(wT_data, 2)) for wT_data in wT]

    for (wb, α, β, data) in zip(wbs, αs, βs, datasets)
        wb .= data.metadata["gravitational_acceleration"] .* (α .* data.metadata["temperature_flux"] .- β .* data.metadata["salinity_flux"])
    end

    for i in eachindex(u)
        if datasets[i].metadata["momentum_flux"] == 0
            u[i] .= 0
            v[i] .= 0
            uw[i] .= 0
            vw[i] .= 0
        end
    end

    zCs = [coarse_grain(data["ubar"].grid.zᵃᵃᶜ[1:data["ubar"].grid.Nz], coarse_size, Center) for data in datasets]
    Dᶠs = [Dᶠ(coarse_size, zC[2] - zC[1]) for zC in zCs]

    ∂u∂z = [hcat([D * u′[:, i] for i in axes(u′, 2)]...) for (D, u′) in zip(Dᶠs, u)]
    ∂v∂z = [hcat([D * v′[:, i] for i in axes(v′, 2)]...) for (D, v′) in zip(Dᶠs, v)]
    ∂T∂z = [hcat([D * T′[:, i] for i in axes(T′, 2)]...) for (D, T′) in zip(Dᶠs, T)]
    ∂S∂z = [hcat([D * S′[:, i] for i in axes(S′, 2)]...) for (D, S′) in zip(Dᶠs, S)]
    ∂ρ∂z = [hcat([D * ρ′[:, i] for i in axes(ρ′, 2)]...) for (D, ρ′) in zip(Dᶠs, ρ)]

    if abs_f
        f = [abs(data.metadata["coriolis_parameter"]) for data in datasets]
    else
        f = [data.metadata["coriolis_parameter"] for data in datasets]
    end

    scalings = (   u = scaling([(u...)...]), 
                   v = scaling([(v...)...]), 
                   T = scaling([(T...)...]), 
                   S = scaling([(S...)...]), 
                   ρ = scaling([(ρ...)...]), 
                  uw = scaling([(uw...)...]), 
                  vw = scaling([(vw...)...]), 
                  wT = scaling([(wT...)...]), 
                  wS = scaling([(wS...)...]),
                  wb = scaling([(wbs...)...]),
                ∂u∂z = scaling([(∂u∂z...)...]),
                ∂v∂z = scaling([(∂v∂z...)...]),
                ∂T∂z = scaling([(∂T∂z...)...]),
                ∂S∂z = scaling([(∂S∂z...)...]),
                ∂ρ∂z = scaling([(∂ρ∂z...)...]),
                   f = scaling([f...]))

    return LESDatasets([LESData(data, scalings, timeframe, coarse_size) for (data, timeframe) in zip(datasets, timeframes)], scalings)
end

function LESDatasets(datasets::Vector, scalings, timeframes::Vector, coarse_size=32; abs_f=false)
    eos = TEOS10EquationOfState()
    u = [hcat([coarse_grain(interior(data["ubar"][i], 1, 1, :), coarse_size, Center) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    v = [hcat([coarse_grain(interior(data["vbar"][i], 1, 1, :), coarse_size, Center) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    T = [hcat([coarse_grain(interior(data["Tbar"][i], 1, 1, :), coarse_size, Center) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    S = [hcat([coarse_grain(interior(data["Sbar"][i], 1, 1, :), coarse_size, Center) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    ρ = [hcat([coarse_grain(interior(data["ρbar"][i], 1, 1, :), coarse_size, Center) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    
    uw = [hcat([coarse_grain(interior(data["uw"][i], 1, 1, :), coarse_size+1, Face) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    vw = [hcat([coarse_grain(interior(data["vw"][i], 1, 1, :), coarse_size+1, Face) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    wT = [hcat([coarse_grain(interior(data["wT"][i], 1, 1, :), coarse_size+1, Face) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    wS = [hcat([coarse_grain(interior(data["wS"][i], 1, 1, :), coarse_size+1, Face) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]

    T_faces = [zeros(size(wT_data, 1), size(wT_data, 2)) for wT_data in wT]
    S_faces = [zeros(size(wT_data, 1), size(wT_data, 2)) for wT_data in wT]

    for (T_face, S_face, T_center, S_center) in zip(T_faces, S_faces, T, S)
        T_face[2:end-1, :] .= 0.5 .* (T_center[1:end-1, :] .+ T_center[2:end, :])
        S_face[2:end-1, :] .= 0.5 .* (S_center[1:end-1, :] .+ S_center[2:end, :])
    end

    αs = [zeros(size(wT_data, 1), size(wT_data, 2)) for wT_data in wT]
    βs = [zeros(size(wT_data, 1), size(wT_data, 2)) for wT_data in wT]

    for (α, β, T_face, S_face, T_center, S_center) in zip(αs, βs, T_faces, S_faces, T, S)
        α[2:end-1, :] .= SeawaterPolynomials.thermal_expansion.(T_face[2:end-1, :], S_face[2:end-1, :], 0, Ref(eos))
        β[2:end-1, :] .= SeawaterPolynomials.haline_contraction.(T_face[2:end-1, :], S_face[2:end-1, :], 0, Ref(eos))
        
        α[end, :] .= SeawaterPolynomials.thermal_expansion.(T_center[end-1, :], S_center[end-1, :], 0, Ref(eos))
        β[end, :] .= SeawaterPolynomials.haline_contraction.(T_center[end-1, :], S_center[end-1, :], 0, Ref(eos))
    end

    wbs = [zeros(size(wT_data, 1), size(wT_data, 2)) for wT_data in wT]

    for (wb, α, β, wT_data, wS_data, data) in zip(wbs, αs, βs, wT, wS, datasets)
        wb[2:end-1, :] .= data.metadata["gravitational_acceleration"] .* (α[2:end-1, :] .* wT_data[2:end-1, :] .- β[2:end-1, :] .* wS_data[2:end-1, :])
        wb[end, :] .= data.metadata["gravitational_acceleration"] .* (α[end, :] .* data.metadata["temperature_flux"] .- β[end, :] .* data.metadata["salinity_flux"])
    end

    for i in eachindex(u)
        if datasets[i].metadata["momentum_flux"] == 0
            u[i] .= 0
            v[i] .= 0
            uw[i] .= 0
            vw[i] .= 0
        end
    end

    zCs = [coarse_grain(data["ubar"].grid.zᵃᵃᶜ[1:data["ubar"].grid.Nz], coarse_size, Center) for data in datasets]
    Dᶠs = [Dᶠ(coarse_size, zC[2] - zC[1]) for zC in zCs]

    ∂u∂z = [hcat([D * u′[:, i] for i in axes(u′, 2)]...) for (D, u′) in zip(Dᶠs, u)]
    ∂v∂z = [hcat([D * v′[:, i] for i in axes(v′, 2)]...) for (D, v′) in zip(Dᶠs, v)]
    ∂T∂z = [hcat([D * T′[:, i] for i in axes(T′, 2)]...) for (D, T′) in zip(Dᶠs, T)]
    ∂S∂z = [hcat([D * S′[:, i] for i in axes(S′, 2)]...) for (D, S′) in zip(Dᶠs, S)]
    ∂ρ∂z = [hcat([D * ρ′[:, i] for i in axes(ρ′, 2)]...) for (D, ρ′) in zip(Dᶠs, ρ)]

    if abs_f
        f = [abs(data.metadata["coriolis_parameter"]) for data in datasets]
    else
        f = [data.metadata["coriolis_parameter"] for data in datasets]
    end

    return LESDatasets([LESData(data, scalings, timeframe, coarse_size) for (data, timeframe) in zip(datasets, timeframes)], scalings)
end

function LESDatasetsB(datasets::Vector, scaling::Type{<:AbstractFeatureScaling}, timeframes::Vector, coarse_size=32)
    u = [hcat([coarse_grain(interior(data["ubar"][i], 1, 1, :), coarse_size, Center) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    v = [hcat([coarse_grain(interior(data["vbar"][i], 1, 1, :), coarse_size, Center) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    b = [hcat([coarse_grain(interior(data["bbar"][i], 1, 1, :), coarse_size, Center) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    ρ = [hcat([coarse_grain(interior(data["ρbar"][i], 1, 1, :), coarse_size, Center) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    
    uw = [hcat([coarse_grain(interior(data["uw"][i], 1, 1, :), coarse_size+1, Face) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    vw = [hcat([coarse_grain(interior(data["vw"][i], 1, 1, :), coarse_size+1, Face) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    wb = [hcat([coarse_grain(interior(data["wb"][i], 1, 1, :), coarse_size+1, Face) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    wρ = [hcat([coarse_grain(interior(data["wρ"][i], 1, 1, :), coarse_size+1, Face) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]

    for i in eachindex(u)
        if datasets[i].metadata["momentum_flux"] == 0
            u[i] .= 0
            v[i] .= 0
            uw[i] .= 0
            vw[i] .= 0
        end
    end

    zCs = [coarse_grain(data["ubar"].grid.zᵃᵃᶜ[1:data["ubar"].grid.Nz], coarse_size, Center) for data in datasets]
    Dᶠs = [Dᶠ(coarse_size, zC[2] - zC[1]) for zC in zCs]

    ∂u∂z = [hcat([D * u′[:, i] for i in axes(u′, 2)]...) for (D, u′) in zip(Dᶠs, u)]
    ∂v∂z = [hcat([D * v′[:, i] for i in axes(v′, 2)]...) for (D, v′) in zip(Dᶠs, v)]
    ∂b∂z = [hcat([D * b′[:, i] for i in axes(b′, 2)]...) for (D, b′) in zip(Dᶠs, b)]
    ∂ρ∂z = [hcat([D * ρ′[:, i] for i in axes(ρ′, 2)]...) for (D, ρ′) in zip(Dᶠs, ρ)]

    f = [data.metadata["coriolis_parameter"] for data in datasets]

    scalings = (   u = scaling([(u...)...]), 
                   v = scaling([(v...)...]), 
                   b = scaling([(b...)...]), 
                   ρ = scaling([(ρ...)...]), 
                  uw = scaling([(uw...)...]), 
                  vw = scaling([(vw...)...]), 
                  wb = scaling([(wb...)...]), 
                  wρ = scaling([(wρ...)...]),
                ∂u∂z = scaling([(∂u∂z...)...]),
                ∂v∂z = scaling([(∂v∂z...)...]),
                ∂b∂z = scaling([(∂b∂z...)...]),
                ∂ρ∂z = scaling([(∂ρ∂z...)...]),
                   f = scaling([f...]))

    return LESDatasets([LESDataB(data, scalings, timeframe, coarse_size) for (data, timeframe) in zip(datasets, timeframes)], scalings)
end