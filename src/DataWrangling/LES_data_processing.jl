using Oceananigans: FieldDataset, interior, Face, Center

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

struct Profiles{UU, VV, TT, SS, ΡΡ}
    u :: UU
    v :: VV
    T :: TT
    S :: SS
    ρ :: ΡΡ
end

function Profiles(u_data, v_data, T_data, S_data, ρ_data, u_scaling::AbstractFeatureScaling, v_scaling::AbstractFeatureScaling, T_scaling::AbstractFeatureScaling, S_scaling::AbstractFeatureScaling, ρ_scaling::AbstractFeatureScaling)
    u = Profile(u_data, u_scaling)
    v = Profile(v_data, v_scaling)
    T = Profile(T_data, T_scaling)
    S = Profile(S_data, S_scaling)
    ρ = Profile(ρ_data, ρ_scaling)
    return Profiles(u, v, T, S, ρ)
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

struct LESData{M, T, P, F}
    metadata :: M
    times :: T
    profile :: P
    flux :: F
end

get_surface_fluxes(surface_flux::Number) = surface_flux

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
    
    uw_surface = get_surface_fluxes(data.metadata["momentum_flux"])
    wT_surface = get_surface_fluxes(data.metadata["temperature_flux"])
    wS_surface = get_surface_fluxes(data.metadata["salinity_flux"])
    
    # uw_surface = 0
    # wT_surface = 0
    # wS_surface = 0
    vw_surface = 0

    uw_bottom = 0
    vw_bottom = 0
    wT_bottom = 0
    wS_bottom = 0

    profile = Profiles(u, v, T, S, ρ, scalings.u, scalings.v, scalings.T, scalings.S, scalings.ρ)
    flux = Fluxes(uw, vw, wT, wS, uw_surface, vw_surface, wT_surface, wS_surface, uw_bottom, vw_bottom, wT_bottom, wS_bottom, scalings.uw, scalings.vw, scalings.wT, scalings.wS)

    metadata = data.metadata
    metadata["original_grid"] = data["ubar"].grid
    metadata["Nz"] = coarse_size
    metadata["zC"] = coarse_grain(data["ubar"].grid.zᵃᵃᶜ[1:data["ubar"].grid.Nz], coarse_size, Center)
    metadata["zF"] = coarse_grain_downsampling(data["ubar"].grid.zᵃᵃᶠ[1:data["ubar"].grid.Nz+1], coarse_size+1, Face)
    metadata["original_times"] = data["ubar"].times

    return LESData(metadata, data["ubar"].times[timeframes], profile, flux)
end

struct LESDatasets{D, S}
    data :: D
    scaling :: S
end

"""
    Construct a `LESDatasets` object from a list of `LESData` objects. `scaling` is applied to all `LESData` objects in `datasets`. `tim`
"""
function LESDatasets(datasets::Vector, scaling::Type{<:AbstractFeatureScaling}, timeframes::Vector, coarse_size=32)
    u = [hcat([coarse_grain(interior(data["ubar"][i], 1, 1, :), coarse_size, Center) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    v = [hcat([coarse_grain(interior(data["vbar"][i], 1, 1, :), coarse_size, Center) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    T = [hcat([coarse_grain(interior(data["Tbar"][i], 1, 1, :), coarse_size, Center) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    S = [hcat([coarse_grain(interior(data["Sbar"][i], 1, 1, :), coarse_size, Center) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    ρ = [hcat([coarse_grain(interior(data["ρbar"][i], 1, 1, :), coarse_size, Center) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    
    uw = [hcat([coarse_grain(interior(data["uw"][i], 1, 1, :), coarse_size+1, Face) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    vw = [hcat([coarse_grain(interior(data["vw"][i], 1, 1, :), coarse_size+1, Face) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    wT = [hcat([coarse_grain(interior(data["wT"][i], 1, 1, :), coarse_size+1, Face) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]
    wS = [hcat([coarse_grain(interior(data["wS"][i], 1, 1, :), coarse_size+1, Face) for i in timeframe]...) for (data, timeframe) in zip(datasets, timeframes)]

    scalings = (u=scaling([(u...)...]), v=scaling([(v...)...]), T=scaling([(T...)...]), S=scaling([(S...)...]), ρ=scaling([(ρ...)...]), uw=scaling([(uw...)...]), vw=scaling([(vw...)...]), wT=scaling([(wT...)...]), wS=scaling([(wS...)...]))

    return LESDatasets([LESData(data, scalings, timeframe, coarse_size) for (data, timeframe) in zip(datasets, timeframes)], scalings)
end

