using Oceananigans: FieldDataset, interior

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

struct Profiles{UU, VV, TT, SS}
    u :: UU
    v :: VV
    T :: TT
    S :: SS
end

function Profiles(u_data, v_data, T_data, S_data, u_scaling::AbstractFeatureScaling, v_scaling::AbstractFeatureScaling, T_scaling::AbstractFeatureScaling, S_scaling::AbstractFeatureScaling)
    u = Profile(u_data, u_scaling)
    v = Profile(v_data, v_scaling)
    T = Profile(T_data, T_scaling)
    S = Profile(S_data, S_scaling)
    return Profiles(u, v, T, S)
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

abstract type AbstractSurfaceFlux end

struct SurfaceFlux{S, U} <: AbstractSurfaceFlux
    scaled :: S
    unscaled :: U
end

function SurfaceFlux(unscaled_data, scaling::AbstractFeatureScaling)
    scaled = scaling(unscaled_data)
    return SurfaceFlux(scaled, unscaled_data)
end

struct SurfaceFluxes{UW, VW, WT, WS}
    uw :: UW
    vw :: VW
    wT :: WT
    wS :: WS
end

abstract type AbstractFlux end

struct Flux{SC, S, C} <: AbstractFlux
    scaling :: SC
    surface :: S
    column :: C
end

function Flux(data, scaling, surface_flux::Number)
    return Flux(scaling, SurfaceFlux(surface_flux, scaling), ColumnFlux(data, scaling))
end

struct Fluxes{UW, VW, WT, WS}
    uw :: UW
    vw :: VW
    wT :: WT
    wS :: WS
end

function Fluxes(uw_data, vw_data, wT_data, wS_data, uw_surface, vw_surface, wT_surface, wS_surface, uw_scaling::AbstractFeatureScaling, vw_scaling::AbstractFeatureScaling, wT_scaling::AbstractFeatureScaling, wS_scaling::AbstractFeatureScaling)
    uw = Flux(uw_data, uw_scaling, uw_surface)
    vw = Flux(vw_data, vw_scaling, vw_surface)
    wT = Flux(wT_data, wT_scaling, wT_surface)
    wS = Flux(wS_data, wS_scaling, wS_surface)
    return Fluxes(uw, vw, wT, wS)
end

struct LESData{M, T, P, F}
    metadata :: M
    times :: T
    profile :: P
    flux :: F
end

get_surface_fluxes(surface_flux::Number) = surface_flux

function LESData(data::FieldDataset, scalings, timeframes)
    u = hcat([interior(data["ubar"][i], 1, 1, :) for i in eachindex(timeframes)]...)
    v = hcat([interior(data["vbar"][i], 1, 1, :) for i in eachindex(timeframes)]...)
    T = hcat([interior(data["Tbar"][i], 1, 1, :) for i in eachindex(timeframes)]...)
    S = hcat([interior(data["Sbar"][i], 1, 1, :) for i in eachindex(timeframes)]...)
    
    uw = hcat([interior(data["uw"][i], 1, 1, :) for i in eachindex(timeframes)]...)
    vw = hcat([interior(data["vw"][i], 1, 1, :) for i in eachindex(timeframes)]...)
    wT = hcat([interior(data["wT"][i], 1, 1, :) for i in eachindex(timeframes)]...)
    wS = hcat([interior(data["wS"][i], 1, 1, :) for i in eachindex(timeframes)]...)
    
    # uw_surface = get_surface_fluxes(data.metadata["momentum_flux"])
    # wT_surface = get_surface_fluxes(data.metadata["temperature_flux"])
    # wS_surface = get_surface_fluxes(data.metadata["salinity_flux"])
    
    uw_surface = 0
    wT_surface = 0
    wS_surface = 0
    vw_surface = 0

    profile = Profiles(u, v, T, S, scalings.u, scalings.v, scalings.T, scalings.S)
    flux = Fluxes(uw, vw, wT, wS, uw_surface, vw_surface, wT_surface, wS_surface, scalings.uw, scalings.vw, scalings.wT, scalings.wS)

    metadata = data.metadata
    metadata["grid"] = data["ubar"].grid
    metadata["times"] = data["ubar"].times

    return LESData(metadata, data["ubar"].times[timeframes], profile, flux)
end