using SaltyOceanParameterizations.Operators: Dᶜ, Dᶠ

struct ODEParam{F, FS, T, NT, ST, Z, L, G, CS, DC, DF, DCH, DFH, UW, VW, WT, WS, SC}
                       f :: F
                f_scaled :: FS
                       τ :: T
             N_timesteps :: NT
             scaled_time :: ST
                      zC :: Z
                       H :: L
                       g :: G
             coarse_size :: CS
                      Dᶜ :: DC
                      Dᶠ :: DF
                  Dᶜ_hat :: DCH
                  Dᶠ_hat :: DFH
                      uw :: UW
                      vw :: VW
                      wT :: WT
                      wS :: WS
                 scaling :: SC
end

function ODEParam(data::LESData, scaling; abs_f=false)
    coarse_size = data.metadata["Nz"]
    if abs_f
        f_scaled = scaling.f(abs(data.coriolis.unscaled))
    else
        f_scaled = scaling.f(data.coriolis.unscaled)
    end

    return ODEParam(data.coriolis.unscaled,
                    f_scaled,
                    data.times[end] - data.times[1],
                    length(data.times),
                    (data.times .- data.times[1]) ./ (data.times[end] - data.times[1]),
                    data.metadata["zC"],
                    data.metadata["original_grid"].Lz,
                    data.metadata["gravitational_acceleration"],
                    coarse_size,
                    Dᶜ(coarse_size, data.metadata["zC"][2] - data.metadata["zC"][1]),
                    Dᶠ(coarse_size, data.metadata["zF"][3] - data.metadata["zF"][2]),
                    Dᶜ(coarse_size, data.metadata["zC"][2] - data.metadata["zC"][1]) .* data.metadata["original_grid"].Lz,
                    Dᶠ(coarse_size, data.metadata["zF"][3] - data.metadata["zF"][2]) .* data.metadata["original_grid"].Lz,
                    (scaled = (top=scaling.uw(data.flux.uw.surface.unscaled), bottom=scaling.uw(data.flux.uw.bottom.unscaled)),
                     unscaled = (top=data.flux.uw.surface.unscaled, bottom=data.flux.uw.bottom.unscaled)),
                    (scaled = (top=scaling.vw(data.flux.vw.surface.unscaled), bottom=scaling.vw(data.flux.vw.bottom.unscaled)),
                     unscaled = (top=data.flux.vw.surface.unscaled, bottom=data.flux.vw.bottom.unscaled)),
                    (scaled = (top=scaling.wT(data.flux.wT.surface.unscaled), bottom=scaling.wT(data.flux.wT.bottom.unscaled)),
                     unscaled = (top=data.flux.wT.surface.unscaled, bottom=data.flux.wT.bottom.unscaled)),
                    (scaled = (top=scaling.wS(data.flux.wS.surface.unscaled), bottom=scaling.wS(data.flux.wS.bottom.unscaled)),
                     unscaled = (top=data.flux.wS.surface.unscaled, bottom=data.flux.wS.bottom.unscaled)),
                    scaling)
end

function ODEParams(dataset::LESDatasets, scaling=dataset.scaling; abs_f=false)
    return [ODEParam(data, scaling, abs_f=abs_f) for data in dataset.data]
end