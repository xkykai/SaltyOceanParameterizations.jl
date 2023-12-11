using Oceananigans
using CairoMakie
using Statistics
using LinearAlgebra
using JLD2
using FileIO

FILE_DIRS = [
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_32_64_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_32_64_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_32_64_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_32_64_f",

    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_32_64_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_32_64_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_32_64_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_32_64_f",

    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_64_128_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_64_128_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_64_128_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_64_128_f",

    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_64_128_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_64_128_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_64_128_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_64_128_f",

    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_128_256_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_128_256_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_128_256_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_128_256_f",
    
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_128_256_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_128_256_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_128_256_f",
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_128_256_f",
    
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_256_512_f"
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_256_512_f"
]

labels = [
    "AMD, 2m resolution",
    "WENO(9), ν = κ = 0, 2m resolution",
    # "WENO(9), ν = κ = 1e-5, 2m resolution",
    "WENO(9) + AMD, 2m resolution",

    "AMD, 1m resolution",
    "WENO(9), ν = κ = 0, 1m resolution",
    # "WENO(9), ν = κ = 1e-5, 1m resolution",
    "WENO(9) + AMD, 1m resolution",

    "AMD, 0.5m resolution",
    "WENO(9), ν = κ = 0, 0.5m resolution",
    # "WENO(9), ν = κ = 1e-5, 0.5m resolution",
    "WENO(9) + AMD, 0.5m resolution",

    "AMD, 0.25m resolution"
]

parameters = jldopen("$(FILE_DIRS[1])/instantaneous_timeseries.jld2", "r") do file
    return Dict([(key, file["metadata/parameters/$(key)"]) for key in keys(file["metadata/parameters"])])
end 

Qᴮ = parameters["buoyancy_flux"]
Qᵁ = parameters["momentum_flux"]

video_name = "./Data/QU_$(Qᵁ)_QB_$(Qᴮ)_btop_0_AMD_WENO_WENOAMD_spectra.mp4"

b_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_b.jld2", "b", backend=OnDisk()) for FILE_DIR in FILE_DIRS]
u_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_u.jld2", "u", backend=OnDisk()) for FILE_DIR in FILE_DIRS]
v_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_v.jld2", "v", backend=OnDisk()) for FILE_DIR in FILE_DIRS]
w_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_w.jld2", "w", backend=OnDisk()) for FILE_DIR in FILE_DIRS]

bbar_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar") for FILE_DIR in FILE_DIRS]

Nxs = [size(data.grid)[1] for data in bbar_datas]
Nys = [size(data.grid)[2] for data in bbar_datas]
Nzs = [size(data.grid)[3] for data in bbar_datas]

Δxs = [data.grid.Δxᶜᵃᵃ for data in bbar_datas]
Δys = [data.grid.Δyᵃᶜᵃ for data in bbar_datas]

zCs = [data.grid.zᵃᵃᶜ[1:Nzs[i]] for (i, data) in enumerate(bbar_datas)]
zFs = [data.grid.zᵃᵃᶠ[1:Nzs[i]+1] for (i, data) in enumerate(bbar_datas)]

times_coarse = b_datas[1].times[1:end-2]
times_fine = bbar_datas[1].times[1:end-2]

Nt_coarse = length(times_coarse)
Nt_fine = length(times_fine)

b²_spectras = [load("$(FILE_DIR)/spectra.jld2", "b") for FILE_DIR in FILE_DIRS]
u²_spectras = [load("$(FILE_DIR)/spectra.jld2", "u") for FILE_DIR in FILE_DIRS]
v²_spectras = [load("$(FILE_DIR)/spectra.jld2", "v") for FILE_DIR in FILE_DIRS]
w²_spectras = [load("$(FILE_DIR)/spectra.jld2", "w") for FILE_DIR in FILE_DIRS]

TKE_kxs = [0.5 .* (u²_spectras[i].spectra²_kx .+ v²_spectras[i].spectra²_kx .+ w²_spectras[i].spectra²_kx) for i in eachindex(u²_spectras)]
TKE_kys = [0.5 .* (u²_spectras[i].spectra²_ky .+ v²_spectras[i].spectra²_ky .+ w²_spectras[i].spectra²_ky) for i in eachindex(u²_spectras)]

#%%
function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

#%%
zrange = (-50, -2)
# zrange = (-4, -2)

z_levels = [findfirst(z -> z>zrange[1], zC):findlast(z -> z<zrange[2], zC) for zC in zCs]

TKEs_u = [0.5 * [mean(interior(u_datas[i][t], :, :, :) .^ 2) for t in 1:Nt_coarse] for i in eachindex(u_datas)]
TKEs_v = [0.5 * [mean(interior(v_datas[i][t], :, :, :) .^ 2) for t in 1:Nt_coarse] for i in eachindex(v_datas)]
TKEs_w = [0.5 * [mean(interior(w_datas[i][t], :, :, :) .^ 2) for t in 1:Nt_coarse] for i in eachindex(w_datas)]

TKEs = TKEs_u .+ TKEs_v .+ TKEs_w

t_start = length(times_coarse) - 20

#%%
t_window = t_start:length(times_coarse)
n = Observable(t_start)
n_fine = @lift findfirst(x -> x ≈ times_coarse[$n], times_fine)

fig = Figure(size=(2500, 1300))
axbbar = Axis(fig[1, 1], xlabel="b", ylabel="z", title="<b>")

axkx_b = Axis(fig[2, 1], xlabel="kx", ylabel="S(b)", title="b² spectra, y-average", xscale=log10, yscale=log10)
axky_b = Axis(fig[2, 2], xlabel="ky", ylabel="S(b)", title="b² spectra, x-average", xscale=log10, yscale=log10)

axkx_TKE = Axis(fig[3, 1], xlabel="kx", ylabel="S(TKE)", title="0.5 (u² + v² + w²) spectra, y-average", xscale=log10, yscale=log10)
axky_TKE = Axis(fig[3, 2], xlabel="ky", ylabel="S(TKE)", title="0.5 (u² + v² + w²) spectra, x-average", xscale=log10, yscale=log10)

axkx_TKE_horizontal = Axis(fig[2, 3], xlabel="kx", ylabel="S(horizontal TKE)", title="0.5 (u² + v²) spectra, y-average", xscale=log10, yscale=log10)
axky_TKE_horizontal = Axis(fig[2, 4], xlabel="ky", ylabel="S(horizontal TKE)", title="0.5 (u² + v²) spectra, x-average", xscale=log10, yscale=log10)

axkx_TKE_vertical = Axis(fig[3, 3], xlabel="kx", ylabel="S(vertical TKE)", title="0.5 w² spectra, y-average", xscale=log10, yscale=log10)
axky_TKE_vertical = Axis(fig[3, 4], xlabel="ky", ylabel="S(vertical TKE)", title="0.5 w² spectra, x-average", xscale=log10, yscale=log10)

axTKE = Axis(fig[1, 4], xlabel="t (days)", ylabel="TKE", title="Mean TKE", yscale=log10)

times_day = times_coarse ./ 24 ./ 60^2
time = @lift times_coarse[$n] / 24 / 60^2
time_str = @lift "Time = $(round($time, digits=3)) days"
title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)

bbarₙs = [@lift interior(data[$n_fine], 1, 1, :) for data in bbar_datas]

for (i, data) in enumerate(bbar_datas)
    lines!(axbbar, bbarₙs[i], zCs[i], label=labels[i])
end
axislegend(axbbar, position=:lt)

for i in eachindex(b²_spectras)
    lines!(axkx_b, b²_spectras[i].kx[2:end], mean(b²_spectras[i].spectra²_kx[2:end, :, z_levels[i], t_window], dims=(3, 4))[:], label=labels[i])
    lines!(axky_b, b²_spectras[i].ky[2:end], mean(b²_spectras[i].spectra²_ky[:, 2:end, z_levels[i], t_window], dims=(3, 4))[:], label=labels[i])

    lines!(axkx_TKE, b²_spectras[i].kx[2:end], mean(TKE_kxs[i][2:end, :, z_levels[i], t_window], dims=(3, 4))[:], label=labels[i])
    lines!(axky_TKE, b²_spectras[i].ky[2:end], mean(TKE_kys[i][:, 2:end, z_levels[i], t_window], dims=(3, 4))[:], label=labels[i])

    lines!(axkx_TKE_horizontal, b²_spectras[i].kx[2:end], mean(0.5 .* (u²_spectras[i].spectra²_kx[2:end, :, z_levels[i], t_window] .+ v²_spectras[i].spectra²_kx[2:end, :, z_levels[i], t_window]), dims=(3, 4))[:], label=labels[i])
    lines!(axky_TKE_horizontal, b²_spectras[i].ky[2:end], mean(0.5 .* (u²_spectras[i].spectra²_ky[:, 2:end, z_levels[i], t_window] .+ v²_spectras[i].spectra²_ky[:, 2:end, z_levels[i], t_window]), dims=(3, 4))[:], label=labels[i])

    lines!(axkx_TKE_vertical, b²_spectras[i].kx[2:end], mean(0.5 .* w²_spectras[i].spectra²_kx[2:end, :, z_levels[i], t_window], dims=(3, 4))[:], label=labels[i])
    lines!(axky_TKE_vertical, b²_spectras[i].ky[2:end], mean(0.5 .* w²_spectras[i].spectra²_ky[:, 2:end, z_levels[i], t_window], dims=(3, 4))[:], label=labels[i])

    # lines!(axkx_TKE, b²_spectras[i].kx[2:end], b²_spectras[i].kx[2:end] .* mean(TKE_kxs[i][2:end, :, z_levels[i], t_window], dims=(3, 4))[:], label=labels[i])
    # lines!(axky_TKE, b²_spectras[i].ky[2:end], b²_spectras[i].ky[2:end] .* mean(TKE_kys[i][:, 2:end, z_levels[i], t_window], dims=(3, 4))[:], label=labels[i])

    # lines!(axkx_TKE_horizontal, b²_spectras[i].kx[2:end], b²_spectras[i].kx[2:end] .* mean(0.5 .* (u²_spectras[i].spectra²_kx[2:end, :, z_levels[i], t_window] .+ v²_spectras[i].spectra²_kx[2:end, :, z_levels[i], t_window]), dims=(3, 4))[:], label=labels[i])
    # lines!(axky_TKE_horizontal, b²_spectras[i].ky[2:end], b²_spectras[i].ky[2:end] .* mean(0.5 .* (u²_spectras[i].spectra²_ky[:, 2:end, z_levels[i], t_window] .+ v²_spectras[i].spectra²_ky[:, 2:end, z_levels[i], t_window]), dims=(3, 4))[:], label=labels[i])

    # lines!(axkx_TKE_vertical, b²_spectras[i].kx[2:end], b²_spectras[i].kx[2:end] .* mean(0.5 .* w²_spectras[i].spectra²_kx[2:end, :, z_levels[i], t_window], dims=(3, 4))[:], label=labels[i])
    # lines!(axky_TKE_vertical, b²_spectras[i].ky[2:end], b²_spectras[i].ky[2:end] .* mean(0.5 .* w²_spectras[i].spectra²_ky[:, 2:end, z_levels[i], t_window], dims=(3, 4))[:], label=labels[i])

    lines!(axTKE, times_day[2:end], TKEs[i][2:end], label=labels[i])
end

hlines!(axbbar, [zrange[1], zrange[2]], color=:black, linestyle=:dash)
# vlines!(axTKE, [time.val], color=:black, linestyle=:dash)

record(fig, video_name, t_start:Nt_coarse, framerate=1) do nn
    n[] = nn
end

@info "Animation completed"
#%%

#=
using Oceananigans.Grids: φnode

struct Spectrum{S, F}
    spec :: S
    freq :: F
end

import Base

Base.:(+)(s::Spectrum, t::Spectrum) = Spectrum(s.spec .+ t.spec, s.freq)
Base.:(*)(s::Spectrum, t::Spectrum) = Spectrum(s.spec .* t.spec, s.freq)
Base.:(/)(s::Spectrum, t::Int)      = Spectrum(s.spec ./ t, s.freq)

Base.real(s::Spectrum) = Spectrum(real.(s.spec), s.freq)

@inline onefunc(args...)  = 1.0
@inline hann_window(n, N) = sin(π * n / N)^2 

function average_spectra(var::FieldTimeSeries, xlim, ylim; k = 69, spectra = power_spectrum_1d_x, windowing = onefunc)

    xdomain = xnodes(var[1])[xlim]
    ydomain = ynodes(var[1])[ylim]

    Nt = length(var.times)

    spec = spectra(interior(var[1], xlim, ylim, k), xdomain, ydomain; windowing) 

    for i in 2:Nt
        spec.spec .+= spectra(interior(var[i], xlim, ylim, k), xdomain, ydomain).spec 
    end

    spec.spec ./= Nt

    return spec
end

function power_cospectrum_1d_x(var1, var2, x; windowing = onefunc)

    Nx = length(x)
    Nfx = Int64(Nx)
    
    spectra = zeros(ComplexF64, Int(Nfx/2))
    
    dx = x[2] - x[1]

    freqs = fftfreq(Nfx, 1.0 / dx) # 0, +ve freq,-ve freqs (lowest to highest)
    freqs = freqs[1:Int(Nfx/2)] .* 2.0 .* π
    
    windowed_var1 = [var1[i] * windowing(i, Nfx) for i in 1:Nfx]
    windowed_var2 = [var2[i] * windowing(i, Nfx) for i in 1:Nfx]
    fourier1      = fft(windowed_var1) / Nfx
    fourier2      = fft(windowed_var2) / Nfx
    spectra[1]    += fourier1[1] .* conj(fourier2[1]) .+ fourier2[1] .* conj(fourier1[1])

    for m in 2:Int(Nfx/2)
        spectra[m] += fourier1[m] .* conj(fourier2[m]) .+ fourier2[m] .* conj(fourier1[m])
    end
    return Spectrum(spectra, freqs)
end

function power_spectrum_1d_x(var, x; windowing = onefunc)

    Nx = length(x)
    Nfx = Int64(Nx)
    
    spectra = zeros(ComplexF64, Int(Nfx/2))
    
    dx = x[2] - x[1]

    freqs = fftfreq(Nfx, 1.0 / dx) # 0,+ve freq,-ve freqs (lowest to highest)
    freqs = freqs[1:Int(Nfx/2)] .* 2.0 .* π
    
    windowed_var = [var[i] * windowing(i, Nfx) for i in 1:Nfx]
    fourier      = fft(windowed_var) / Nfx
    spectra[1]  += fourier[1] .* conj(fourier[1])

    for m in 2:Int(Nfx/2)
        spectra[m] += 2.0 * fourier[m] * conj(fourier[m]) # factor 2 for neg freq contribution
    end
    return Spectrum(spectra, freqs)
end

@kernel function _compute_zonal_spectra!(Uspec, Vspec, Bspec, Ωspec, WBspec, grid, u, v, ζ, w, b)
    j, k = @index(Global, NTuple)

    Nx = size(grid, 1)

    Uspec[j, k]  = power_spectrum_1d_x(Array(interior(u,  :, j, k)), Array(grid.λᶠᵃᵃ.parent)[1:Nx])
    Vspec[j, k]  = power_spectrum_1d_x(Array(interior(v,  :, j, k)), Array(grid.λᶜᵃᵃ.parent)[1:Nx])
    Bspec[j, k]  = power_spectrum_1d_x(Array(interior(b,  :, j, k)), Array(grid.λᶜᵃᵃ.parent)[1:Nx])
    Ωspec[j, k]  = power_spectrum_1d_x(Array(interior(ζ,  :, j, k)), Array(grid.λᶠᵃᵃ.parent)[1:Nx])
    WBspec[j, k] = power_cospectrum_1d_x(Array(interior(w,  :, j, k)), Array(interior(b, :, j, k)), Array(grid.λᶜᵃᵃ.parent)[1:Nx])
end

@kernel function _update_spectra!(Ufinal, Vfinal, Bfinal, Ωfinal, WBfinal, Uspec, Vspec, Bspec, Ωspec, WBspec)
    j, k = @index(Global, NTuple)

    Ufinal[j, k].spec  .+= Uspec[j, k].spec
    Vfinal[j, k].spec  .+= Vspec[j, k].spec
    Bfinal[j, k].spec  .+= Bspec[j, k].spec
    Ωfinal[j, k].spec  .+= Ωspec[j, k].spec
    WBfinal[j, k].spec .+= WBspec[j, k].spec
end

function compute_spectra(f::Dict, time)
    grid = f[:u].grid

    Nx, Ny, Nz = size(grid)

    U  = Array{Spectrum}(undef, Ny, Nz)
    V  = Array{Spectrum}(undef, Ny, Nz)
    B  = Array{Spectrum}(undef, Ny, Nz)
    Ω  = Array{Spectrum}(undef, Ny, Nz)
    WB = Array{Spectrum}(undef, Ny, Nz)
    ST = Array{Spectrum}(undef, Ny, Nz)
    PV = Array{Spectrum}(undef, Ny, Nz)

    Uspec  = Array{Spectrum}(undef, Ny, Nz)
    Vspec  = Array{Spectrum}(undef, Ny, Nz)
    Bspec  = Array{Spectrum}(undef, Ny, Nz)
    Ωspec  = Array{Spectrum}(undef, Ny, Nz)
    WBspec = Array{Spectrum}(undef, Ny, Nz)
    STspec = Array{Spectrum}(undef, Ny, Nz)
    PVspec = Array{Spectrum}(undef, Ny, Nz)

    u = Field{Face, Center, Center}(grid)
    v = Field{Center, Face, Center}(grid)

    b = Field{Center, Center, Center}(grid)
    set!(u, f[:u][time[1]])
    set!(v, f[:v][time[1]])
    set!(b, f[:b][time[1]])
    fill_halo_regions!((u, v, b))

    w, b = f[:w][time[1]], f[:b][time[1]]

    ζ = compute!(Field(VerticalVorticityOperation((; u, v))))

    Nx, Ny, Nz = size(grid)
    launch!(CPU(), grid, (Ny, Nz), _compute_zonal_spectra!, U, V, B, Ω, WB, grid, u, v, ζ, w, b)

    @show length(time)
    if length(time) > 1
        for t in time[2:end]
            @info "doing time $time"
            set!(u, f[:u][t])
            set!(v, f[:v][t])
            fill_halo_regions!((u, v))
            ζ = compute!(Field(VerticalVorticityOperation((; u, v))))

            w, b = f[:w][t], f[:b][t]

            launch!(CPU(), grid, (Ny, Nz), _compute_zonal_spectra!, Uspec, Vspec, Bspec, Ωspec, WBspec, grid, u, v, ζ, w, b)
            launch!(CPU(), grid, (Ny, Nz), _update_spectra!, U, V, B, Ω, WB, Uspec, Vspec, Bspec, Ωspec, WBspec)
        end
    end

    @info "finished"
    
    return (; U, V, B, Ω, WB)
end
=#