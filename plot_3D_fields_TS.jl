using Oceananigans
using CairoMakie
using ColorSchemes

# filename = "linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_7.5e-5_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_64.0_64.0_Nxz_256_256"
filename = "linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.0003_QT_0.0_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_64.0_64.0_Nxz_256_256"
FILE_DIR = "./LES/$(filename)"

T_xy_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xy.jld2", "T", backend=OnDisk())
T_xz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz.jld2", "T", backend=OnDisk())
T_yz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz.jld2", "T", backend=OnDisk())

S_xy_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xy.jld2", "S", backend=OnDisk())
S_xz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz.jld2", "S", backend=OnDisk())
S_yz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz.jld2", "S", backend=OnDisk())

ρ_xy_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xy.jld2", "ρ", backend=OnDisk())
ρ_xz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz.jld2", "ρ", backend=OnDisk())
ρ_yz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz.jld2", "ρ", backend=OnDisk())

w_xy_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xy.jld2", "w", backend=OnDisk())
w_xz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz.jld2", "w", backend=OnDisk())
w_yz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz.jld2", "w", backend=OnDisk())

ubar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "ubar")
vbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "vbar")
Tbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "Tbar")
Sbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "Sbar")
bbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar")
ρbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "ρbar")

ns = [300, 800, 1441]
times = T_xy_data.times ./ (24*3600)

time_frames = T_xy_data.times[ns]

Nx, Ny, Nz = T_xy_data.grid.Nx, T_xy_data.grid.Ny, T_xy_data.grid.Nz
xC, yC, zC = T_xy_data.grid.xᶜᵃᵃ[1:Nx], T_xy_data.grid.yᵃᶜᵃ[1:Ny], T_xy_data.grid.zᵃᵃᶜ[1:Nz]
zF = T_xy_data.grid.zᵃᵃᶠ[1:Nz+1]

Lx, Ly, Lz = T_xy_data.grid.Lx, T_xy_data.grid.Ly, T_xy_data.grid.Lz

xCs_xy = xC
yCs_xy = yC
zCs_xy = [zC[Nz] for x in xCs_xy, y in yCs_xy]

yCs_yz = yC
xCs_yz = range(xC[1], stop=xC[1], length=length(zC))
zCs_yz = zeros(length(xCs_yz), length(yCs_yz))
for j in axes(zCs_yz, 2)
  zCs_yz[:, j] .= zC
end

xCs_xz = xC
yCs_xz = range(yC[1], stop=yC[1], length=length(zC))
zCs_xz = zeros(length(xCs_xz), length(yCs_xz))
for i in axes(zCs_xz, 1)
  zCs_xz[i, :] .= zC
end

xFs_xy = xC
yFs_xy = yC
zFs_xy = [zF[Nz+1] for x in xFs_xy, y in yFs_xy]

yFs_yz = yC
xFs_yz = range(xC[1], stop=xC[1], length=length(zF))
zFs_yz = zeros(length(xFs_yz), length(yFs_yz))
for j in axes(zFs_yz, 2)
  zFs_yz[:, j] .= zF
end

xFs_xz = xC
yFs_xz = range(yC[1], stop=yC[1], length=length(zF))
zFs_xz = zeros(length(xFs_xz), length(yFs_xz))
for i in axes(zFs_xz, 1)
  zFs_xz[i, :] .= zF
end

function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

T1_xy = interior(T_xy_data[ns[1]], :, :, 1)
T1_yz = transpose(interior(T_yz_data[ns[1]], 1, :, :))
T1_xz = interior(T_xz_data[ns[1]], :, 1, :)

T2_xy = interior(T_xy_data[ns[2]], :, :, 1)
T2_yz = transpose(interior(T_yz_data[ns[2]], 1, :, :))
T2_xz = interior(T_xz_data[ns[2]], :, 1, :)

T3_xy = interior(T_xy_data[ns[3]], :, :, 1)
T3_yz = transpose(interior(T_yz_data[ns[3]], 1, :, :))
T3_xz = interior(T_xz_data[ns[3]], :, 1, :)

S1_xy = interior(S_xy_data[ns[1]], :, :, 1)
S1_yz = transpose(interior(S_yz_data[ns[1]], 1, :, :))
S1_xz = interior(S_xz_data[ns[1]], :, 1, :)

S2_xy = interior(S_xy_data[ns[2]], :, :, 1)
S2_yz = transpose(interior(S_yz_data[ns[2]], 1, :, :))
S2_xz = interior(S_xz_data[ns[2]], :, 1, :)

S3_xy = interior(S_xy_data[ns[3]], :, :, 1)
S3_yz = transpose(interior(S_yz_data[ns[3]], 1, :, :))
S3_xz = interior(S_xz_data[ns[3]], :, 1, :)

ρ1_xy = interior(ρ_xy_data[ns[1]], :, :, 1)
ρ1_yz = transpose(interior(ρ_yz_data[ns[1]], 1, :, :))
ρ1_xz = interior(ρ_xz_data[ns[1]], :, 1, :)

ρ2_xy = interior(ρ_xy_data[ns[2]], :, :, 1)
ρ2_yz = transpose(interior(ρ_yz_data[ns[2]], 1, :, :))
ρ2_xz = interior(ρ_xz_data[ns[2]], :, 1, :)

ρ3_xy = interior(ρ_xy_data[ns[3]], :, :, 1)
ρ3_yz = transpose(interior(ρ_yz_data[ns[3]], 1, :, :))
ρ3_xz = interior(ρ_xz_data[ns[3]], :, 1, :)

w1_xy = interior(w_xy_data[ns[1]], :, :, 1)
w1_yz = transpose(interior(w_yz_data[ns[1]], 1, :, :))
w1_xz = interior(w_xz_data[ns[1]], :, 1, :)

w2_xy = interior(w_xy_data[ns[2]], :, :, 1)
w2_yz = transpose(interior(w_yz_data[ns[2]], 1, :, :))
w2_xz = interior(w_xz_data[ns[2]], :, 1, :)

w3_xy = interior(w_xy_data[ns[3]], :, :, 1)
w3_yz = transpose(interior(w_yz_data[ns[3]], 1, :, :))
w3_xz = interior(w_xz_data[ns[3]], :, 1, :)

# for freeconvection
# startheight = 64

# for wind mixing
startheight = 56
Tlim = (find_min(T1_xz[:, startheight:end], T2_xz[:, startheight:end], T3_xz[:, startheight:end]), find_max(T1_xz[:, startheight:end], T2_xz[:, startheight:end], T3_xz[:, startheight:end]))
Slim = (find_min(S1_xz[:, startheight:end], S2_xz[:, startheight:end], S3_xz[:, startheight:end]), find_max(S1_xz[:, startheight:end], S2_xz[:, startheight:end], S3_xz[:, startheight:end]))
ρlim = (find_min(ρ1_xz[:, startheight:end], ρ2_xz[:, startheight:end], ρ3_xz[:, startheight:end]), find_max(ρ1_xz[:, startheight:end], ρ2_xz[:, startheight:end], ρ3_xz[:, startheight:end]))
wlim = (-maximum(abs, [find_min(w1_yz, w2_yz, w3_yz), find_max(w1_yz, w2_yz, w3_yz)]), 
        maximum(abs, [find_min(w1_yz, w2_yz, w3_yz), find_max(w1_yz, w2_yz, w3_yz)]))

colorscheme = colorschemes[:balance]
T_colormap = colorscheme
S_colormap = colorscheme
ρ_colormap = colorscheme
w_colormap = colorscheme

T_color_range = Tlim
S_color_range = Slim
ρ_color_range = ρlim
w_color_range = wlim

with_theme(theme_latexfonts()) do
    fig = Figure(size=(2000, 1400), fontsize=20)
    axT1 = Axis3(fig[1, 1], title="Time = $(round(times[ns][1], digits=3)) days", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data)
    axT2 = Axis3(fig[1, 2], title="Time = $(round(times[ns][2], digits=3)) days", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data)
    axT3 = Axis3(fig[1, 3], title="Time = $(round(times[ns][3], digits=3)) days", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data)

    axS1 = Axis3(fig[2, 1], xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data)
    axS2 = Axis3(fig[2, 2], xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data)
    axS3 = Axis3(fig[2, 3], xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data)

    axρ1 = Axis3(fig[3, 1], xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data)
    axρ2 = Axis3(fig[3, 2], xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data)
    axρ3 = Axis3(fig[3, 3], xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data)

    axw1 = Axis3(fig[4, 1], xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data)
    axw2 = Axis3(fig[4, 2], xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data)
    axw3 = Axis3(fig[4, 3], xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data)

    axTbar = CairoMakie.Axis(fig[1, 5], xlabel=L"$\overline{T}$ (°C)", ylabel="z (m)")
    axSbar = CairoMakie.Axis(fig[2, 5], xlabel=L"$\overline{S}$ (g kg$^{-1}$)", ylabel="z (m)")
    axρbar = CairoMakie.Axis(fig[3, 5], xlabel=L"$\overline{\sigma}$ (kg m$^{-3}$)", ylabel="z (m)", xticks=LinearTicks(3))
    axubar = CairoMakie.Axis(fig[1, 6], xlabel=L"$\overline{u}$ (m s$^{-1}$)", ylabel="z (m)")
    axvbar = CairoMakie.Axis(fig[2, 6], xlabel=L"$\overline{v}$ (m s$^{-1}$)", ylabel="z (m)")

    T1_xy_surface = surface!(axT1, xCs_xy, yCs_xy, zCs_xy, color=T1_xy, colormap=T_colormap, colorrange = T_color_range, lowclip=T_colormap[1])
    T1_yz_surface = surface!(axT1, xCs_yz, yCs_yz, zCs_yz, color=T1_yz, colormap=T_colormap, colorrange = T_color_range, lowclip=T_colormap[1])
    T1_xz_surface = surface!(axT1, xCs_xz, yCs_xz, zCs_xz, color=T1_xz, colormap=T_colormap, colorrange = T_color_range, lowclip=T_colormap[1])

    T2_xy_surface = surface!(axT2, xCs_xy, yCs_xy, zCs_xy, color=T2_xy, colormap=T_colormap, colorrange = T_color_range, lowclip=T_colormap[1])
    T2_yz_surface = surface!(axT2, xCs_yz, yCs_yz, zCs_yz, color=T2_yz, colormap=T_colormap, colorrange = T_color_range, lowclip=T_colormap[1])
    T2_xz_surface = surface!(axT2, xCs_xz, yCs_xz, zCs_xz, color=T2_xz, colormap=T_colormap, colorrange = T_color_range, lowclip=T_colormap[1])

    T3_xy_surface = surface!(axT3, xCs_xy, yCs_xy, zCs_xy, color=T3_xy, colormap=T_colormap, colorrange = T_color_range, lowclip=T_colormap[1])
    T3_yz_surface = surface!(axT3, xCs_yz, yCs_yz, zCs_yz, color=T3_yz, colormap=T_colormap, colorrange = T_color_range, lowclip=T_colormap[1])
    T3_xz_surface = surface!(axT3, xCs_xz, yCs_xz, zCs_xz, color=T3_xz, colormap=T_colormap, colorrange = T_color_range, lowclip=T_colormap[1])

    S1_xy_surface = surface!(axS1, xCs_xy, yCs_xy, zCs_xy, color=S1_xy, colormap=S_colormap, colorrange = S_color_range, lowclip=S_colormap[1])
    S1_yz_surface = surface!(axS1, xCs_yz, yCs_yz, zCs_yz, color=S1_yz, colormap=S_colormap, colorrange = S_color_range, lowclip=S_colormap[1])
    S1_xz_surface = surface!(axS1, xCs_xz, yCs_xz, zCs_xz, color=S1_xz, colormap=S_colormap, colorrange = S_color_range, lowclip=S_colormap[1])

    S2_xy_surface = surface!(axS2, xCs_xy, yCs_xy, zCs_xy, color=S2_xy, colormap=S_colormap, colorrange = S_color_range, lowclip=S_colormap[1])
    S2_yz_surface = surface!(axS2, xCs_yz, yCs_yz, zCs_yz, color=S2_yz, colormap=S_colormap, colorrange = S_color_range, lowclip=S_colormap[1])
    S2_xz_surface = surface!(axS2, xCs_xz, yCs_xz, zCs_xz, color=S2_xz, colormap=S_colormap, colorrange = S_color_range, lowclip=S_colormap[1])

    S3_xy_surface = surface!(axS3, xCs_xy, yCs_xy, zCs_xy, color=S3_xy, colormap=S_colormap, colorrange = S_color_range, lowclip=S_colormap[1])
    S3_yz_surface = surface!(axS3, xCs_yz, yCs_yz, zCs_yz, color=S3_yz, colormap=S_colormap, colorrange = S_color_range, lowclip=S_colormap[1])
    S3_xz_surface = surface!(axS3, xCs_xz, yCs_xz, zCs_xz, color=S3_xz, colormap=S_colormap, colorrange = S_color_range, lowclip=S_colormap[1])

    ρ1_xy_surface = surface!(axρ1, xCs_xy, yCs_xy, zCs_xy, color=ρ1_xy, colormap=ρ_colormap, colorrange = ρ_color_range, highclip=ρ_colormap[end])
    ρ1_yz_surface = surface!(axρ1, xCs_yz, yCs_yz, zCs_yz, color=ρ1_yz, colormap=ρ_colormap, colorrange = ρ_color_range, highclip=ρ_colormap[end])
    ρ1_xz_surface = surface!(axρ1, xCs_xz, yCs_xz, zCs_xz, color=ρ1_xz, colormap=ρ_colormap, colorrange = ρ_color_range, highclip=ρ_colormap[end])

    ρ2_xy_surface = surface!(axρ2, xCs_xy, yCs_xy, zCs_xy, color=ρ2_xy, colormap=ρ_colormap, colorrange = ρ_color_range, highclip=ρ_colormap[end])
    ρ2_yz_surface = surface!(axρ2, xCs_yz, yCs_yz, zCs_yz, color=ρ2_yz, colormap=ρ_colormap, colorrange = ρ_color_range, highclip=ρ_colormap[end])
    ρ2_xz_surface = surface!(axρ2, xCs_xz, yCs_xz, zCs_xz, color=ρ2_xz, colormap=ρ_colormap, colorrange = ρ_color_range, highclip=ρ_colormap[end])

    ρ3_xy_surface = surface!(axρ3, xCs_xy, yCs_xy, zCs_xy, color=ρ3_xy, colormap=ρ_colormap, colorrange = ρ_color_range, highclip=ρ_colormap[end])
    ρ3_yz_surface = surface!(axρ3, xCs_yz, yCs_yz, zCs_yz, color=ρ3_yz, colormap=ρ_colormap, colorrange = ρ_color_range, highclip=ρ_colormap[end])
    ρ3_xz_surface = surface!(axρ3, xCs_xz, yCs_xz, zCs_xz, color=ρ3_xz, colormap=ρ_colormap, colorrange = ρ_color_range, highclip=ρ_colormap[end])

    w1_xy_surface = surface!(axw1, xFs_xy, yFs_xy, zFs_xy, color=w1_xy, colormap=w_colormap, colorrange = w_color_range)
    w1_yz_surface = surface!(axw1, xFs_yz, yFs_yz, zFs_yz, color=w1_yz, colormap=w_colormap, colorrange = w_color_range)
    w1_xz_surface = surface!(axw1, xFs_xz, yFs_xz, zFs_xz, color=w1_xz, colormap=w_colormap, colorrange = w_color_range)

    w2_xy_surface = surface!(axw2, xFs_xy, yFs_xy, zFs_xy, color=w2_xy, colormap=w_colormap, colorrange = w_color_range)
    w2_yz_surface = surface!(axw2, xFs_yz, yFs_yz, zFs_yz, color=w2_yz, colormap=w_colormap, colorrange = w_color_range)
    w2_xz_surface = surface!(axw2, xFs_xz, yFs_xz, zFs_xz, color=w2_xz, colormap=w_colormap, colorrange = w_color_range)

    w3_xy_surface = surface!(axw3, xFs_xy, yFs_xy, zFs_xy, color=w3_xy, colormap=w_colormap, colorrange = w_color_range)
    w3_yz_surface = surface!(axw3, xFs_yz, yFs_yz, zFs_yz, color=w3_yz, colormap=w_colormap, colorrange = w_color_range)
    w3_xz_surface = surface!(axw3, xFs_xz, yFs_xz, zFs_xz, color=w3_xz, colormap=w_colormap, colorrange = w_color_range)

    Colorbar(fig[1,4], T1_xy_surface, label="Temperature (°C)")
    Colorbar(fig[2,4], S1_xy_surface, label="Salinity (g kg⁻¹)")
    Colorbar(fig[3,4], ρ1_xy_surface, label="Potential Density (kg m⁻³)")
    Colorbar(fig[4,4], w1_xy_surface, label="w (m s⁻¹)")

    lines!(axTbar, interior(Tbar_data[1], 1, 1, :), zC, label="$(round(Tbar_data.times[1] / 24 / 60^2, digits=3)) days")
    lines!(axTbar, interior(Tbar_data[ns[1]], 1, 1, :), zC, label="$(round(times[ns][1], digits=3)) days")
    lines!(axTbar, interior(Tbar_data[ns[2]], 1, 1, :), zC, label="$(round(times[ns][2], digits=3)) days")
    lines!(axTbar, interior(Tbar_data[ns[3]], 1, 1, :), zC, label="$(round(times[ns][3], digits=3)) days")

    lines!(axSbar, interior(Sbar_data[1], 1, 1, :), zC, label="$(round(Sbar_data.times[1] / 24 / 60^2, digits=3)) days")
    lines!(axSbar, interior(Sbar_data[ns[1]], 1, 1, :), zC, label="$(round(times[ns][1], digits=3)) days")
    lines!(axSbar, interior(Sbar_data[ns[2]], 1, 1, :), zC, label="$(round(times[ns][2], digits=3)) days")
    lines!(axSbar, interior(Sbar_data[ns[3]], 1, 1, :), zC, label="$(round(times[ns][3], digits=3)) days")

    lines!(axubar, interior(ubar_data[1], 1, 1, :), zC, label="$(round(ubar_data.times[1] / 24 / 60^2, digits=3)) days")
    lines!(axubar, interior(ubar_data[ns[1]], 1, 1, :), zC, label="$(round(times[ns][1], digits=3)) days")
    lines!(axubar, interior(ubar_data[ns[2]], 1, 1, :), zC, label="$(round(times[ns][2], digits=3)) days")
    lines!(axubar, interior(ubar_data[ns[3]], 1, 1, :), zC, label="$(round(times[ns][3], digits=3)) days")

    lines!(axvbar, interior(vbar_data[1], 1, 1, :), zC, label="$(round(vbar_data.times[1] / 24 / 60^2, digits=3)) days")
    lines!(axvbar, interior(vbar_data[ns[1]], 1, 1, :), zC, label="$(round(times[ns][1], digits=3)) days")
    lines!(axvbar, interior(vbar_data[ns[2]], 1, 1, :), zC, label="$(round(times[ns][2], digits=3)) days")
    lines!(axvbar, interior(vbar_data[ns[3]], 1, 1, :), zC, label="$(round(times[ns][3], digits=3)) days")

    lines!(axρbar, interior(ρbar_data[1], 1, 1, :), zC, label="$(round(ρbar_data.times[1] / 24 / 60^2, digits=3)) days")
    lines!(axρbar, interior(ρbar_data[ns[1]], 1, 1, :), zC, label="$(round(times[ns][1], digits=3)) days")
    lines!(axρbar, interior(ρbar_data[ns[2]], 1, 1, :), zC, label="$(round(times[ns][2], digits=3)) days")
    lines!(axρbar, interior(ρbar_data[ns[3]], 1, 1, :), zC, label="$(round(times[ns][3], digits=3)) days")
    
    # for free convection case only
    # xlims!(axubar, (-0.05, 0.05))
    # xlims!(axvbar, (-0.05, 0.05))

    xlims!(axT1, (0, Lx))
    xlims!(axT2, (0, Lx))
    xlims!(axT3, (0, Lx))
    xlims!(axS1, (0, Lx))
    xlims!(axS2, (0, Lx))
    xlims!(axS3, (0, Lx))
    xlims!(axρ1, (0, Lx))
    xlims!(axρ2, (0, Lx))
    xlims!(axρ3, (0, Lx))
    xlims!(axw1, (0, Lx))
    xlims!(axw2, (0, Lx))
    xlims!(axw3, (0, Lx))

    ylims!(axT1, (0, Ly))
    ylims!(axT2, (0, Ly))
    ylims!(axT3, (0, Ly))
    ylims!(axS1, (0, Ly))
    ylims!(axS2, (0, Ly))
    ylims!(axS3, (0, Ly))
    ylims!(axρ1, (0, Ly))
    ylims!(axρ2, (0, Ly))
    ylims!(axρ3, (0, Ly))
    ylims!(axw1, (0, Ly))
    ylims!(axw2, (0, Ly))
    ylims!(axw3, (0, Ly))

    zlims!(axT1, (-Lz, 0))
    zlims!(axT2, (-Lz, 0))
    zlims!(axT3, (-Lz, 0))
    zlims!(axS1, (-Lz, 0))
    zlims!(axS2, (-Lz, 0))
    zlims!(axS3, (-Lz, 0))
    zlims!(axρ1, (-Lz, 0))
    zlims!(axρ2, (-Lz, 0))
    zlims!(axρ3, (-Lz, 0))
    zlims!(axw1, (-Lz, 0))
    zlims!(axw2, (-Lz, 0))
    zlims!(axw3, (-Lz, 0))

    hidedecorations!(axTbar, ticks=false, ticklabels=false, label=false)
    hidedecorations!(axSbar, ticks=false, ticklabels=false, label=false)
    hidedecorations!(axρbar, ticks=false, ticklabels=false, label=false)
    hidedecorations!(axubar, ticks=false, ticklabels=false, label=false)
    hidedecorations!(axvbar, ticks=false, ticklabels=false, label=false)

    Legend(fig[3, 6], axρbar, tellwidth=false)
    display(fig)
    # save("./Data/test_LES_figure.png", fig, px_per_unit=8)
    # save("./figures/freeconvection_snapshot.png", fig, px_per_unit=8)
    save("./figures/windmixing_snapshot.png", fig, px_per_unit=8)
end