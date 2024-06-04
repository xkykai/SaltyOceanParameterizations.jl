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

# for freeconvection
# startheight = 64

# for wind mixing
startheight = 56
Tlim = (find_min(T1_xz[:, startheight:end], T2_xz[:, startheight:end], T3_xz[:, startheight:end]), find_max(T1_xz[:, startheight:end], T2_xz[:, startheight:end], T3_xz[:, startheight:end]))
Slim = (find_min(S1_xz[:, startheight:end], S2_xz[:, startheight:end], S3_xz[:, startheight:end]), find_max(S1_xz[:, startheight:end], S2_xz[:, startheight:end], S3_xz[:, startheight:end]))

colorscheme = colorschemes[:balance]
T_colormap = colorscheme
S_colormap = colorscheme

T_color_range = Tlim
S_color_range = Slim

#%%
with_theme(theme_latexfonts()) do
    fig = Figure(size=(1500, 1000), fontsize=20)
    axT1 = Axis3(fig[1, 1], title="Time = $(round(times[ns][1], digits=3)) days", titlesize=30, xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data, protrusions=(50, 30, 30, 30))
    axT2 = Axis3(fig[1, 2], title="Time = $(round(times[ns][2], digits=3)) days", titlesize=30, xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data)
    axT3 = Axis3(fig[1, 3], title="Time = $(round(times[ns][3], digits=3)) days", titlesize=30, xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data)

    axS1 = Axis3(fig[2, 1], xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data)
    axS2 = Axis3(fig[2, 2], xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data)
    axS3 = Axis3(fig[2, 3], xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data)

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

    Colorbar(fig[1,4], T1_xy_surface, label="Temperature (°C)", labelfont=:bold, labelsize=30)
    Colorbar(fig[2,4], S1_xy_surface, label="Salinity (g kg⁻¹)", labelfont=:bold, labelsize=30)

    xlims!(axT1, (0, Lx))
    xlims!(axT2, (0, Lx))
    xlims!(axT3, (0, Lx))
    xlims!(axS1, (0, Lx))
    xlims!(axS2, (0, Lx))
    xlims!(axS3, (0, Lx))

    ylims!(axT1, (0, Ly))
    ylims!(axT2, (0, Ly))
    ylims!(axT3, (0, Ly))
    ylims!(axS1, (0, Ly))
    ylims!(axS2, (0, Ly))
    ylims!(axS3, (0, Ly))

    zlims!(axT1, (-Lz, 0))
    zlims!(axT2, (-Lz, 0))
    zlims!(axT3, (-Lz, 0))
    zlims!(axS1, (-Lz, 0))
    zlims!(axS2, (-Lz, 0))
    zlims!(axS3, (-Lz, 0))
    trim!(fig.layout)

    display(fig)

    # save("./poster_figures/freeconvection_snapshot_poster.png", fig, px_per_unit=8)
    save("./poster_figures/windmixing_snapshot_poster.png", fig, px_per_unit=8)
end
#%%