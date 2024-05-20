using Oceananigans
# using CairoMakie
using GLMakie
using ColorSchemes

# filename = "linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_7.5e-5_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_64.0_64.0_Nxz_256_256"
filename = "linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.0003_QT_0.0_QS_0.0_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_64.0_64.0_Nxz_256_256"
# FILE_DIR = "/storage6/xinkai/LES/$(filename)"
FILE_DIR = "./LES/$(filename)"

T_xy_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xy.jld2", "T")
T_xz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz.jld2", "T")
T_yz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz.jld2", "T")

S_xy_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xy.jld2", "S")
S_xz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz.jld2", "S")
S_yz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz.jld2", "S")

ρ_xy_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xy.jld2", "ρ")
ρ_xz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz.jld2", "ρ")
ρ_yz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz.jld2", "ρ")

w_xy_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xy.jld2", "w")
w_xz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz.jld2", "w")
w_yz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz.jld2", "w")

ubar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "ubar")
vbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "vbar")
Tbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "Tbar")
Sbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "Sbar")
bbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar")
ρbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "ρbar")

times = T_xy_data.times ./ 3600
Nt = length(times)
timeframes = 1:Nt

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

# for freeconvection
# startheight = 64

# for wind mixing
startheight = 56
Tlim = (find_min(interior(T_xy_data, :, :, 1, timeframes), interior(T_yz_data, 1, :, startheight:Nz, timeframes), interior(T_xz_data, :, 1, startheight:Nz, timeframes)), 
        find_max(interior(T_xy_data, :, :, 1, timeframes), interior(T_yz_data, 1, :, startheight:Nz, timeframes), interior(T_xz_data, :, 1, startheight:Nz, timeframes)))
Slim = (find_min(interior(S_xy_data, :, :, 1, timeframes), interior(S_yz_data, 1, :, startheight:Nz, timeframes), interior(S_xz_data, :, 1, startheight:Nz, timeframes)), 
        find_max(interior(S_xy_data, :, :, 1, timeframes), interior(S_yz_data, 1, :, startheight:Nz, timeframes), interior(S_xz_data, :, 1, startheight:Nz, timeframes)))
ρlim = (find_min(interior(ρ_xy_data, :, :, 1, timeframes), interior(ρ_yz_data, 1, :, startheight:Nz, timeframes), interior(ρ_xz_data, :, 1, startheight:Nz, timeframes)),
        find_max(interior(ρ_xy_data, :, :, 1, timeframes), interior(ρ_yz_data, 1, :, startheight:Nz, timeframes), interior(ρ_xz_data, :, 1, startheight:Nz, timeframes)))

Tbarlim = (minimum(Tbar_data), maximum(Tbar_data))
Sbarlim = (minimum(Sbar_data), maximum(Sbar_data))
ρbarlim = (minimum(ρbar_data), maximum(ρbar_data))

colorscheme = colorschemes[:balance]
T_colormap = colorscheme
S_colormap = colorscheme
ρ_colormap = colorscheme

T_color_range = Tlim
S_color_range = Slim
ρ_color_range = ρlim
#%%
with_theme(theme_latexfonts()) do
    fig = Figure(size=(1920, 980), fontsize=20)
    # fig = Figure(size=(1000, 1000), fontsize=20)
    # fig = Figure(size=(3480, 2000), fontsize=20)
    axT = Axis3(fig[1, 1], title="Temperature (°C)", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data)
    axS = Axis3(fig[1, 3], title="Salinity (g kg⁻¹)", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data)
    axρ = Axis3(fig[1, 5], title="Potential Density (kg m⁻³)", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data)

    # axTbar = CairoMakie.Axis(fig[2, 1:2], xlabel=L"$\overline{T}$ (°C)", ylabel="z (m)")
    # axSbar = CairoMakie.Axis(fig[2, 3:4], xlabel=L"$\overline{S}$ (g kg$^{-1}$)", ylabel="z (m)")
    # axρbar = CairoMakie.Axis(fig[2, 5:6], xlabel=L"$\overline{\sigma}$ (kg m$^{-3}$)", ylabel="z (m)", xticks=LinearTicks(3))
    axTbar = GLMakie.Axis(fig[2, 1:2], xlabel=L"$\overline{T}$ (°C)", ylabel="z (m)")
    axSbar = GLMakie.Axis(fig[2, 3:4], xlabel=L"$\overline{S}$ (g kg$^{-1}$)", ylabel="z (m)")
    axρbar = GLMakie.Axis(fig[2, 5:6], xlabel=L"$\overline{\sigma}$ (kg m$^{-3}$)", ylabel="z (m)", xticks=LinearTicks(3))

    n = Observable(30)
    
    T_xy = @lift interior(T_xy_data[$n], :, :, 1)
    T_yz = @lift transpose(interior(T_yz_data[$n], 1, :, :))
    T_xz = @lift interior(T_xz_data[$n], :, 1, :)

    S_xy = @lift interior(S_xy_data[$n], :, :, 1)
    S_yz = @lift transpose(interior(S_yz_data[$n], 1, :, :))
    S_xz = @lift interior(S_xz_data[$n], :, 1, :)

    ρ_xy = @lift interior(ρ_xy_data[$n], :, :, 1)
    ρ_yz = @lift transpose(interior(ρ_yz_data[$n], 1, :, :))
    ρ_xz = @lift interior(ρ_xz_data[$n], :, 1, :)

    Tbar = @lift interior(Tbar_data[$n], 1, 1, :)
    Sbar = @lift interior(Sbar_data[$n], 1, 1, :)
    ρbar = @lift interior(ρbar_data[$n], 1, 1, :)

    # time_str = @lift "Surface Cooling, Time = $(round(times[$n], digits=2)) hours"
    time_str = @lift "Surface Wind Stress, Time = $(round(times[$n], digits=2)) hours"
    Label(fig[0, :], text=time_str, tellwidth=false, font=:bold)

    T_xy_surface = surface!(axT, xCs_xy, yCs_xy, zCs_xy, color=T_xy, colormap=T_colormap, colorrange = T_color_range, lowclip=T_colormap[1])
    T_yz_surface = surface!(axT, xCs_yz, yCs_yz, zCs_yz, color=T_yz, colormap=T_colormap, colorrange = T_color_range, lowclip=T_colormap[1])
    T_xz_surface = surface!(axT, xCs_xz, yCs_xz, zCs_xz, color=T_xz, colormap=T_colormap, colorrange = T_color_range, lowclip=T_colormap[1])

    S_xy_surface = surface!(axS, xCs_xy, yCs_xy, zCs_xy, color=S_xy, colormap=S_colormap, colorrange = S_color_range, lowclip=S_colormap[1])
    S_yz_surface = surface!(axS, xCs_yz, yCs_yz, zCs_yz, color=S_yz, colormap=S_colormap, colorrange = S_color_range, lowclip=S_colormap[1])
    S_xz_surface = surface!(axS, xCs_xz, yCs_xz, zCs_xz, color=S_xz, colormap=S_colormap, colorrange = S_color_range, lowclip=S_colormap[1])

    ρ_xy_surface = surface!(axρ, xCs_xy, yCs_xy, zCs_xy, color=ρ_xy, colormap=ρ_colormap, colorrange = ρ_color_range, highclip=ρ_colormap[end])
    ρ_yz_surface = surface!(axρ, xCs_yz, yCs_yz, zCs_yz, color=ρ_yz, colormap=ρ_colormap, colorrange = ρ_color_range, highclip=ρ_colormap[end])
    ρ_xz_surface = surface!(axρ, xCs_xz, yCs_xz, zCs_xz, color=ρ_xz, colormap=ρ_colormap, colorrange = ρ_color_range, highclip=ρ_colormap[end])

    Colorbar(fig[1,2], T_xy_surface)
    Colorbar(fig[1,4], S_xy_surface)
    Colorbar(fig[1,6], ρ_xy_surface)

    lines!(axTbar, Tbar, zC, linewidth=3)
    lines!(axSbar, Sbar, zC, linewidth=3)
    lines!(axρbar, ρbar, zC, linewidth=3)
    
    xlims!(axT, (0, Lx))
    xlims!(axS, (0, Lx))
    xlims!(axρ, (0, Lx))

    ylims!(axT, (0, Ly))
    ylims!(axS, (0, Ly))
    ylims!(axρ, (0, Ly))

    zlims!(axT, (-Lz, 0))
    zlims!(axS, (-Lz, 0))
    zlims!(axρ, (-Lz, 0))

    xlims!(axTbar, Tbarlim)
    xlims!(axSbar, Sbarlim)
    xlims!(axρbar, ρbarlim)

    hidedecorations!(axTbar, ticks=false, ticklabels=false, label=false)
    hidedecorations!(axSbar, ticks=false, ticklabels=false, label=false)
    hidedecorations!(axρbar, ticks=false, ticklabels=false, label=false)

    linkyaxes!(axTbar, axSbar, axρbar)

    # display(fig)
    # save("./slides_figures/LES_freeconvection_test.png", fig)

    # GLMakie.record(fig, "./slides_figures/LES_freeconvection_test_1_2xres.mp4", 1:Nt, framerate=30, px_per_unit=2) do nn
    # GLMakie.record(fig, "./slides_figures/LES_freeconvection_test_1.mp4", 1:Nt, framerate=30) do nn
    GLMakie.record(fig, "./slides_figures/LES_windmixing_test_1_2xres.mp4", 1:Nt, framerate=30, px_per_unit=2) do nn
      @info nn
      n[] = nn
    end
end
#%%