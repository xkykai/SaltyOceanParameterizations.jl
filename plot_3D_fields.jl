using Oceananigans
using CairoMakie

FILE_DIR = "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_256_512_f"

b_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_b.jld2", "b", backend=OnDisk())
w_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_w.jld2", "w", backend=OnDisk())

bbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar")

ns = [2, 5, 10]
times = b_data.times ./ (24*3600)
times_fine = bbar_data.times ./ (24*3600)

time_frames = b_data.times[ns]
ns_fine = vcat([findfirst(x -> x ≈ time_frames[i], bbar_data.times) for i in eachindex(time_frames)], [length(times_fine)])
time_frames_fine = bbar_data.times[ns_fine]

Nx, Ny, Nz = b_data.grid.Nx, b_data.grid.Ny, b_data.grid.Nz
xC, yC, zC = b_data.grid.xᶜᵃᵃ[1:Nx], b_data.grid.yᵃᶜᵃ[1:Ny], b_data.grid.zᵃᵃᶜ[1:Nz]
zF = b_data.grid.zᵃᵃᶠ[1:Nz+1]

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


b1_xy = interior(b_data[ns[1]], :, :, Nz)
b1_yz = transpose(interior(b_data[ns[1]], 1, :, :))
b1_xz = interior(b_data[ns[1]], :, 1, :)

b2_xy = interior(b_data[ns[2]], :, :, Nz)
b2_yz = transpose(interior(b_data[ns[2]], 1, :, :))
b2_xz = interior(b_data[ns[2]], :, 1, :)

b3_xy = interior(b_data[ns[3]], :, :, Nz)
b3_yz = transpose(interior(b_data[ns[3]], 1, :, :))
b3_xz = interior(b_data[ns[3]], :, 1, :)

# w1_xy = interior(w_data[ns[1]], :, :, Nz+1)
w1_xy = interior(w_data[ns[1]], :, :, Nz)
w1_yz = transpose(interior(w_data[ns[1]], 1, :, :))
w1_xz = interior(w_data[ns[1]], :, 1, :)

# w2_xy = interior(w_data[ns[2]], :, :, Nz+1)
w2_xy = interior(w_data[ns[2]], :, :, Nz)
w2_yz = transpose(interior(w_data[ns[2]], 1, :, :))
w2_xz = interior(w_data[ns[2]], :, 1, :)

# w3_xy = interior(w_data[ns[3]], :, :, Nz+1)
w3_xy = interior(w_data[ns[3]], :, :, Nz)
w3_yz = transpose(interior(w_data[ns[3]], 1, :, :))
w3_xz = interior(w_data[ns[3]], :, 1, :)

# blim = (find_min(interior(b_data[ns[1]]), interior(b_data[ns[2]]), interior(b_data[ns[3]])), 
        # find_max(interior(b_data[ns[1]]), interior(b_data[ns[2]]), interior(b_data[ns[3]])))

# blim = (find_min(b1_xy, b2_xy, b3_xy), find_max(b1_xy, b2_xy, b3_xy))
blim = (find_min(b1_xz[:, 275:end], b2_xz[:, 275:end], b3_xz[:, 275:end]), find_max(b1_xz[:, 275:end], b2_xz[:, 275:end], b3_xz[:, 275:end]))

wlim = (-maximum(abs, [find_min(w1_yz, w2_yz, w3_yz), find_max(w1_yz, w2_yz, w3_yz)]), 
        maximum(abs, [find_min(w1_yz, w2_yz, w3_yz), find_max(w1_yz, w2_yz, w3_yz)]))

b_colormap = :balance
w_colormap = :balance

b_color_range = blim
w_color_range = wlim

with_theme(theme_latexfonts()) do
    fig = Figure(size=(2000, 1000))
    axb1 = Axis3(fig[1, 1], title="Time = $(round(times[ns][1], digits=3)) days", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data)
    axb2 = Axis3(fig[1, 2], title="Time = $(round(times[ns][2], digits=3)) days", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data)
    axb3 = Axis3(fig[1, 3], title="Time = $(round(times[ns][3], digits=3)) days", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data)

    axw1 = Axis3(fig[2, 1], title="Time = $(round(times[ns][1], digits=3)) days", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data)
    axw2 = Axis3(fig[2, 2], title="Time = $(round(times[ns][2], digits=3)) days", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data)
    axw3 = Axis3(fig[2, 3], title="Time = $(round(times[ns][3], digits=3)) days", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=:data)

    axbbar = Axis(fig[1:2, 5:6], title="Horizontal average", xlabel="Buoyancy (m s⁻²)", ylabel="z (m)")

    b1_xy_surface = surface!(axb1, xCs_xy, yCs_xy, zCs_xy, color=b1_xy, colormap=b_colormap, colorrange = b_color_range)
    b1_yz_surface = surface!(axb1, xCs_yz, yCs_yz, zCs_yz, color=b1_yz, colormap=b_colormap, colorrange = b_color_range)
    b1_xz_surface = surface!(axb1, xCs_xz, yCs_xz, zCs_xz, color=b1_xz, colormap=b_colormap, colorrange = b_color_range)

    b2_xy_surface = surface!(axb2, xCs_xy, yCs_xy, zCs_xy, color=b2_xy, colormap=b_colormap, colorrange = b_color_range)
    b2_yz_surface = surface!(axb2, xCs_yz, yCs_yz, zCs_yz, color=b2_yz, colormap=b_colormap, colorrange = b_color_range)
    b2_xz_surface = surface!(axb2, xCs_xz, yCs_xz, zCs_xz, color=b2_xz, colormap=b_colormap, colorrange = b_color_range)

    b3_xy_surface = surface!(axb3, xCs_xy, yCs_xy, zCs_xy, color=b3_xy, colormap=b_colormap, colorrange = b_color_range)
    b3_yz_surface = surface!(axb3, xCs_yz, yCs_yz, zCs_yz, color=b3_yz, colormap=b_colormap, colorrange = b_color_range)
    b3_xz_surface = surface!(axb3, xCs_xz, yCs_xz, zCs_xz, color=b3_xz, colormap=b_colormap, colorrange = b_color_range)

    w1_xy_surface = surface!(axw1, xFs_xy, yFs_xy, zFs_xy, color=w1_xy, colormap=w_colormap, colorrange = w_color_range)
    w1_yz_surface = surface!(axw1, xFs_yz, yFs_yz, zFs_yz, color=w1_yz, colormap=w_colormap, colorrange = w_color_range)
    w1_xz_surface = surface!(axw1, xFs_xz, yFs_xz, zFs_xz, color=w1_xz, colormap=w_colormap, colorrange = w_color_range)

    w2_xy_surface = surface!(axw2, xFs_xy, yFs_xy, zFs_xy, color=w2_xy, colormap=w_colormap, colorrange = w_color_range)
    w2_yz_surface = surface!(axw2, xFs_yz, yFs_yz, zFs_yz, color=w2_yz, colormap=w_colormap, colorrange = w_color_range)
    w2_xz_surface = surface!(axw2, xFs_xz, yFs_xz, zFs_xz, color=w2_xz, colormap=w_colormap, colorrange = w_color_range)

    w3_xy_surface = surface!(axw3, xFs_xy, yFs_xy, zFs_xy, color=w3_xy, colormap=w_colormap, colorrange = w_color_range)
    w3_yz_surface = surface!(axw3, xFs_yz, yFs_yz, zFs_yz, color=w3_yz, colormap=w_colormap, colorrange = w_color_range)
    w3_xz_surface = surface!(axw3, xFs_xz, yFs_xz, zFs_xz, color=w3_xz, colormap=w_colormap, colorrange = w_color_range)


    Colorbar(fig[1,4], b1_xy_surface, label="Buoyancy (m s⁻²)")
    Colorbar(fig[2,4], w1_xy_surface, label="Vertical velocity (m s⁻¹)")

    lines!(axbbar, interior(bbar_data[ns_fine[1]], 1, 1, :), zC, label="$(round(times_fine[ns_fine[1]], digits=3)) days")
    lines!(axbbar, interior(bbar_data[ns_fine[2]], 1, 1, :), zC, label="$(round(times_fine[ns_fine[2]], digits=3)) days")
    lines!(axbbar, interior(bbar_data[ns_fine[3]], 1, 1, :), zC, label="$(round(times_fine[ns_fine[3]], digits=3)) days")
    lines!(axbbar, interior(bbar_data[ns_fine[4]], 1, 1, :), zC, label="$(round(times_fine[ns_fine[4]], digits=3)) days")
    axislegend(axbbar, position=:rb)
    display(fig)
    save("./Data/b_3D_fields_WENO9nu0.png", fig, px_per_unit=8)
end