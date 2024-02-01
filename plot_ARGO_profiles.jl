using NCDatasets
using CairoMakie, GeoMakie
import Downloads
using GeoMakie.GeoJSON
using GeometryBasics
using GeoMakie.GeoInterface
using Glob
import Dates
using Statistics
using ArgParse
using JLD2

# minlon = -30.
# maxlon = -20.
# minlat = 50.
# maxlat = 60.
# month = "JJA"
# description = "Atlantic, JJA, 50-60N, 20-30W"

# minlon = -5.
# maxlon = 5.
# minlat = -65.
# maxlat = -55.
# month = "DJF"
# description = "Southern Ocean, DJF, 55-65S, 5W-5E"

# minlon = -30.
# maxlon = -20.
# minlat = -5.
# maxlat = 5.
# month = "JJA"
# description = "Atlantic, JJA, 5S-5N, 20-30W"

minlon = -30.
maxlon = -20.
minlat = 30.
maxlat = 40.
month = "JJA"
description = "Atlantic, JJA, 30-40N, 20-30W"

# minlon = -180.
# maxlon = -170.
# minlat = 30.
# maxlat = 40.
# month = "JJA"
# description = "Pacific, JJA, 30-40N, 170-180W"

# minlon = -180.
# maxlon = -170.
# minlat = -5.
# maxlat = 5.
# month = "JJA"
# description = "Pacific, JJA, 5S-5N, 170-180W"

# minlon = -180.
# maxlon = -170.
# minlat = 50.
# maxlat = 60.
# month = "JJA"
# description = "Pacific, JJA, 50-60N, 170-180W"

# filename = "lon_$(minlon)_$(maxlon)_lat_$(minlat)_$(maxlat)_month_$(month)"
filename = "lon_$(minlon)_$(maxlon)_lat_$(minlat)_$(maxlat)_month_$(month)_skipmissing"

FILE_DIR = "./Data/$(filename).jld2"

s_ds = NCDataset(glob("ARGO/field/*/*_PSAL.nc"))
T_ds = NCDataset(glob("ARGO/field/*/*_TEMP.nc"))

zs = -collect(s_ds["depth"])
sbar, Tbar = jldopen(FILE_DIR, "r") do file
    sbar = file["S"]
    Tbar = file["T"]
    return sbar, Tbar
end

fig = Figure(size=(1200, 600))
axS = Axis(fig[1, 1], title="Salinity", ylabel="z (m)")
axT = Axis(fig[1, 2], title="Temperature", ylabel="z (m)")
Label(fig[0, :], description, font=:bold, tellwidth=false)

for i in axes(sbar, 4)
    lines!(axS, sbar[1, 1, :, i], zs, color=:royalblue, alpha=0.3, linewidth=2)
end

for i in axes(Tbar, 4)
    lines!(axT, Tbar[1, 1, :, i], zs, color=:salmon, alpha=0.3, linewidth=2)
end

lines!(axS, mean(sbar, dims=4)[1, 1, :, 1], zs, color=:black, linewidth=5)
lines!(axT, mean(Tbar, dims=4)[1, 1, :, 1], zs, color=:black, linewidth=5)

display(fig)
save("./Data/$(filename).png", fig, px_per_unit=8)

#%%
map_lons = -180:180
map_lats = -90:90
field = [exp(cosd(l)) + 3(y/90) for l in map_lons, y in map_lats]

fig = Figure()
ax = GeoAxis(fig[1,1])
surface!(ax, map_lons, map_lats, field; shading = NoShading)
fig
#%%
path = GeoMakie.assetpath("vector", "countries.geo.json")
json_str = read(path, String)
worldCountries = GeoJSON.read(json_str)
n = length(worldCountries)
map_lons = -180:180
map_lats = -90:90
field = [exp(cosd(l)) + 3(y/90) for l in map_lons, y in map_lats]

# data_locs = [exp(cosd(l)) + 3(y/90) for l in lons, y in lats]

fig = Figure(size = (1200,800), fontsize = 22)

ax = GeoAxis(
    fig[1,1];
    dest="+proj=wintri",
    title = "World Countries",
    tellheight = true,
)
# ax = Axis(fig[1,1])


hm1 = surface!(ax, map_lons, map_lats, field; shading = NoShading)
# scatter!(ax, lons, lats, color = :black)
translate!(hm1, 0, 0, -10)

hm2 = poly!(
    ax, worldCountries;
    strokecolor = :black,
    strokewidth = 0.25
)


fig

save("./Data/world_map.png", fig, px_per_unit=8)
#%%