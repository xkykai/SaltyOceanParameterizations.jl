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

function parse_commandline()
    s = ArgParseSettings()
  
    @add_arg_table! s begin
      "--minlon"
        help = "minimum longitude (degrees)"
        arg_type = Float64
        default = -5.
      "--maxlon"
        help = "maximum longitude (degrees)"
        arg_type = Float64
        default = 5.
      "--minlat"
        help = "minimum latitude (degrees)"
        arg_type = Float64
        default = -65.
      "--maxlat"
        help = "maximum latitude (degrees)"
        arg_type = Float64
        default = -55.
      "--month"
        help = "months to average"
        arg_type = String
        default = "DJF"
    end
    return parse_args(s)
end

args = parse_commandline()

minlon = args["minlon"]
maxlon = args["maxlon"]
minlat = args["minlat"]
maxlat = args["maxlat"]

@info args["month"]

# s_ds = NCDataset(glob("ARGO/field/*/*_PSAL.nc", "/storage6/xinkai"))
# T_ds = NCDataset(glob("ARGO/field/*/*_TEMP.nc", "/storage6/xinkai"))

s_ds = NCDataset(glob("ARGO/field/*/*_PSAL.nc"))
T_ds = NCDataset(glob("ARGO/field/*/*_TEMP.nc"))

times = collect(s_ds["time"])
lats = collect(s_ds["latitude"])
lons = collect(s_ds["longitude"])
zs = -collect(s_ds["depth"])

if args["month"] == "DJF"
    month_indices = findall(x -> Dates.month(x) == 1 || Dates.month(x) == 2 || Dates.month(x) == 12, times)
elseif args["month"] == "JJA"
    month_indices = findall(x -> Dates.month(x) == 6 || Dates.month(x) == 7 || Dates.month(x) == 8, times)
end

lat_indices = findall(x -> x >= minlat && x <= maxlat, lats)
lon_indices = findall(x -> x >= minlon && x <= maxlon, lons)

# sbar = mean(s_ds["PSAL"][lon_indices, lat_indices, :, month_indices], dims=(1, 2))
# Tbar = mean(T_ds["TEMP"][lon_indices, lat_indices, :, month_indices], dims=(1, 2))

s = s_ds["PSAL"][lon_indices, lat_indices, :, month_indices]
T = T_ds["TEMP"][lon_indices, lat_indices, :, month_indices]

sbar = zeros(1, 1, size(s)[3:end]...)
Tbar = zeros(1, 1, size(T)[3:end]...)

Threads.@threads for k in axes(sbar, 3)
    for l in axes(sbar, 4)
        # sbar[1, 1, k, l] = mean(skipmissing(s[:, :, k, l]))
        sbar[1, 1, k, l] = mean(s[:, :, k, l])
    end
end

Threads.@threads for k in axes(Tbar, 3)
    for l in axes(Tbar, 4)
        # Tbar[1, 1, k, l] = mean(skipmissing(T[:, :, k, l]))
        Tbar[1, 1, k, l] = mean(T[:, :, k, l])
    end
end

jldopen("./Data/lon_$(minlon)_$(maxlon)_lat_$(minlat)_$(maxlat)_month_$(args["month"]).jld2", "w") do file
    file["S"] = sbar
    file["T"] = Tbar
end

#=
ds.attrib
ds

lons = collect(ds["LONGITUDE"])
lats = collect(ds["LATITUDE"])

ds["PSAL"]
collect(ds["DEPH"])
keys(ds)

ds.dim

ds["REFERENCE_DATE_TIME"]
ds["PLATFORM_NUMBER"] |> collect

varbyattrib(ds, standard_name = "longitude")

scatter(lon, lat)

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

hm1 = surface!(ax, map_lons, map_lats, field; shading = NoShading)
scatter!(ax, lons, lats, color = :black)
translate!(hm1, 0, 0, -10)

hm2 = poly!(
    ax, worldCountries;
    strokecolor = :black,
    strokewidth = 0.25
)


fig
#%%
scatter(ds["PSAL"][:, 1], -collect(ds["DEPH"]))
#%%
ds_field = NCDataset("./ARGO/field/2020/ISAS20_ARGO_20200115_fld_PSAL.nc")

lons = collect(ds_field["longitude"])
lats = collect(ds_field["latitude"])
#%%
path = GeoMakie.assetpath("vector", "countries.geo.json")
json_str = read(path, String)
worldCountries = GeoJSON.read(json_str)
n = length(worldCountries)
map_lons = -180:180
map_lats = -90:90
field = [exp(cosd(l)) + 3(y/90) for l in map_lons, y in map_lats]

fig = Figure(size = (1200,800), fontsize = 22)

ax = GeoAxis(
    fig[1,1];
    dest="+proj=wintri",
    title = "World Countries",
    tellheight = true,
)

grid_points = [(lon, lat) for lon in lons, lat in lats][:]

hm1 = surface!(ax, map_lons, map_lats, field; shading = NoShading)
# scatter!(ax, grid_points, color = :black)
translate!(hm1, 0, 0, -10)

hm2 = poly!(
    ax, worldCountries;
    strokecolor = :black,
    strokewidth = 0.25
)

fig
#%%
=#