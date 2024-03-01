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
using LsqFit
using SeawaterPolynomials: ρ′
using SeawaterPolynomials.TEOS10

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

# minlon = -30.
# maxlon = -20.
# minlat = 30.
# maxlat = 40.
# month = "JJA"
# description = "Atlantic, JJA, 30-40N, 20-30W"

# minlon = -180.
# maxlon = -170.
# minlat = 30.
# maxlat = 40.
# month = "JJA"
# description = "Pacific, JJA, 30-40N, 170-180W"

minlon = -180.
maxlon = -170.
minlat = -5.
maxlat = 5.
month = "JJA"
description = "Pacific, JJA, 5S-5N, 170-180W"

# minlon = -180.
# maxlon = -170.
# minlat = 50.
# maxlat = 60.
# month = "JJA"
# description = "Pacific, JJA, 50-60N, 170-180W"

filename = "lon_$(minlon)_$(maxlon)_lat_$(minlat)_$(maxlat)_month_$(month)"
# filename = "lon_$(minlon)_$(maxlon)_lat_$(minlat)_$(maxlat)_month_$(month)_skipmissing"

FILE_DIR = "./Data/$(filename).jld2"

s_ds = NCDataset(glob("ARGO/field/*/*_PSAL.nc"))
T_ds = NCDataset(glob("ARGO/field/*/*_TEMP.nc"))

zs = -collect(s_ds["depth"])
sbar, Tbar = jldopen(FILE_DIR, "r") do file
    sbar = file["S"]
    Tbar = file["T"]
    return sbar, Tbar
end

upper_indices = findall(z -> z <= -300 && z >= -500, zs)
# upper_indices = findall(z -> z >= -250, zs)
zs_upper = zs[upper_indices]

sbar_upper = sbar[1, 1, upper_indices, :]
Tbar_upper = Tbar[1, 1, upper_indices, :]

#%%
fig = Figure(size=(1200, 600))
axS = Axis(fig[1, 1], title="Salinity", ylabel="z (m)")
axT = Axis(fig[1, 2], title="Temperature", ylabel="z (m)")
Label(fig[0, :], description, font=:bold, tellwidth=false)

for i in axes(sbar, 4)
    lines!(axS, sbar_upper[:, i], zs_upper, color=:royalblue, alpha=0.3, linewidth=2)
end

for i in axes(Tbar, 4)
    lines!(axT, Tbar_upper[:, i], zs_upper, color=:salmon, alpha=0.3, linewidth=2)
end

lines!(axS, mean(sbar_upper, dims=2)[:, 1], zs_upper, color=:black, linewidth=5)
lines!(axT, mean(Tbar_upper, dims=2)[:, 1], zs_upper, color=:black, linewidth=5)

display(fig)

#%%
linear(x, p) = p[1] .* x .+ p[2]

p0_S = [10., 10.]
p0_T = [10., 10.]

s_fit = curve_fit(linear, zs_upper, mean(sbar_upper, dims=2)[:, 1], p0_S)
T_fit = curve_fit(linear, zs_upper, mean(Tbar_upper, dims=2)[:, 1], p0_T)

s_fit.param
T_fit.param

eos = TEOS10EquationOfState()
ρ′(T_fit.param[2], s_fit.param[2], 0., eos) + eos.reference_density
#%%