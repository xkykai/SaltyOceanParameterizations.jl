using SaltyOceanParameterizations
using SaltyOceanParameterizations.DataWrangling
using SaltyOceanParameterizations: calculate_Ri, local_Ri_ν_piecewise_linear, local_Ri_κ_piecewise_linear
using Oceananigans
using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL, OptimizationOptimisers, Random, SciMLSensitivity, LuxCUDA
using Statistics
using CairoMakie
using SeawaterPolynomials.TEOS10
using Printf
using Dates
using JLD2
using SciMLBase
using Colors
using Distributions
import SeawaterPolynomials.TEOS10: s, ΔS, Sₐᵤ
s(Sᴬ) = Sᴬ + ΔS >= 0 ? √((Sᴬ + ΔS) / Sₐᵤ) : NaN

function find_min(a...)
    return minimum(minimum.([a...]))
end
  
function find_max(a...)
    return maximum(maximum.([a...]))
end

LES_FILE_DIRS = [
    "./LES_training/linearTS_dTdz_0.0013_dSdz_-0.0014_QU_-0.0002_QT_3.0e-5_QS_-3.0e-5_T_4.3_S_33.5_f_-0.00012_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.013_dSdz_0.00075_QU_-0.0002_QT_0.0003_QS_-3.0e-5_T_14.5_S_35.0_f_0.0_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_0.014_dSdz_0.0021_QU_-0.0002_QT_0.0003_QS_-3.0e-5_T_18.0_S_36.6_f_8.0e-5_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
    "./LES_training/linearTS_dTdz_-0.025_dSdz_-0.0045_QU_-0.0002_QT_-0.0003_QS_-3.0e-5_T_-3.6_S_33.9_f_-0.000125_WENO9nu0_Lxz_512.0_256.0_Nxz_256_128/instantaneous_timeseries.jld2",
]

field_datasets = [FieldDataset(FILE_DIR, backend=OnDisk()) for FILE_DIR in LES_FILE_DIRS]

full_timeframes = [25:length(data["ubar"].times) for data in field_datasets]
timeframes = [25:5:length(data["ubar"].times) for data in field_datasets]
train_data = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, timeframes)
coarse_size = 32

train_data_plot = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, full_timeframes)

#%%
ρ₀ = mean(train_data.data[3].profile.ρ.unscaled[:, end])
T₀ = mean(train_data.data[3].profile.T.unscaled[:, end])
S₀ = mean(train_data.data[3].profile.S.unscaled[:, end])
fig = Figure()
ax = CairoMakie.Axis(fig[1,1])
# lines!(ax, (train_data_plot.data[3].profile.ρ.unscaled[:, end] .- ρ₀) ./ ρ₀, train_data_plot.data[3].metadata["zC"], label="density")
lines!(ax, 2e-4 .* (train_data_plot.data[3].profile.T.unscaled[:, end] .- T₀), train_data_plot.data[3].metadata["zC"], label="temperature")
lines!(ax, 8e-4 .* (train_data_plot.data[3].profile.S.unscaled[:, end] .- S₀), train_data_plot.data[3].metadata["zC"], label="salinity")
axislegend(ax, position=:lb)
display(fig)
#%%