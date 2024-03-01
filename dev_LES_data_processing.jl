using SaltyOceanParameterizations.DataWrangling
using SaltyOceanParameterizations.DataWrangling: Profile, Profiles, LESData, coarse_grain, coarse_grain_downsampling, LESDatasets
using Oceananigans
using JLD2

u_data = rand(10, 5)
v_data = rand(10, 5)
T_data = rand(10, 5)
S_data = rand(10, 5)

u_scaling = ZeroMeanUnitVarianceScaling(u_data)
v_scaling = ZeroMeanUnitVarianceScaling(v_data)
T_scaling = ZeroMeanUnitVarianceScaling(T_data)
S_scaling = ZeroMeanUnitVarianceScaling(S_data)

uw_scaling = ZeroMeanUnitVarianceScaling(rand(10))
vw_scaling = ZeroMeanUnitVarianceScaling(rand(10))
wT_scaling = ZeroMeanUnitVarianceScaling(rand(10))
wS_scaling = ZeroMeanUnitVarianceScaling(rand(10))

scalings = (u=u_scaling, v=v_scaling, T=T_scaling, S=S_scaling, uw=uw_scaling, vw=vw_scaling, wT=wT_scaling, wS=wS_scaling)

Profile(u_data, u_scaling)

profile = Profiles(u_data, v_data, T_data, S_data, u_scaling, v_scaling, T_scaling, S_scaling)
profile.S.unscaled

field_dataset = FieldDataset("./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_32_64_f/instantaneous_timeseries.jld2", backend=OnDisk())

times = field_dataset["ubar"].times

hcat([interior(field_dataset["uw"][i], 1, 1, :) for i in eachindex(times)]...)

file = jldopen("./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_32_64_f/instantaneous_timeseries.jld2")

field_dataset = FieldDataset("./LES/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0003_QS_3.0e-5_T_18.0_S_36.6_WENO9nu0_Lxz_256.0_128.0_Nxz_256_128/instantaneous_timeseries.jld2", backend=OnDisk())
field_datasets = [FieldDataset("./LES/linearTS_dTdz_0.014_dSdz_0.0021_QU_0.0_QT_0.0003_QS_3.0e-5_T_18.0_S_36.6_WENO9nu0_Lxz_256.0_128.0_Nxz_256_128/instantaneous_timeseries.jld2", backend=OnDisk()),
                  FieldDataset("./LES/linearTS_dTdz_-0.025_dSdz_-0.0045_QU_0.0_QT_-0.0003_QS_-3.0e-5_T_-3.6_S_33.9_WENO9nu0_Lxz_256.0_128.0_Nxz_256_128/instantaneous_timeseries.jld2", backend=OnDisk())]

timeframes = [1:10, 1:5]
datasets = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, timeframes)
