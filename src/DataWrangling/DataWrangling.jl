module DataWrangling
export ZeroMeanUnitVarianceScaling, MinMaxScaling, DiffusivityScaling
export LESData, LESDatasets

include("feature_scaling.jl")
include("LES_data_processing.jl")
include("coarse_graining.jl")

end