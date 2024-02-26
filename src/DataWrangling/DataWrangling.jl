module DataWrangling
export ZeroMeanUnitVarianceScaling, MinMaxScaling, LESData, LESDatasets

include("feature_scaling.jl")
include("LES_data_processing.jl")
include("coarse_graining.jl")

end