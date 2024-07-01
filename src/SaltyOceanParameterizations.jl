module SaltyOceanParameterizations

export 
    find_min, find_max,
    LESData, LESDatasets, LESDatasetsB,
    ODEParam, ODEParams,
    ZeroMeanUnitVarianceScaling, MinMaxScaling, DiffusivityScaling,
    construct_zeromeanunitvariance_scaling,
    Dᶜ, Dᶠ, Dᶜ!, Dᶠ!,
    calculate_Ri, calculate_Ri!,
    local_Ri_ν_convectivetanh_shearlinear, local_Ri_κ_convectivetanh_shearlinear,
    nonlocal_Ri_ν_convectivetanh_shearlinear, nonlocal_Ri_κ_convectivetanh_shearlinear,
    local_Ri_ν_convectivetanh_shearlinear_2Pr, local_Ri_κ_convectivetanh_shearlinear_2Pr,
    nonlocal_Ri_ν_convectivetanh_shearlinear_2Pr, nonlocal_Ri_κ_convectivetanh_shearlinear_2Pr,
    predict_boundary_flux, predict_boundary_flux!, predict_diffusive_flux, predict_diffusive_boundary_flux_dimensional,
    compute_density_contribution, s,
    LES_suite

# Write your package code here.
include("utils.jl")
include("Operators/Operators.jl")
include("DataWrangling/DataWrangling.jl")
include("Closures/Closures.jl")
include("TrainingOperations/TrainingOperations.jl")
include("ODEOperations/ODEOperations.jl")

import SeawaterPolynomials.TEOS10: s, ΔS, Sₐᵤ
s(Sᴬ::Number) = Sᴬ + ΔS >= 0 ? √((Sᴬ + ΔS) / Sₐᵤ) : NaN

using .Operators
using .ODEOperations
using .DataWrangling
using .Closures
using .TrainingOperations

end
