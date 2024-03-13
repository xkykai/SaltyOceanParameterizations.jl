module SaltyOceanParameterizations
export Dᶜ, Dᶠ, Dᶜ!, Dᶠ!, calculate_Ri, calculate_Ri!

# Write your package code here.
include("differentiation_operators.jl")
include("DataWrangling/DataWrangling.jl")
include("base_closure.jl")

end
