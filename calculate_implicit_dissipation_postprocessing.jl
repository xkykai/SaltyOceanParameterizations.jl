using Oceananigans
using CairoMakie

FILE_DIR = "./LES/linearb_experiments_dbdz_0.08_QU_0.0_QB_1.0e-6_b_0.0_f_0.0_WENO9nu0_Lxz_2.0_1.0_Nxz_128_64"

timeseries_data = FieldDataset("$FILE_DIR/instantaneous_timeseries.jld2")
dissipation_data = FieldDataset("$FILE_DIR/instantaneous_dissipation.jld2")
b_data = FieldDataset("$FILE_DIR/instantaneous_fields_b.jld2")

b = b_data["b"]
χᵁ = dissipation_data["χᵁ"]
χⱽ = dissipation_data["χⱽ"]
χᵂ = dissipation_data["χᵂ"]

function compute_implicit_diffusivity(b::FieldTimeSeries, χᵁ::FieldTimeSeries, χⱽ::FieldTimeSeries, χᵂ::FieldTimeSeries)
    κx = FieldTimeSeries{Nothing, Nothing, Center}(b.grid, b.times)
    κy = FieldTimeSeries{Nothing, Nothing, Center}(b.grid, b.times)
    κz = FieldTimeSeries{Nothing, Nothing, Center}(b.grid, b.times)

    for i in eachindex(b.times)
        @info i
        _κx = Field(Average(χᵁ[i] / ∂x(b[i])^2, dims=(1, 2)))
        _κy = Field(Average(χⱽ[i] / ∂y(b[i])^2, dims=(1, 2)))
        _κz = @at (Nothing, Nothing, Center) Field(Average(χᵂ[i] / ∂z(b[i])^2, dims=(1, 2)))
        
        compute!(_κx)
        compute!(_κy)
        compute!(_κz)

        set!(κx[i], _κx)
        set!(κy[i], _κy)
        set!(κz[i], _κz)
    end
    return κx, κy, κz
end

κx, κy, κz = compute_implicit_diffusivity(b, χᵁ, χⱽ, χᵂ)

function compute_average_implicit_dissipation(χᵁ::FieldTimeSeries, χⱽ::FieldTimeSeries, χᵂ::FieldTimeSeries)
    χx = FieldTimeSeries{Nothing, Nothing, Center}(χᵁ.grid, χᵁ.times)
    χy = FieldTimeSeries{Nothing, Nothing, Center}(χᵁ.grid, χᵁ.times)
    χz = FieldTimeSeries{Nothing, Nothing, Center}(χᵁ.grid, χᵁ.times)
    # χz = FieldTimeSeries{Nothing, Nothing, Face}(χᵁ.grid, χᵁ.times)

    for i in eachindex(χᵁ.times)
        @info i
        _χx = Field(Average(χᵁ[i], dims=(1, 2)))
        _χy = Field(Average(χⱽ[i], dims=(1, 2)))
        _χz = Field(Average(@at((Center, Center, Center), χᵂ[i]), dims=(1, 2)))
        
        compute!(_χx)
        compute!(_χy)
        compute!(_χz)

        set!(χx[i], _χx)
        set!(χy[i], _χy)
        set!(χz[i], _χz)
    end
    return χx, χy, χz
end

χx, χy, χz = compute_average_implicit_dissipation(χᵁ, χⱽ, χᵂ)

Nz = b.grid.Nz
Nt = length(b.times)
zC = b.grid.zᵃᵃᶜ[1:Nz]
#%%
fig = Figure()

ax = Axis(fig[1, 1], xlabel="χz", ylabel="z", title="z dissipation")
n = Observable(1)

χzₙ = @lift interior(χz[$n], 1, 1, :)

lines!(ax, χzₙ, zC)

record(fig, "./Data/test_implicit_dissipation.mp4", 1:Nt, framerate=1) do nn
    n[] = nn
    xlims!(ax, nothing, nothing)
end
#%%