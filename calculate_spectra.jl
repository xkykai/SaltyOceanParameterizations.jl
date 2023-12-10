using FFTW
using Oceananigans
using Statistics
using LinearAlgebra
using SparseArrays
using JLD2

FILE_DIRs = ["./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_32_64_f"]

b_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_b.jld2", "b") for FILE_DIR in FILE_DIRs]
u_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_u.jld2", "u") for FILE_DIR in FILE_DIRs]
v_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_v.jld2", "v") for FILE_DIR in FILE_DIRs]
w_datas = [FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_w.jld2", "w") for FILE_DIR in FILE_DIRs]

face_to_centers = [spzeros(size(interior(u_datas[i][1]), 3), size(interior(w_datas[i][1]), 3)) for i in eachindex(FILE_DIRs)]

for face_to_center in face_to_centers
    for i in axes(face_to_center, 1)
        face_to_center[i, i:i+1] .= 0.5
    end
end

w_center_datas = [zeros(size(data)) for data in u_datas]

for (num, w_center) in enumerate(w_center_datas)
    for i in axes(w_center, 1), j in axes(w_center, 2), l in axes(w_center, 4)
        w_center[i, j, :, l] .= face_to_centers[num] * interior(w_datas[num][l], i, j, :)
    end
end

function calculate_spectra(data::FieldTimeSeries)
    Nx, Ny, Nz = data.grid.Nx, data.grid.Ny, data.grid.Nz
    Δx, Δy, Δz = data.grid.Δxᶜᵃᵃ, data.grid.Δyᵃᶜᵃ, data.grid.Δzᵃᵃᶜ

    Nt_coarse = size(data, 4)

    kx = fftshift(rfftfreq(Nx, 1/Δx))
    ky = fftshift(rfftfreq(Ny, 1/Δy))

    spectra_kx = cat([rfft(mean(interior(data[i]), dims=2), 1) for i in 1:Nt_coarse]..., dims=4)
    spectra_ky = cat([rfft(mean(interior(data[i]), dims=1), 2) for i in 1:Nt_coarse]..., dims=4)

    spectra²_kx = real.(spectra_kx .* conj.(spectra_kx)) ./ Nx^2
    spectra²_ky = real.(spectra_ky .* conj.(spectra_ky)) ./ Ny^2

    return (; spectra²_kx, spectra²_ky, kx, ky)
end

function calculate_spectra(data, Nx, Ny, Nz, Δx, Δy)
    Nt_coarse = size(data, 4)

    kx = fftshift(rfftfreq(Nx, 1/Δx))
    ky = fftshift(rfftfreq(Ny, 1/Δy))

    spectra_kx = cat([rfft(mean(@view(data[:, :, :, i]), dims=2), 1) for i in 1:Nt_coarse]..., dims=4)
    spectra_ky = cat([rfft(mean(@view(data[:, :, :, i]), dims=1), 2) for i in 1:Nt_coarse]..., dims=4)

    spectra²_kx = real.(spectra_kx .* conj.(spectra_kx)) ./ Nx^2
    spectra²_ky = real.(spectra_ky .* conj.(spectra_ky)) ./ Ny^2

    return (; spectra²_kx, spectra²_ky, kx, ky)
end

b_spectras = [calculate_spectra(data) for data in b_datas]
u_spectras = [calculate_spectra(data) for data in u_datas]
v_spectras = [calculate_spectra(data) for data in v_datas]
w_center_spectras = [calculate_spectra(data, b_datas[i].grid.Nx, b_datas[i].grid.Ny, b_datas[i].grid.Nz, b_datas[i].grid.Δxᶜᵃᵃ, b_datas[i].grid.Δyᵃᶜᵃ) for (i, data) in enumerate(w_center_datas)]

for i in eachindex(FILE_DIRs)
    jldsave("$(FILE_DIRs[i])/spectra.jld2"; b=b_spectras[i], u=u_spectras[i], v=v_spectras[i], w=w_center_spectras[i])
end
