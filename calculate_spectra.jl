using FFTW
using Oceananigans
using Statistics
using LinearAlgebra
using SparseArrays
using JLD2

FILE_DIRs = [
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_32_64_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_32_64_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_32_64_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_32_64_f",

    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_32_64_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_32_64_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_32_64_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_32_64_f",

    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_64_128_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_64_128_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_64_128_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_64_128_f",

    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_64_128_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_64_128_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_64_128_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_64_128_f",

    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_128_256_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_128_256_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_128_256_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_128_256_f",

    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_128_256_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_128_256_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu1e-5_Lxz_64.0_128.0_Nxz_128_256_f",
    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9AMD_Lxz_64.0_128.0_Nxz_128_256_f",

    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.001_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_256_512_f",

    # "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_256_512_f"
    "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_WENO9nu0_Lxz_64.0_128.0_Nxz_256_512_f"
]

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

function calculate_spectra(FILE_DIRs::Vector{String})
    for FILE_DIR in FILE_DIRs
        @info "Calculating spectra for $(FILE_DIR)"
        b_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_b.jld2", "b")
        u_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_u.jld2", "u")
        v_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_v.jld2", "v")
        w_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_w.jld2", "w")

        face_to_center = spzeros(size(interior(u_data[1]), 3), size(interior(w_data[1]), 3))

        for i in axes(face_to_center, 1)
            face_to_center[i, i:i+1] .= 0.5
        end

        w_center_data = zeros(size(u_data))

        Threads.@threads for l in axes(w_center_data, 4)
            for i in axes(w_center_data, 1), j in axes(w_center_data, 2)
                w_center_data[i, j, :, l] .= face_to_center * interior(w_data[l], i, j, :)
            end
        end

        b_spectras = calculate_spectra(b_data)
        u_spectras = calculate_spectra(u_data)
        v_spectras = calculate_spectra(v_data)
        w_center_spectras = calculate_spectra(w_center_data, b_data.grid.Nx, b_data.grid.Ny, b_data.grid.Nz, b_data.grid.Δxᶜᵃᵃ, b_data.grid.Δyᵃᶜᵃ)

        jldsave("$(FILE_DIR)/spectra.jld2"; b=b_spectras, u=u_spectras, v=v_spectras, w=w_center_spectras)
    end
end

calculate_spectra(FILE_DIRs)