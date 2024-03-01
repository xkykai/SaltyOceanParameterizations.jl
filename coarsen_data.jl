using Oceananigans
using LinearAlgebra
using Statistics
using SparseArrays
using JLD2
using FFTW

# FILE_DIR_COARSE = "/storage6/xinkai/LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_256_512_f"
FILE_DIR_COARSE = "./LES/linearb_turbulencestatistics_dbdz_1.953125e-5_QU_-0.0005_QB_1.0e-7_b_0.0_AMD_Lxz_64.0_128.0_Nxz_256_512_f"

Δx = 0.25
Δy = 0.25
Δz = 0.25
scaling = 4

function coarsen_data(FILE_DIR_COARSE, scaling)
    @info "Loading u data..."
    # u_data = FieldTimeSeries("$(FILE_DIR_COARSE)/instantaneous_fields_u.jld2", "u", backend=OnDisk())
    u_data = FieldTimeSeries("$(FILE_DIR_COARSE)/instantaneous_fields_u.jld2", "u")
    @info "Loading v data..."
    # v_data = FieldTimeSeries("$(FILE_DIR_COARSE)/instantaneous_fields_v.jld2", "v", backend=OnDisk())
    v_data = FieldTimeSeries("$(FILE_DIR_COARSE)/instantaneous_fields_v.jld2", "v")
    @info "Loading w data..."
    # w_data = FieldTimeSeries("$(FILE_DIR_COARSE)/instantaneous_fields_w.jld2", "w", backend=OnDisk())
    w_data = FieldTimeSeries("$(FILE_DIR_COARSE)/instantaneous_fields_w.jld2", "w")
    @info "Loading b data..."
    # b_data = FieldTimeSeries("$(FILE_DIR_COARSE)/instantaneous_fields_b.jld2", "b", backend=OnDisk())
    b_data = FieldTimeSeries("$(FILE_DIR_COARSE)/instantaneous_fields_b.jld2", "b")

    Nx, Ny, Nz = u_data.grid.Nx, u_data.grid.Ny, u_data.grid.Nz
    Nx_coarse, Ny_coarse, Nz_coarse = Int(Nx / scaling), Int(Ny / scaling), Int(Nz / scaling)

    face_to_center = spzeros(Nz, Nz+1)

    for i in axes(face_to_center, 1)
        face_to_center[i, i:i+1] .= 0.5
    end

    w_center_data = zeros(size(u_data))
    Threads.@threads for i in axes(w_center_data, 1)
        @info i
        for j in axes(w_center_data, 2), l in axes(w_center_data, 4)
            w_center_data[i, j, :, l] .= face_to_center * interior(w_data[l], i, j, :)
        end
    end

    u_coarse = zeros(Nx_coarse, Ny_coarse, Nz_coarse, size(u_data, 4))
    v_coarse = zeros(Nx_coarse, Ny_coarse, Nz_coarse, size(v_data, 4))
    w_coarse = zeros(Nx_coarse, Ny_coarse, Nz_coarse, size(w_center_data, 4))
    b_coarse = zeros(Nx_coarse, Ny_coarse, Nz_coarse, size(b_data, 4))

    Threads.@threads for l in axes(u_coarse, 4)
        for i in axes(u_coarse, 1), j in axes(u_coarse, 2), k in axes(u_coarse, 3)
            u_coarse[i, j, k, l] = mean(interior(u_data[l], (i-1)*scaling+1:i*scaling, (j-1)*scaling+1:j*scaling, (k-1)*scaling+1:k*scaling))
            v_coarse[i, j, k, l] = mean(interior(v_data[l], (i-1)*scaling+1:i*scaling, (j-1)*scaling+1:j*scaling, (k-1)*scaling+1:k*scaling))
            w_coarse[i, j, k, l] = mean(w_center_data[(i-1)*scaling+1:i*scaling, (j-1)*scaling+1:j*scaling, (k-1)*scaling+1:k*scaling, l])
            b_coarse[i, j, k, l] = mean(interior(b_data[l], (i-1)*scaling+1:i*scaling, (j-1)*scaling+1:j*scaling, (k-1)*scaling+1:k*scaling))
        end
    end
    return u_coarse, v_coarse, w_coarse, b_coarse
end

@info "Coarsening data..."
u_coarse, v_coarse, w_coarse, b_coarse = coarsen_data(FILE_DIR_COARSE, scaling)

@info "Saving data..."
jldsave("$(FILE_DIR_COARSE)/coarsened_data_$(scaling).jld2"; u=u_coarse, v=v_coarse, w=w_coarse, b=b_coarse)

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

Nx_coarse, Ny_coarse, Nz_coarse = size(u_coarse)
Δx_coarse = Δx * scaling
Δy_coarse = Δy * scaling

@info "Calculating spectra..."
b²_spectras = calculate_spectra(b_coarse, Nx_coarse, Ny_coarse, Nz_coarse, Δx_coarse, Δy_coarse)
u²_spectras = calculate_spectra(u_coarse, Nx_coarse, Ny_coarse, Nz_coarse, Δx_coarse, Δy_coarse)
v²_spectras = calculate_spectra(v_coarse, Nx_coarse, Ny_coarse, Nz_coarse, Δx_coarse, Δy_coarse)
w²_spectras = calculate_spectra(w_coarse, Nx_coarse, Ny_coarse, Nz_coarse, Δx_coarse, Δy_coarse)

@info "Saving spectra..."
jldsave("$(FILE_DIR_COARSE)/coarsened_spectra_$(scaling).jld2"; b=b²_spectras, u=u²_spectras, v=v²_spectras, w=w²_spectras)