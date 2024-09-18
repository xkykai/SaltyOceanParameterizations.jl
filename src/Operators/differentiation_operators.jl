using SparseArrays

"""
    Dᶜ(N, Δ)

Returns a discrete 1D derivative operator for taking the derivative of a face-centered field with `N+1` grid points and `Δ` grid spacing and producing a cell-centered field with `N` grid points.
"""
function Dᶜ(N, Δ)
    D = zeros(N, N+1)
    for k in 1:N
        D[k, k]   = -1.0
        D[k, k+1] =  1.0
    end
    D .= 1/Δ .* D
    # return sparse(D)
    return D
end

"""
    Dᶜ!(C, F, Δ)

Takes the derivative of a face-centered field `F` with `N+1` grid points and `Δ` grid spacing and stores the result in a cell-centered field `C` with `N` grid points.
"""
function Dᶜ!(C, F, Δ)
    for k in eachindex(C)
        C[k] = (F[k+1] - F[k]) / Δ
    end
end

"""
    Dᶠ(N, Δ)

Returns a discrete 1D derivative operator for taking the derivative of a cell-centered field with `N` grid points and `Δ` grid spacing and producing a face-centered field with `N+1` grid points.
"""
function Dᶠ(N, Δ)
    D = zeros(N+1, N)
    for k in 2:N
        D[k, k-1] = -1.0
        D[k, k]   =  1.0
    end
    D .= 1/Δ .* D
    # return sparse(D)
    return D
end

"""
    Dᶠ!(F, C, Δ)

Takes the derivative of a cell-centered field `C` with `N` grid points and `Δ` grid spacing and stores the result in a face-centered field `F` with `N+1` grid points.
"""
function Dᶠ!(F, C, Δ)
    for k in 2:length(F)-1
        F[k] = (C[k] - C[k-1]) / Δ
    end
    F[1] = 0
    F[end] = 0
end

"""
    D²ᶜ(N, Δ)
Take the second derivative of a cell-centered field with `N` grid points and `Δ` grid spacing.
"""
function D²ᶜ(N, Δ)
   return Dᶜ(N, Δ) * Dᶠ(N, Δ) 
end