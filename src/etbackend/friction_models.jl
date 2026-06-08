# System-level onsite-only friction model on the ET backend.
#
# This is the first fully runnable ET-based friction model: given an AtomsBase
# system it assembles the block-diagonal Σ (per replica) and the friction matrix
# Γ = Σ Σᵀ. It mirrors `OnsiteOnlyMatrixModel` (matrixmodels/onsiteonlymatrixmodels.jl)
# but on flattened ET site models, with neighbours from NeighbourLists.
#
# Scope: matrix-equivariant onsite blocks (the clean PSD friction case). Invariant/
# vector specialisations and the full MatrixModels integration come with the
# live-code migration.

using StaticArrays, LinearAlgebra
using AtomsBase: AbstractSystem, atomic_number
using NeighbourLists: PairList, sites
using Unitful: @u_str

# JuLIP-free neighbour helpers (same as MatrixModels._species / _sites)
_species_vec(at::AbstractSystem) = Int[ Int(atomic_number(at, i)) for i in 1:length(at) ]
_sites_iter(at::AbstractSystem, rcut::Real) = sites(PairList(at, rcut * u"Å"))

"""
    ETOnsiteOnlyModel(onsite::Dict{Int,ETOnsiteModel}, rcut; id=:onsite)

Block-diagonal friction model: one onsite ET site model per centre species.
"""
struct ETOnsiteOnlyModel{NR, TM}
   onsite::Dict{Int, TM}
   rcut::Float64
   id::Symbol
end

function ETOnsiteOnlyModel(onsite::Dict{Int, TM}; id::Symbol = :onsite) where {TM}
   NRs = unique(n_rep(m) for m in values(onsite))
   @assert length(NRs) == 1 "all onsite models must share n_rep"
   rcuts = unique(m.basis.rbasis.trans.rcut for m in values(onsite))
   @assert length(rcuts) == 1 "all onsite models must share rcut"
   return ETOnsiteOnlyModel{NRs[1], TM}(onsite, rcuts[1], id)
end

n_rep(::ETOnsiteOnlyModel{NR}) where {NR} = NR

"""
    sigma(M, at) -> Vector{Vector{SMatrix{3,3}}}

Per-replica block-diagonal Σ: `Σ[r][i]` is the 3×3 onsite block at atom `i`.
"""
function sigma(M::ETOnsiteOnlyModel{NR}, at::AbstractSystem) where {NR}
   Z = _species_vec(at); N = length(at)
   Z3 = SMatrix{3,3,Float64,9}
   Σ = [ fill(zero(Z3), N) for _ in 1:NR ]
   for (i, neigs, Rs) in _sites_iter(at, M.rcut)
      (haskey(M.onsite, Z[i]) && length(neigs) > 0) || continue
      Σi = evaluate(M.onsite[Z[i]], Rs, Z[neigs])    # SVector{NR, SMatrix{3,3}}
      for r in 1:NR
         Σ[r][i] = Σi[r]
      end
   end
   return Σ
end

"""
    gamma(M, at) -> Vector{SMatrix{3,3}}

Block-diagonal friction matrix Γ: `Γ[i] = Σᵣ Σ[r][i] Σ[r][i]ᵀ` (3×3 PSD per atom).
"""
function gamma(M::ETOnsiteOnlyModel, at::AbstractSystem)
   Σ = sigma(M, at)
   N = length(at)
   return [ sum(Σ[r][i] * Σ[r][i]' for r in eachindex(Σ)) for i in 1:N ]
end

"""dense `3N×3N` friction matrix (block-diagonal) for inspection/testing."""
function gamma_dense(M::ETOnsiteOnlyModel, at::AbstractSystem)
   Γb = gamma(M, at); N = length(Γb)
   Γ = zeros(3N, 3N)
   for i in 1:N
      Γ[3i-2:3i, 3i-2:3i] .= Γb[i]
   end
   return Γ
end
