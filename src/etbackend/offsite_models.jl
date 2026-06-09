# Flattened ET offsite (bond) model + system-level pairwise-coupled (PWC) friction
# model. Mirrors PWCMatrixModel (matrixmodels/pwcmatrixmodels.jl) on the ET bond
# basis: per directed bond (i,j) an offsite Σ block; Γ = Σ Σᵀ.

using StaticArrays, LinearAlgebra
using AtomsBase: AbstractSystem, atomic_number

"""
    ETOffsiteModel(basis, c, cutoff)
    ETOffsiteModel(basis, n_rep, cutoff)

Flattened offsite site model: a bond `ETFrictionSiteBasis` + coefficients `c` +
the `EllipsoidCutoff` defining its bond environment.
"""
mutable struct ETOffsiteModel{NR, P, TB}
   basis::TB
   c::Vector{SVector{NR, Float64}}
   cutoff::EllipsoidCutoff{Float64}
end

function ETOffsiteModel(basis::ETFrictionSiteBasis{P}, c::Vector{SVector{NR,Float64}},
                        cutoff::EllipsoidCutoff) where {P, NR}
   @assert length(basis) == length(c)
   return ETOffsiteModel{NR, P, typeof(basis)}(basis, c, EllipsoidCutoff{Float64}(
            cutoff.rcutbond, cutoff.rcutenv, cutoff.zcutenv))
end

ETOffsiteModel(basis::ETFrictionSiteBasis, n_rep::Integer, cutoff::EllipsoidCutoff) =
      ETOffsiteModel(basis, rand(SVector{n_rep,Float64}, length(basis)), cutoff)

n_rep(::ETOffsiteModel{NR}) where {NR} = NR
Base.length(m::ETOffsiteModel) = length(m.basis)
nparams(m::ETOffsiteModel) = length(m.c)
params(m::ETOffsiteModel) = m.c
function set_params!(m::ETOffsiteModel{NR}, c::Vector{SVector{NR,Float64}}) where {NR}
   @assert length(c) == length(m.c); copyto!(m.c, c); return m
end

"""basis blocks on a bond `(rrij, Rs, Zs)` (raw env; cutoff transform applied here)."""
function evaluate_basis(m::ETOffsiteModel, rrij::SVector{3}, Rs, Zs)
   rbond, Rst, Zst = ellipsoid_env_transform(rrij, Rs, Zs, m.cutoff)
   return evaluate_bond(m.basis, rbond, Rst, Zst)
end

"""contracted offsite Σ block per replica for a bond."""
function evaluate(m::ETOffsiteModel{NR}, rrij::SVector{3}, Rs, Zs) where {NR}
   B = evaluate_basis(m, rrij, Rs, Zs)
   TB = block_type(m.basis)
   Σ = zero(MVector{NR, TB})
   @inbounds for k in eachindex(B), r in 1:NR
      Σ[r] += m.c[k][r] * B[k]
   end
   return SVector(Σ)
end

# ---------------------------------------------------------------------------

_msort2(z1, z2) = z1 <= z2 ? (z1, z2) : (z2, z1)

"""
    ETPWCModel(offsite::Dict{Tuple{Int,Int},ETOffsiteModel}; id=:offsite)

Pairwise-coupled offsite friction model: one offsite model per (sorted) species
pair. All models must share `n_rep` and `cutoff`.
"""
struct ETPWCModel{NR, TM}
   offsite::Dict{Tuple{Int,Int}, TM}
   cutoff::EllipsoidCutoff{Float64}
   id::Symbol
end

function ETPWCModel(offsite::Dict{Tuple{Int,Int}, TM}; id::Symbol = :offsite) where {TM}
   NRs = unique(n_rep(m) for m in values(offsite))
   @assert length(NRs) == 1 "offsite models must share n_rep"
   cuts = unique((m.cutoff.rcutbond, m.cutoff.rcutenv, m.cutoff.zcutenv) for m in values(offsite))
   @assert length(cuts) == 1 "offsite models must share cutoff"
   return ETPWCModel{NRs[1], TM}(offsite, first(values(offsite)).cutoff, id)
end

n_rep(::ETPWCModel{NR}) where {NR} = NR

"""
    sigma(M, at) -> Vector{Matrix{SMatrix{3,3}}}

Per-replica N×N block matrix Σ; `Σ[r][i,j]` is the offsite block for bond (i,j).
"""
function sigma(M::ETPWCModel{NR}, at::AbstractSystem) where {NR}
   Z = Int[ Int(atomic_number(at, i)) for i in 1:length(at) ]
   N = length(at)
   Z3 = SMatrix{3,3,Float64,9}
   Σ = [ fill(zero(Z3), N, N) for _ in 1:NR ]
   for (i, j, rrij, Js, Rs, Zs) in et_bonds(at, M.cutoff)
      zz = _msort2(Z[i], Z[j])
      haskey(M.offsite, zz) || continue
      Σij = evaluate(M.offsite[zz], rrij, Rs, Zs)
      for r in 1:NR
         Σ[r][i, j] = Σij[r]
      end
   end
   return Σ
end

"""
    gamma(M, at) -> Matrix{SMatrix{3,3}}  (N×N block friction matrix Γ = Σ Σᵀ)
"""
function gamma(M::ETPWCModel, at::AbstractSystem)
   Σ = sigma(M, at)
   N = size(Σ[1], 1)
   Z3 = SMatrix{3,3,Float64,9}
   Γ = fill(zero(Z3), N, N)
   for Σr in Σ, i in 1:N, k in 1:N
      acc = zero(Z3)
      for j in 1:N
         acc += Σr[i, j] * Σr[k, j]'
      end
      Γ[i, k] += acc
   end
   return Γ
end

function gamma_dense(M::ETPWCModel, at::AbstractSystem)
   Γb = gamma(M, at); N = size(Γb, 1)
   Γ = zeros(3N, 3N)
   for i in 1:N, k in 1:N
      Γ[3i-2:3i, 3k-2:3k] .= Γb[i, k]
   end
   return Γ
end
