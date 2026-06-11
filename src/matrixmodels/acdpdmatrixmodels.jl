# Atom-centred momentum-conserving DPD matrix model. Like an `RWCMatrixModel` but with
# NO independently-fitted onsite block: the diagonal is *derived* from the off-diagonal
# (spherical, atom-centred) blocks so that each column of Σ sums to zero,
#
#     Σ_ij = -∑_{k≠i} Σ_ki   (j = i)        # zero-column-sum diagonal
#     Σ_ij                    (i ≠ j)         # offsite blocks
#
# With the package conventions Γ = Σ Σᵀ and random force F = Σ·R, zero column sums make
# ∑_i F_i = 0 and Γ translation-invariant ⇒ momentum-conserving. Only the offsite
# coefficients are free parameters; the diagonal is a linear function of them, so the
# precompute-B fitting path is unchanged (the diagonal is folded into each basis block).

"""
    ACDPDMatrixModel{O3S, Z2S, SC}

Atom-centred, momentum-conserving DPD matrix model. The diffusion matrix has ordinary
offsite (atom-centred, spherical) off-diagonal blocks and a *derived* diagonal that makes
each column sum to zero,
```math
\\Sigma_{ij} = \\begin{cases} -\\sum_{k\\neq i}\\Sigma_{ki}, & j=i,\\\\ \\Sigma_{ij}, & i\\neq j. \\end{cases}
```
With ``\\Gamma=\\Sigma\\Sigma^T`` and random force ``F=\\Sigma R`` this yields a
translation-invariant friction tensor and momentum-conserving dynamics. Construct via
`mdDPD_ac_matrixmodel`.
"""
struct ACDPDMatrixModel{O3S, Z2S, SC} <: MatrixModel{O3S}
    offsite::OffSiteModels{O3S, Z2S, <:SphericalCutoff}
    n_rep::Int
    inds::SiteInds
    id::Symbol
    function ACDPDMatrixModel(offsite::OffSiteModels{O3S, Z2S, <:SphericalCutoff}, id::Symbol,
                              sc::SC=SpeciesUnCoupled()) where {O3S, Z2S, SC}
        @assert length(unique([_n_rep(mo) for mo in values(offsite)])) == 1
        return new{O3S, Z2S, SC}(offsite, _n_rep(offsite), SiteInds(_get_basisinds(offsite)), id)
    end
end

_get_SC(::ACDPDMatrixModel{O3S, Z2S, SC}) where {O3S, Z2S, SC} = SC

# add the momentum-conserving diagonal Σ_ii = -∑_k Σ_ki (negative column sums) to an
# off-diagonal block matrix `A` (whose diagonal is zero).
function _add_mc_diagonal(A::SparseMatrixCSC)
    colsums = vec(sum(A, dims=1))                 # block-valued column sums (length N)
    return A + spdiagm(0 => -colsums)
end

# ---- Σ assembly: off-diagonal offsite blocks + derived momentum-conserving diagonal ----
function matrix(M::ACDPDMatrixModel{O3S, Z2S, SC}, at::AbstractSystem;
                filter=(_,_)->true, T=Float64) where {O3S, Z2S, SC}
    N = length(at); Z = _species(at)
    Is = [Int[] for _=1:M.n_rep]; Js = [Int[] for _=1:M.n_rep]; Vs = [_block_type(M,T)[] for _=1:M.n_rep]
    for (i, neigs, Rs) in _sites(at, env_cutoff(M.offsite))
        (filter(i, at) && length(neigs) > 0) || continue
        Zs = Z[neigs]
        for (j_loc, j) in enumerate(neigs)
            filter(j, at) || continue
            (Zi, Zj) = _mreduce(Z[i], Z[j], SC); haskey(M.offsite, (Zi, Zj)) || continue
            Σij = evaluate(M.offsite[(Zi, Zj)], j_loc, Rs, Zs)
            for r = 1:M.n_rep; push!(Is[r], i); push!(Js[r], j); push!(Vs[r], Σij[r]); end
        end
    end
    return [ _add_mc_diagonal(sparse(Is[r], Js[r], Vs[r], N, N)) for r = 1:M.n_rep ]
end

# ---- un-contracted basis (diagonal folded into each offsite basis function) ----
function basis(M::ACDPDMatrixModel{O3S, Z2S, SC}, at::AbstractSystem;
               join_sites=false, filter=(_,_)->true, T=Float64) where {O3S, Z2S, SC}
    N = length(at); Z = _species(at); K = length(M.inds, :offsite)
    Is = [Int[] for _=1:K]; Js = [Int[] for _=1:K]; Vs = [_block_type(M,T)[] for _=1:K]
    for (i, neigs, Rs) in _sites(at, env_cutoff(M.offsite))
        (filter(i, at) && length(neigs) > 0) || continue
        Zs = Z[neigs]
        for (j_loc, j) in enumerate(neigs)
            filter(j, at) || continue
            (Zi, Zj) = _mreduce(Z[i], Z[j], SC); haskey(M.offsite, (Zi, Zj)) || continue
            Bij = evaluate_basis(M.offsite[(Zi, Zj)], j_loc, Rs, Zs)
            for (k, b) in zip(get_range(M, (Zi, Zj)), Bij); push!(Is[k], i); push!(Js[k], j); push!(Vs[k], b); end
        end
    end
    B = [ _add_mc_diagonal(sparse(Is[k], Js[k], Vs[k], N, N)) for k = 1:K ]
    return (join_sites ? B : (offsite = B,))
end

# momentum-conserving random force F = Σ·R (R iid per atom). Σ has zero column sums, so
# ∑_i F_i = ∑_j (∑_i Σ_ij) R_j = 0. Matrix-equivariant (SMatrix) and vector-equivariant
# (SVector, scalar noise) block types.
randf(::ACDPDMatrixModel, Σ::SparseMatrixCSC{SMatrix{3,3,T,9}, TI}) where {T<:Real, TI<:Int} =
    Σ * randn(SVector{3,T}, size(Σ, 2))
randf(::ACDPDMatrixModel, Σ::SparseMatrixCSC{SVector{3,T}, TI}) where {T<:Real, TI<:Int} =
    Σ * randn(T, size(Σ, 2))

# ---- serialization ----
function write_dict(M::ACDPDMatrixModel{O3S, Z2S, SC}) where {O3S, Z2S, SC}
    return Dict("__id__" => "ACEfriction_ACDPDMatrixModel",
                "offsite" => write_dict(M.offsite), "sc" => string(nameof(SC)), "id" => string(M.id))
end
function read_dict(::Val{:ACEfriction_ACDPDMatrixModel}, D::Dict)
    offsite = read_dict(D["offsite"]); sc = getfield(@__MODULE__, Symbol(D["sc"]))()
    return ACDPDMatrixModel(offsite, Symbol(D["id"]), sc)
end
