# Onsite ET friction site basis: a thin ETACE-style container bundling the
# species-channel radial embedding, the spherical-harmonic embedding, the ET
# equivariant tensor, and the Cartesian output transform. Forward-only (the
# friction fitting path precomputes B and differentiates only over the linear
# coefficients), so no Lux ps/st or ChainRules plumbing is implemented here.
#
# Equivariance is carried by the tensor's LL (derived from the property), not a
# type parameter.

import Polynomials4ML as P4ML
import EquivariantTensors as ET
using StaticArrays, LinearAlgebra

# ----------------------------------------------------------------------
# mb_spec generation (reproduces the role of SparseBasis/CategorySparseBasis).
#
# A many-body basis function is an ordered multiset of (n = ñ, l) factors, where
# ñ is a radial *channel* (species, poly-degree). Selection rules:
#   - correlation order (number of factors)  <= maxorder
#   - weighted total degree  Σ (wn·degₙ(ñ) + wl·l)  <= maxdeg
# This is the standard p=1 ACE total-degree sparsification; `weight` mirrors
# ACEfrictionCore's `Dict(:n=>…, :l=>…)`.

const _NL = NamedTuple{(:n, :l), Tuple{Int, Int}}

# with-replacement combinations of `xs` of length k (non-decreasing index order)
function _wr_combinations(xs, k)
   k == 0 && return [eltype(xs)[]]
   n = length(xs)
   out = Vector{eltype(xs)}[]
   idx = ones(Int, k)
   while true
      push!(out, [xs[i] for i in idx])
      p = k
      while p >= 1 && idx[p] == n; p -= 1; end
      p == 0 && break
      idx[p] += 1
      for q = p+1:k; idx[q] = idx[p]; end
   end
   return out
end

"""
    generate_mb_spec(rbasis, maxl; maxorder, maxdeg, wn, wl)

Generate the many-body `(n=ñ, l)` specification over the radial channels of
`rbasis` (a `SpeciesRadialBasis`) and angular degrees `0:maxl`.
"""
function generate_mb_spec(rbasis::SpeciesRadialBasis, maxl::Integer;
                          maxorder::Integer, maxdeg::Real,
                          wn::Real = 1.0, wl::Real = 1.0)
   degn(ñ) = channel_info(rbasis, ñ).n - 1        # 0-based poly degree
   fdeg(b) = wn * degn(b.n) + wl * b.l
   factors = _NL[ (n = ñ, l = l)
                  for ñ in 1:nchannels(rbasis) for l in 0:maxl
                  if wn * degn(ñ) + wl * l <= maxdeg ]
   spec = Vector{_NL}[]
   for k in 1:maxorder
      for bb in _wr_combinations(factors, k)
         sum(fdeg(b) for b in bb) <= maxdeg && push!(spec, sort(bb))
      end
   end
   return unique(spec)
end

# ----------------------------------------------------------------------
# Site basis container

"""
    ETFrictionSiteBasis(property, rbasis, ybasis, tensor, out, meta)

Forward-only ET site basis. Build via [`onsite_basis`](@ref). `property` is one
of `ETInvariant`/`ETVector`/`ETMatrix`/`ETSymMatrix`; `rbasis` the species-channel
radial; `ybasis` the P4ML spherical harmonics; `tensor` the ET `SparseACEbasis`;
`out` the cached output transform.
"""
struct ETFrictionSiteBasis{P, TR, TY, TT, TO}
   property::P
   rbasis::TR
   ybasis::TY
   tensor::TT
   out::TO
   meta::Dict{String, Any}
end

Base.length(b::ETFrictionSiteBasis) = sum(b.tensor.lens)
_o3property(b::ETFrictionSiteBasis) = b.property
block_type(b::ETFrictionSiteBasis, T = Float64) = block_type(b.property, T)

"""
    onsite_basis(property, species; rcut, maxorder, maxdeg, maxl, wn, wl, radial_kwargs...)

Construct an onsite ET friction site basis for the given `property` and `species`
(list of element symbols / atomic numbers). Mirrors the inputs of
ACEfrictionCore's `onsite_linbasis`.
"""
function onsite_basis(property::ETProperty, species;
                      rcut::Real, maxorder::Integer, maxdeg::Real,
                      maxl::Integer = Int(floor(maxdeg)),
                      wn::Real = 1.0, wl::Real = 1.0,
                      radial_kwargs...)
   rbasis = RnYlm_radial(species; rcut = rcut, maxn = Int(floor(maxdeg)),
                         radial_kwargs...)
   ybasis = P4ML.real_sphericalharmonics(maxl)
   Ylm_spec = P4ML.natural_indices(ybasis)
   mb_spec = generate_mb_spec(rbasis, maxl; maxorder = maxorder,
                              maxdeg = maxdeg, wn = wn, wl = wl)
   tensor = ET.sparse_equivariant_tensors(;
            LL = output_LL(property), mb_spec = mb_spec,
            Rnl_spec = radial_spec(rbasis), Ylm_spec = Ylm_spec, basis = real)
   out = ETOutput(property)
   recipe = Dict{String, Any}(
      "kind" => "onsite",
      "property" => _property_str(property),
      "species" => [ _atomic_number(s) for s in species ],
      "rcut" => Float64(rcut), "maxorder" => Int(maxorder),
      "maxdeg" => Float64(maxdeg), "maxl" => Int(maxl),
      "wn" => Float64(wn), "wl" => Float64(wl),
      "radial" => Dict{String, Any}(string(k) => v for (k, v) in radial_kwargs))
   meta = Dict{String, Any}("recipe" => recipe)
   return ETFrictionSiteBasis(property, rbasis, ybasis, tensor, out, meta)
end

# ----------------------------------------------------------------------
# Forward evaluation

"""
    evaluate(basis, Rs, Zs) -> Vector{block}

Evaluate every basis function on an environment given by neighbour vectors `Rs`
(raw, un-scaled; the radial transform handles the cutoff) and neighbour species
`Zs`. Returns a `Vector` of Cartesian blocks (one per basis function), the array
that fills `B` for fitting and is contracted in `Gamma`.
"""
function evaluate(basis::ETFrictionSiteBasis,
                  Rs::AbstractVector{<:SVector{3}}, Zs::AbstractVector)
   rs = norm.(Rs)
   Rnl = evaluate_batched(basis.rbasis, rs, Zs)
   Ylm = P4ML.evaluate(basis.ybasis, Rs)
   BB = ET.evaluate(basis.tensor, Rnl, Ylm)
   blocks = Vector{block_type(basis)}(undef, nblocks(BB))
   return assemble_blocks!(blocks, basis.out, BB)
end

# empty environment -> all-zero blocks (matches "no neighbours" -> zero basis)
function evaluate(basis::ETFrictionSiteBasis,
                  Rs::AbstractVector{<:SVector{3}}, Zs::AbstractVector,
                  ::Val{:empty})
   z = zero(block_type(basis))
   return fill(z, length(basis))
end

# ----------------------------------------------------------------------
# Regularization scaling: per-basis-function weight based on its (n,l) degree.

"""
    scaling(basis, p) -> Vector

Per-basis-function scaling `(1 + Σ(n+l))^p` over each basis function's (nn,ll)
spec, used as a smoothness prior (mirrors `ACEfrictionCore.scaling`).
"""
function scaling(basis::ETFrictionSiteBasis, p::Real)
   sc = Float64[]
   for il in eachindex(basis.tensor.LL)
      nnll = ET.get_nnll_spec(basis.tensor, il)
      for bb in nnll
         deg = isempty(bb) ? 0 : sum(b.n + b.l for b in bb)
         push!(sc, (1 + deg)^p)
      end
   end
   return sc
end
