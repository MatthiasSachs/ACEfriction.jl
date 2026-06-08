# Production native bond (offsite) basis with Z2 parity grading.
#
# Reuses the onsite machinery (species-channel radial + ET equivariant tensor +
# Cartesian output transform). The bond direction r̂ij is a distinguished radial
# channel (a sentinel "species" `BOND_Z = 0`), so the pooled A over the bond
# channel receives only the bond particle => the bond appears exactly once per
# basis function. Z2 parity = parity of the single bond factor's l:
#   :even -> keep even bond-l (Z2-invariant)   ~ ACEfrictionCore bondsymmetry "Invariant"
#   :odd  -> keep odd  bond-l (Z2-covariant)   ~ "Covariant"
#   :none -> keep both                          ~ no Z2 symmetry
#
# Validated in test/etbackend/test_bond_prototype.jl (mechanism) and
# test/etbackend/test_bondbasis.jl (this production code).

import Polynomials4ML as P4ML
import EquivariantTensors as ET
using StaticArrays, LinearAlgebra

"sentinel atomic number marking the bond channel"
const BOND_Z = 0

"""
    generate_bond_mb_spec(rbasis, maxl; maxorder, maxdeg, wn, wl, z2sym)

mb_spec for the bond basis: each function has exactly one bond-channel factor
(species `BOND_Z`) plus `0:(maxorder-1)` env-channel factors, with the Z2 parity
filter applied to the bond factor's `l`.
"""
function generate_bond_mb_spec(rbasis::SpeciesRadialBasis, maxl::Integer;
                               maxorder::Integer, maxdeg::Real,
                               wn::Real = 1.0, wl::Real = 1.0,
                               z2sym::Symbol = :none)
   degn(ñ) = channel_info(rbasis, ñ).n - 1
   isbond(ñ) = channel_info(rbasis, ñ).z == BOND_Z
   fdeg(b) = wn * degn(b.n) + wl * b.l
   allfac = _NL[ (n = ñ, l = l)
                 for ñ in 1:nchannels(rbasis) for l in 0:maxl
                 if wn * degn(ñ) + wl * l <= maxdeg ]
   bondfac = filter(b -> isbond(b.n), allfac)
   envfac  = filter(b -> !isbond(b.n), allfac)

   if z2sym == :even
      bondfac = filter(b -> iseven(b.l), bondfac)
   elseif z2sym == :odd
      bondfac = filter(b -> isodd(b.l), bondfac)
   elseif z2sym != :none
      error("z2sym must be :even, :odd or :none (got $z2sym)")
   end

   spec = Vector{_NL}[]
   for bf in bondfac, k in 0:(maxorder - 1)
      for envc in _wr_combinations(envfac, k)
         fdeg(bf) + sum(fdeg(e) for e in envc; init = 0) <= maxdeg || continue
         push!(spec, sort(vcat([bf], envc)))
      end
   end
   return unique(spec)
end

"""
    bond_basis(property, species; z2sym, rcut, maxorder, maxdeg, maxl, wn, wl, radial_kwargs...)

Construct a native ET bond (offsite) basis. `species` are the environment element
symbols/atomic numbers; the bond channel is added automatically. `z2sym` ∈
`(:none, :even, :odd)`. Returns an `ETFrictionSiteBasis` (the bond constraint and
Z2 grading live in its tensor's mb_spec), so the standard `evaluate` works once
the bond particle is prepended (use [`evaluate_bond`](@ref)).
"""
function bond_basis(property::ETProperty, species;
                    z2sym::Symbol = :none,
                    rcut::Real, maxorder::Integer, maxdeg::Real,
                    maxl::Integer = Int(floor(maxdeg)),
                    wn::Real = 1.0, wl::Real = 1.0,
                    radial_kwargs...)
   zlist = [BOND_Z; [_atomic_number(s) for s in species]...]   # bond channel first
   rbasis = RnYlm_radial(zlist; rcut = rcut, maxn = Int(floor(maxdeg)),
                         radial_kwargs...)
   ybasis = P4ML.real_sphericalharmonics(maxl)
   Ylm_spec = P4ML.natural_indices(ybasis)
   mb_spec = generate_bond_mb_spec(rbasis, maxl; maxorder = maxorder,
                  maxdeg = maxdeg, wn = wn, wl = wl, z2sym = z2sym)
   isempty(mb_spec) && error("empty bond mb_spec (check maxdeg/maxorder/z2sym)")
   tensor = ET.sparse_equivariant_tensors(;
            LL = output_LL(property), mb_spec = mb_spec,
            Rnl_spec = radial_spec(rbasis), Ylm_spec = Ylm_spec, basis = real)
   out = ETOutput(property)
   recipe = Dict{String, Any}(
      "kind" => "bond",
      "property" => _property_str(property),
      "species" => [ _atomic_number(s) for s in species ],
      "z2sym" => String(z2sym),
      "rcut" => Float64(rcut), "maxorder" => Int(maxorder),
      "maxdeg" => Float64(maxdeg), "maxl" => Int(maxl),
      "wn" => Float64(wn), "wl" => Float64(wl),
      "radial" => Dict{String, Any}(string(k) => v for (k, v) in radial_kwargs))
   meta = Dict{String, Any}("recipe" => recipe)
   return ETFrictionSiteBasis(property, rbasis, ybasis, tensor, out, meta)
end

"""
    evaluate_bond(basis, rrij, Rs_env, Zs_env) -> Vector{block}

Evaluate a bond basis: `rrij` is the (normalized) bond vector, `Rs_env`/`Zs_env`
the (transformed) environment vectors/species. Prepends the bond particle (species
`BOND_Z`) and calls the standard site-basis `evaluate`.
"""
function evaluate_bond(basis::ETFrictionSiteBasis,
                       rrij::SVector{3}, Rs_env::AbstractVector{<:SVector{3}},
                       Zs_env::AbstractVector)
   Rs = vcat([rrij], Rs_env)
   Zs = vcat([BOND_Z], Zs_env)
   return evaluate(basis, Rs, Zs)
end
