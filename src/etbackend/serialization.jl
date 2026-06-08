# Serialization for ET backend types (write_dict / read_dict).
#
# Bases are serialized by their *construction recipe* (stored in `basis.meta`) and
# rebuilt with `onsite_basis` / `bond_basis` on read — we do not dump the ET tensor
# or P4ML objects. Models add their coefficient vector. The dicts use the same
# "__id__" convention as ACEfrictionCore so they slot into the friction-model
# serialization at cut-over.

using StaticArrays

"""dispatch a dict produced by `write_dict` back to a concrete object."""
read_dict(D::AbstractDict) = read_dict(Val(Symbol(D["__id__"])), D)

# ---- site basis (onsite or bond) : rebuild from recipe ----

"""rebuild an `ETFrictionSiteBasis` from a serialized construction recipe."""
function rebuild_basis(recipe::AbstractDict)
   prop = _property_from_str(recipe["property"])
   species = Int.(recipe["species"])
   radial = Dict(Symbol(k) => v for (k, v) in recipe["radial"])
   kind = recipe["kind"]
   if kind == "onsite"
      return onsite_basis(prop, species;
               rcut = recipe["rcut"], maxorder = recipe["maxorder"],
               maxdeg = recipe["maxdeg"], maxl = recipe["maxl"],
               wn = recipe["wn"], wl = recipe["wl"], radial...)
   elseif kind == "bond"
      return bond_basis(prop, species;
               z2sym = Symbol(recipe["z2sym"]),
               rcut = recipe["rcut"], maxorder = recipe["maxorder"],
               maxdeg = recipe["maxdeg"], maxl = recipe["maxl"],
               wn = recipe["wn"], wl = recipe["wl"], radial...)
   else
      error("unknown basis recipe kind = $kind")
   end
end

write_dict(b::ETFrictionSiteBasis) =
      Dict{String, Any}("__id__" => "ETBackend_SiteBasis",
                        "recipe" => b.meta["recipe"])

read_dict(::Val{:ETBackend_SiteBasis}, D::AbstractDict) = rebuild_basis(D["recipe"])

# ---- flattened onsite model : recipe + coefficients ----

function write_dict(m::ETOnsiteModel{NR}) where {NR}
   return Dict{String, Any}(
      "__id__" => "ETBackend_OnsiteModel",
      "basis" => write_dict(m.basis),
      "n_rep" => NR,
      "c" => collect(reinterpret(Float64, m.c)))
end

function read_dict(::Val{:ETBackend_OnsiteModel}, D::AbstractDict)
   basis = read_dict(D["basis"])
   NR = Int(D["n_rep"])
   cflat = Vector{Float64}(D["c"])
   c = collect(reinterpret(SVector{NR, Float64}, cflat))
   return ETOnsiteModel(basis, c)
end
