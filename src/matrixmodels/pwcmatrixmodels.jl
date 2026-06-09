struct PWCMatrixModel{O3S, CUTOFF, Z2S, SC} <: MatrixModel{O3S}
    offsite::OffSiteModels{O3S, Z2S, CUTOFF} where {Z2S, CUTOFF}
    n_rep::Int
    inds::SiteInds
    id::Symbol
    function PWCMatrixModel(offsite::OffSiteModels{O3S, Z2S, CUTOFF}, id::Symbol, sc::SC) where {O3S, Z2S, CUTOFF, SC}
        @assert length(unique([_n_rep(mo) for mo in values(offsite)])) == 1
        return new{O3S, CUTOFF, Z2S, SC}(offsite, _n_rep(offsite), SiteInds(_get_basisinds(offsite)), id)
    end
end

_get_SC(::PWCMatrixModel{O3S, TM, Z2S, SC}) where {O3S, Z2S, TM, SC} = SC
_offsite_cutoff(offsite::OffSiteModels) = first(values(offsite)).cutoff

# ---- Σ assembly (ellipsoid: bond iterator) ----
function matrix(M::PWCMatrixModel{O3S, <:EllipsoidCutoff, Z2S, SC}, at::AbstractSystem;
                filter=(_,_)->true, T=Float64) where {O3S, Z2S, SC}
    N = length(at); Z = _species(at)
    Is = [Int[] for _=1:M.n_rep]; Js = [Int[] for _=1:M.n_rep]
    Vs = [_block_type(M,T)[] for _=1:M.n_rep]
    for (i, j, rrij, _Js, Rs, Zs) in et_bonds(at, _offsite_cutoff(M.offsite))
        (filter(i, at) && filter(j, at)) || continue
        (Zi, Zj) = _mreduce(Z[i], Z[j], SC); haskey(M.offsite, (Zi, Zj)) || continue
        Σij = evaluate(M.offsite[(Zi, Zj)], rrij, Rs, Zs)
        for r = 1:M.n_rep; push!(Is[r], i); push!(Js[r], j); push!(Vs[r], Σij[r]); end
    end
    return [ sparse(Is[r], Js[r], Vs[r], N, N) for r = 1:M.n_rep ]
end

# ---- Σ assembly (spherical: site iterator, bond = marked neighbour j) ----
function matrix(M::PWCMatrixModel{O3S, <:SphericalCutoff, Z2S, SC}, at::AbstractSystem;
                filter=(_,_)->true, T=Float64) where {O3S, Z2S, SC}
    N = length(at); Z = _species(at)
    Is = [Int[] for _=1:M.n_rep]; Js = [Int[] for _=1:M.n_rep]
    Vs = [_block_type(M,T)[] for _=1:M.n_rep]
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
    return [ sparse(Is[r], Js[r], Vs[r], N, N) for r = 1:M.n_rep ]
end

# ---- un-contracted basis (ellipsoid) ----
function basis(M::PWCMatrixModel{O3S, <:EllipsoidCutoff, Z2S, SC}, at::AbstractSystem;
               join_sites=false, filter=(_,_)->true, T=Float64) where {O3S, Z2S, SC}
    N = length(at); Z = _species(at); K = length(M.inds, :offsite)
    Is = [Int[] for _=1:K]; Js = [Int[] for _=1:K]; Vs = [_block_type(M,T)[] for _=1:K]
    for (i, j, rrij, _Js, Rs, Zs) in et_bonds(at, _offsite_cutoff(M.offsite))
        (filter(i, at) && filter(j, at)) || continue
        (Zi, Zj) = _mreduce(Z[i], Z[j], SC); haskey(M.offsite, (Zi, Zj)) || continue
        Bij = evaluate_basis(M.offsite[(Zi, Zj)], rrij, Rs, Zs)
        for (k, b) in zip(get_range(M, (Zi, Zj)), Bij); push!(Is[k], i); push!(Js[k], j); push!(Vs[k], b); end
    end
    B = [ sparse(Is[k], Js[k], Vs[k], N, N) for k = 1:K ]
    return (join_sites ? B : (offsite = B,))
end

# ---- un-contracted basis (spherical) ----
function basis(M::PWCMatrixModel{O3S, <:SphericalCutoff, Z2S, SC}, at::AbstractSystem;
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
    B = [ sparse(Is[k], Js[k], Vs[k], N, N) for k = 1:K ]
    return (join_sites ? B : (offsite = B,))
end

function randf(::PWCMatrixModel, Σ::SparseMatrixCSC{SMatrix{3,3,T,9}, TI}) where {T<:Real, TI<:Int}
    I, J, _ = findnz(Σ); Rnz = randn(SVector{3,T}, length(J))
    R = (sparse(I, J, Rnz) .+ sparse(J, I, Rnz)) ./ sqrt(2)
    return vec(sum(Σ .* R, dims=1))
end

# ---- serialization ----
function write_dict(m::OffSiteModel{O3S, Z2S, CUTOFF, NR}) where {O3S, Z2S, CUTOFF, NR}
    return Dict("__id__" => "ACEfriction_OffSiteModel",
                "basis" => write_dict(m.basis),
                "c" => collect(reinterpret(Vector{Float64}, m.c)),
                "n_rep" => NR,
                "z2sym" => string(nameof(Z2S)),
                "cutoff" => write_dict(m.cutoff))
end
function read_dict(::Val{:ACEfriction_OffSiteModel}, D::Dict)
    basis = read_dict(D["basis"]); NR = Int(D["n_rep"])
    c = reinterpret(Vector{SVector{NR,Float64}}, Vector{Float64}(D["c"]))
    z2 = getfield(@__MODULE__, Symbol(D["z2sym"]))()
    cutoff = read_dict(D["cutoff"])
    return OffSiteModel(BondBasis(basis, z2), cutoff, c)
end
function write_dict(offsite::OffSiteModels)
    return Dict("__id__" => "ACEfriction_offsitemodels",
                "vals" => Dict(i => write_dict(v) for (i, v) in enumerate(values(offsite))),
                "z1" => Dict(i => string(_chemical_symbol(zz[1])) for (i, zz) in enumerate(keys(offsite))),
                "z2" => Dict(i => string(_chemical_symbol(zz[2])) for (i, zz) in enumerate(keys(offsite))))
end
read_dict(::Val{:ACEfriction_offsitemodels}, D::Dict) =
        Dict((_atomic_number(Symbol(z1)), _atomic_number(Symbol(z2))) => read_dict(v)
             for (z1, z2, v) in zip(values(D["z1"]), values(D["z2"]), values(D["vals"])))

function write_dict(M::PWCMatrixModel{O3S, CUTOFF, Z2S, SC}) where {O3S, CUTOFF, Z2S, SC}
    return Dict("__id__" => "ACEfriction_PWCMatrixModel",
                "offsite" => write_dict(M.offsite), "sc" => string(nameof(SC)), "id" => string(M.id))
end
function read_dict(::Val{:ACEfriction_PWCMatrixModel}, D::Dict)
    offsite = read_dict(D["offsite"]); sc = getfield(@__MODULE__, Symbol(D["sc"]))()
    return PWCMatrixModel(offsite, Symbol(D["id"]), sc)
end
