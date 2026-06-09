using ACEfriction.MatrixModels
import ACEfriction.ETBackend: _atomic_number
import ACEfriction.MatrixModels: RWCMatrixModel, PWCMatrixModel, OnsiteOnlyMatrixModel
import ACEfriction.MatrixModels: OnSiteModel, OffSiteModel, BondBasis, onsite_linbasis,
       offsite_linbasis, SphericalCutoff, EllipsoidCutoff, _o3symmetry, _default_id,
       _mreduce, NoZ2Sym, Odd, Even, SpeciesCoupled, SpeciesUnCoupled,
       AtomCentered, NeighborCentered, EvaluationCenter, _o3sym
export RWCMatrixModel, PWCMatrixModel, OnsiteOnlyMatrixModel, mbdpd_matrixmodel

# Outer convenience constructors. The basis backend is EquivariantTensors; the
# `polytransform`/`trans` argument of the old backend is gone (the radial transform
# is a generalized Agnesi parameterised by `r0_ratio`/`rin_ratio`).

_o3id(property) = _default_id(_o3sym(property))

"""
    OnsiteOnlyMatrixModel(property, species_friction, species_env;
        maxorder=2, maxdeg=5, rcut=5.0, n_rep=1, ...)

Block-diagonal friction model (onsite only).
"""
function OnsiteOnlyMatrixModel(property, species_friction, species_env;
        id=nothing, n_rep=1, maxorder=2, maxdeg=5, rcut=5.0,
        r0_ratio=0.4, rin_ratio=0.04, pcut=2, pin=2, p_sel=2,
        weight=Dict(:n => 1.0, :l => 1.0),
        species_minorder_dict=Dict{Any,Float64}(),
        species_maxorder_dict=Dict{Any,Float64}(),
        species_weight_cat=Dict(c => 1.0 for c in species_env),
        species_substrat=[])
    onsitebasis = onsite_linbasis(property, species_env;
        rcut=rcut, maxorder=maxorder, maxdeg=maxdeg, r0_ratio=r0_ratio,
        rin_ratio=rin_ratio, pcut=pcut, pin=pin, p_sel=p_sel, weight=weight,
        species_minorder_dict=species_minorder_dict,
        species_maxorder_dict=species_maxorder_dict,
        species_weight_cat=species_weight_cat, species_substrat=species_substrat)
    onsitemodels = Dict(_atomic_number(z) => OnSiteModel(onsitebasis, SphericalCutoff(rcut), n_rep)
                        for z in species_friction)
    id = (id === nothing ? _o3id(property) : id)
    return OnsiteOnlyMatrixModel(onsitemodels, id)
end

"""
    PWCMatrixModel(property, species_friction, species_env;
        maxorder=2, maxdeg=5, rcut=5.0, n_rep=1, ...)

Pairwise-coupled (offsite) friction model with a spherical pair environment.
"""
function PWCMatrixModel(property, species_friction, species_env;
        id=nothing, n_rep=1, maxorder=2, maxdeg=5, rcut=5.0,
        z2sym=NoZ2Sym(), speciescoupling=SpeciesUnCoupled(),
        r0_ratio=0.4, rin_ratio=0.04, pcut=2, pin=2, p_sel=2,
        weight=Dict(:n => 1.0, :l => 1.0), bond_weight=1.0,
        species_minorder_dict=Dict{Any,Float64}(),
        species_maxorder_dict=Dict{Any,Float64}(),
        species_weight_cat=Dict(c => 1.0 for c in species_env),
        species_substrat=[])
    bb = offsite_linbasis(property, species_env;
        z2symmetry=z2sym, rcut=1.0, maxorder=maxorder, maxdeg=maxdeg,
        r0_ratio=r0_ratio, rin_ratio=rin_ratio, pcut=pcut, pin=pin, p_sel=p_sel,
        weight=weight, bond_weight=bond_weight,
        species_minorder_dict=species_minorder_dict,
        species_maxorder_dict=species_maxorder_dict,
        species_weight_cat=species_weight_cat, species_substrat=species_substrat)
    cutoff = SphericalCutoff(rcut)
    offsitemodels = _offsite_dict(bb, cutoff, species_friction, n_rep, speciescoupling)
    id = (id === nothing ? _o3id(property) : id)
    return PWCMatrixModel(offsitemodels, id, speciescoupling)
end

"""
    PWCMatrixModel(property, species_friction, species_env, cutoff::EllipsoidCutoff; ...)

Pairwise-coupled offsite model with an ellipsoidal (bond-centred) environment.
"""
function PWCMatrixModel(property, species_friction, species_env, cutoff::EllipsoidCutoff;
        id=nothing, n_rep=1, maxorder=2, maxdeg=5,
        z2sym=NoZ2Sym(), speciescoupling=SpeciesUnCoupled(),
        r0_ratio=0.4, rin_ratio=0.04, pcut=2, pin=2, p_sel=2,
        weight=Dict(:n => 1.0, :l => 1.0), bond_weight=1.0,
        species_minorder_dict=Dict{Any,Float64}(),
        species_maxorder_dict=Dict{Any,Float64}(),
        species_weight_cat=Dict(c => 1.0 for c in species_env),
        species_substrat=[])
    bb = offsite_linbasis(property, species_env;
        z2symmetry=z2sym, rcut=1.0, maxorder=maxorder, maxdeg=maxdeg,
        r0_ratio=r0_ratio, rin_ratio=rin_ratio, pcut=pcut, pin=pin, p_sel=p_sel,
        weight=weight, bond_weight=bond_weight,
        species_minorder_dict=species_minorder_dict,
        species_maxorder_dict=species_maxorder_dict,
        species_weight_cat=species_weight_cat, species_substrat=species_substrat)
    offsitemodels = _offsite_dict(bb, cutoff, species_friction, n_rep, speciescoupling)
    id = (id === nothing ? _o3id(property) : id)
    return PWCMatrixModel(offsitemodels, id, speciescoupling)
end

function _offsite_dict(bb, cutoff, species_friction, n_rep, sc::SpeciesUnCoupled)
    return Dict(_atomic_number.(zz) => OffSiteModel(bb, cutoff, n_rep)
                for zz in Base.Iterators.product(species_friction, species_friction))
end
function _offsite_dict(bb, cutoff, species_friction, n_rep, sc::SpeciesCoupled)
    return Dict(_atomic_number.(zz) => OffSiteModel(bb, cutoff, n_rep)
                for zz in Base.Iterators.product(species_friction, species_friction)
                if _mreduce(zz..., SpeciesCoupled) == zz)
end

"""
    RWCMatrixModel(property, species_friction, species_env; maxorder=2, maxdeg=5, rcut=5.0, n_rep=1, ...)

Row-wise coupled friction model (onsite + spherical offsite).
"""
function RWCMatrixModel(property, species_friction, species_env;
        id=nothing, n_rep=1, maxorder=2, maxdeg=5, rcut=5.0,
        evalcenter=AtomCentered(), speciescoupling=SpeciesUnCoupled(),
        r0_ratio=0.4, rin_ratio=0.04, pcut=2, pin=2, p_sel=2,
        weight=Dict(:n => 1.0, :l => 1.0), bond_weight=1.0,
        species_minorder_dict=Dict{Any,Float64}(),
        species_maxorder_dict=Dict{Any,Float64}(),
        species_weight_cat=Dict(c => 1.0 for c in species_env),
        species_substrat=[])
    onsitebasis = onsite_linbasis(property, species_env;
        rcut=rcut, maxorder=maxorder, maxdeg=maxdeg, r0_ratio=r0_ratio,
        rin_ratio=rin_ratio, pcut=pcut, pin=pin, p_sel=p_sel, weight=weight,
        species_minorder_dict=species_minorder_dict,
        species_maxorder_dict=species_maxorder_dict,
        species_weight_cat=species_weight_cat, species_substrat=species_substrat)
    bb = offsite_linbasis(property, species_env;
        z2symmetry=NoZ2Sym(), rcut=1.0, maxorder=maxorder, maxdeg=maxdeg,
        r0_ratio=r0_ratio, rin_ratio=rin_ratio, pcut=pcut, pin=pin, p_sel=p_sel,
        weight=weight, bond_weight=bond_weight,
        species_minorder_dict=species_minorder_dict,
        species_maxorder_dict=species_maxorder_dict,
        species_weight_cat=species_weight_cat, species_substrat=species_substrat)
    onsitemodels = Dict(_atomic_number(z) => OnSiteModel(onsitebasis, SphericalCutoff(rcut), n_rep)
                        for z in species_friction)
    offsitemodels = _offsite_dict(bb, SphericalCutoff(rcut), species_friction, n_rep, speciescoupling)
    id = (id === nothing ? _o3id(property) : id)
    return RWCMatrixModel(onsitemodels, offsitemodels, id, evalcenter, speciescoupling)
end

"""
    mbdpd_matrixmodel(property, species_friction, species_env; maxorder=2, maxdeg=5,
        rcutbond=5.0, rcutenv=3.0, zcutenv=6.0, n_rep=3, ...)

Momentum-preserving (DPD) friction model: a pairwise-coupled model on an ellipsoid
bond environment with Z2-odd symmetry and species coupling.
"""
function mbdpd_matrixmodel(property, species_friction, species_env;
        maxorder=2, maxdeg=5, rcutbond=5.0, rcutenv=3.0, zcutenv=6.0, n_rep=3,
        species_substrat=[], id=nothing, kwargs...)
    return PWCMatrixModel(property, species_friction, species_env,
        EllipsoidCutoff(rcutbond, rcutenv, zcutenv);
        n_rep=n_rep, maxorder=maxorder, maxdeg=maxdeg, z2sym=Odd(),
        speciescoupling=SpeciesCoupled(), species_substrat=species_substrat, id=id, kwargs...)
end
