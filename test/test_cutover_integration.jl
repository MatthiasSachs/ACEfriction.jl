# Integration test for the EquivariantTensors backend cutover: the public API
# (model constructors -> FrictionModel -> Gamma/Sigma, params round-trip, and the
# full Flux fitting path) on a real system, with no ACEfrictionCore.
using ACEfriction
using ACEfriction: EuclideanMatrix, EuclideanVector, SymmetricEuclideanMatrix, EllipsoidCutoff, SnowManCutoff
using ACEbase.FIO: write_dict, read_dict
using Test, LinearAlgebra, StaticArrays, SparseArrays
import AtomsBuilder: bulk, rattle!
using AtomsBase: ChemicalSpecies
using Flux: gradient, setup, Adam, update!
import Random

_dense(G, N) = (A = zeros(3N, 3N); for i=1:N, j=1:N; A[3i-2:3i, 3j-2:3j] .= G[i,j]; end; A)

@testset "cutover integration (ET backend, public API)" begin
   Random.seed!(1)
   at = rattle!(bulk(:Cu) * (2,2,2), 0.2); N = length(at)

   @testset "Gamma/Sigma PSD + params round-trip" begin
      m_on = OnsiteOnlyMatrixModel(EuclideanMatrix(Float64), [:Cu], [:Cu]; maxorder=2, maxdeg=4, rcut=5.0, n_rep=2)
      m_pw = PWCMatrixModel(EuclideanMatrix(Float64), [:Cu], [:Cu]; maxorder=2, maxdeg=4, rcut=5.0, n_rep=2)
      fm = FrictionModel((cov=m_on, equ=m_pw))
      Γ = Gamma(fm, at); Gd = _dense(Γ, N)
      @test norm(Gd - Gd') < 1e-8
      @test minimum(eigvals(Symmetric(Gd))) > -1e-8
      c = params(fm; format=:matrix, joinsites=true)
      @test keys(c) == (:cov, :equ)
      set_params!(fm, c)
      @test params(fm; format=:matrix, joinsites=true)[:cov] ≈ c[:cov]
   end

   @testset "ellipsoid PWC builds + PSD" begin
      m = PWCMatrixModel(EuclideanMatrix(Float64), [:Cu], [:Cu], EllipsoidCutoff(3.0,4.0,5.0);
                         maxorder=2, maxdeg=4, n_rep=2, z2sym=ACEfriction.MatrixModels.NoZ2Sym())
      fm = FrictionModel((equ=m,))
      Γ = Gamma(fm, at); Gd = _dense(Γ, N)
      @test minimum(eigvals(Symmetric(Gd))) > -1e-8
   end

   @testset "DPD model: momentum conservation (dummy :X species)" begin
      # Regression: the bond-channel sentinel must not collide with the environment
      # species. The dummy AtomsBase species `:X` has atomic number 0, which used to
      # equal the sentinel, letting env atoms pool into the bond channel and breaking
      # the bond's Z2-odd parity => Σ no longer antisymmetric => Γ not momentum
      # conserving (row sums of Γ must vanish for a pairwise DPD model).
      import AtomsBase: Atom, FlexibleSystem, cell_vectors, periodicity, position
      import Unitful: @u_str
      atX = FlexibleSystem([Atom(0, position(at, i)) for i in 1:N];
                           cell_vectors = cell_vectors(at), periodicity = periodicity(at))
      m = mbdpd_matrixmodel(EuclideanVector(), [:X], [:X];
             maxorder=2, maxdeg=6, rcutbond=5.0, rcutenv=5.0, zcutenv=5.0, n_rep=2)
      fm = FrictionModel((cov=m,))
      # randomise coefficients so the property isn't trivially satisfied at zero
      Random.seed!(3)
      c = params(fm; format=:matrix, joinsites=true)
      set_params!(fm, map(x -> randn(size(x)), c))
      Σnt = Sigma(fm, atX)
      Σ = Σnt.cov[1]
      I, J, _ = findnz(Σ)
      @test maximum(norm(Σ[i,j] + Σ[j,i]) for (i,j) in zip(I,J)) < 1e-12   # antisymmetry
      Γ = Gamma(fm, atX)
      @test maximum(norm(sum(Γ[i,j] for j in 1:N)) for i in 1:N) < 1e-10   # zero row sums

      # randf must produce momentum-conserving noise: the per-atom force vectors
      # sum to zero (symmetric scalar noise contracted with antisymmetric Σ cancels).
      @test all(begin
                   F = randf(fm, Σnt)        # Vector{SVector{3}}, one force per atom
                   norm(sum(F)) < 1e-10
                end for _ in 1:20)
   end

   @testset "SnowMan PWC ($sym): combine, basis/matrix consistency, IO" for sym in (:symmetric, :antisymmetric)
      # Σ_ij = c·basis(sphere_i, bond i→j) ± c·basis(sphere_j, bond j→i): combining both
      # bond ends makes Σ symmetric (+) or antisymmetric (−), selected by the cutoff's
      # type-parameter symmetry via _snowman_combine. Γ stays PSD; the un-contracted
      # basis contracted with c reproduces Σ; the symmetry round-trips through IO.
      m = PWCMatrixModel(EuclideanMatrix(Float64), [:Cu], [:Cu], SnowManCutoff(5.0, sym);
                         maxorder=2, maxdeg=4, n_rep=2)
      fm = FrictionModel((equ=m,))
      Σ = Sigma(fm, at).equ[1]
      I, J, _ = findnz(Σ)
      resid = sym === :symmetric ? maximum(norm(Σ[i,j] - Σ[j,i]) for (i,j) in zip(I,J)) :
                                   maximum(norm(Σ[i,j] + Σ[j,i]) for (i,j) in zip(I,J))
      @test resid < 1e-10                                       # (anti)symmetry of Σ
      Gd = _dense(Gamma(fm, at), N)
      @test norm(Gd - Gd') < 1e-8
      @test minimum(eigvals(Symmetric(Gd))) > -1e-8
      # basis · c == matrix (fitting-path consistency)
      Boff = ACEfriction.MatrixModels.basis(m, at).offsite
      cc = m.offsite[(29,29)].c
      @test norm(sum(cc[k][1] * Boff[k] for k in eachindex(Boff)) - Σ) < 1e-10
      # cached (default) vs naive double-eval assembly agree exactly
      Σn = ACEfriction.MatrixModels.matrix(m, at; cache=false)
      Bn = ACEfriction.MatrixModels.basis(m, at; cache=false).offsite
      @test maximum(norm(Sigma(fm, at).equ[r] - Σn[r]) for r in eachindex(Σn)) < 1e-12
      @test maximum(norm(Boff[k] - Bn[k]) for k in eachindex(Boff)) < 1e-12
      # serialization round-trip (incl. the symmetry type parameter)
      fm2 = read_dict(write_dict(fm))
      @test fm2.matrixmodels.equ.offsite[(29,29)].cutoff isa SnowManCutoff{Float64, sym}
      @test norm(_dense(Gamma(fm2, at), N) - Gd) < 1e-10
   end

   @testset "Flux fitting path: loss decreases" begin
      m_on = OnsiteOnlyMatrixModel(EuclideanMatrix(Float64), [:Cu], [:Cu]; maxorder=2, maxdeg=4, rcut=5.0, n_rep=2)
      m_pw = PWCMatrixModel(EuclideanMatrix(Float64), [:Cu], [:Cu]; maxorder=2, maxdeg=4, rcut=5.0, n_rep=2)
      fm = FrictionModel((cov=m_on, equ=m_pw))
      c_true = params(fm; format=:matrix, joinsites=true); set_params!(fm, c_true)
      ats = [ rattle!(bulk(:Cu) * (2,2,2), 0.3) for _ in 1:4 ]
      fdata = [ FrictionData(a, Gamma(fm, a), collect(1:length(a))) for a in ats ]
      c0 = map(x -> 0.01 .* randn(size(x)), c_true); set_params!(fm, c0)
      ffm = FluxFrictionModel(c0)
      data = flux_assemble(fdata, fm, ffm)
      loss0 = weighted_l2_loss(ffm, data)
      opt = setup(Adam(0.01), ffm)
      for _ in 1:50
         ∂L = gradient(weighted_l2_loss, ffm, data)[1]; update!(opt, ffm, ∂L)
      end
      loss1 = weighted_l2_loss(ffm, data)
      @test isfinite(loss1) && loss1 < loss0
   end
end
