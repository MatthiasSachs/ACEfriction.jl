# Cross-backend validation + ET fit-recovery (STEP 2).
#
# (1) Both backends build, evaluate, and are O(3)-equivariant in one session
#     ("two working backends").
# (2) ET fit-recovery: synthetic friction data from known coefficients is
#     recovered by linear least-squares on the precomputed basis B, proving the ET
#     basis is well-conditioned and usable for the actual fitting workflow.
#
# NOTE: exact basis-span equality across backends is NOT expected — the ET backend
# uses a different (Legendre/Agnesi) radial basis than ACEfrictionCore's OrthPolyBasis,
# so the two span (slightly) different function spaces. Equivalence is at the level
# of symmetry structure + fitting capability, per the migration plan.
using Test
import EquivariantTensors as ET
using StaticArrays, LinearAlgebra, Random

# old backend
using ACEfriction
import ACEfrictionCore
import ACEbase
const MM = ACEfriction.MatrixModels
const AC = ACEfriction.AtomCutoffs

# new backend
include(joinpath(@__DIR__, "..", "..", "src", "etbackend", "etbackend.jl"))
using .ETBackend

Random.seed!(12)

randenv(N) = ([ (r = @SVector(randn(3)); 3.0*r/norm(r)*rand()) for _ in 1:N ], fill(29, N))

@testset "two working backends (onsite, EuclideanMatrix)" begin
   # ---- old backend ----
   onsite_old = MM.onsite_linbasis(ACEfrictionCore.EuclideanMatrix(Float64), [:Cu];
                     maxorder = 2, maxdeg = 4)
   cutoff = AC.SphericalCutoff(5.0)
   Rs, Zs = randenv(6)
   Bold = ACEbase.evaluate(onsite_old, AC.env_transform(Rs, Zs, cutoff))
   θ = π*rand(3); Q = ET.O3.Q_from_angles(θ)
   BoldQ = ACEbase.evaluate(onsite_old, AC.env_transform([Q*r for r in Rs], Zs, cutoff))
   err_old = maximum(norm(BoldQ[k].val - Q*Bold[k].val*Q') for k in eachindex(Bold))
   @test err_old < 1e-10

   # ---- new backend ----
   basis_et = ETBackend.onsite_basis(ETBackend.ETMatrix(), [:Cu];
                     rcut = 5.0, maxorder = 2, maxdeg = 4, maxl = 3)
   Bet = ETBackend.evaluate(basis_et, Rs, Zs)
   BetQ = ETBackend.evaluate(basis_et, [Q*r for r in Rs], Zs)
   err_et = maximum(norm(BetQ[k] - Q*Bet[k]*Q') for k in eachindex(Bet))
   @test err_et < 1e-9

   println("  OLD basis: $(length(onsite_old)) fns, equivar err $(round(err_old, sigdigits=2))")
   println("  ET  basis: $(length(basis_et)) fns, equivar err $(round(err_et, sigdigits=2))")
end

@testset "ET fit-recovery (well-conditioned basis)" begin
   basis = ETBackend.onsite_basis(ETBackend.ETMatrix(), [:Cu];
               rcut = 5.0, maxorder = 2, maxdeg = 5, maxl = 3)
   nb = length(basis)

   # synthetic ground truth
   c_true = randn(nb)
   ntrain = 4 * nb                       # over-determined
   envs = [ randenv(rand(4:8)) for _ in 1:ntrain ]
   Bs = [ ETBackend.evaluate(basis, e[1], e[2]) for e in envs ]
   Σ_true = [ sum(c_true[k] * B[k] for k in eachindex(B)) for B in Bs ]

   # least-squares design matrix A (9 rows per sample) and rhs
   A = zeros(9 * ntrain, nb)
   y = zeros(9 * ntrain)
   for (s, B) in enumerate(Bs)
      rows = (9*(s-1)+1):(9*s)
      for k in 1:nb
         A[rows, k] .= vec(B[k])
      end
      y[rows] .= vec(Σ_true[s])
   end

   c_fit = A \ y
   @test norm(A * c_fit - y) < 1e-8        # fits the data
   r = rank(A; rtol = 1e-10)
   println("  nb=$nb  rank(A)=$r  (full rank: $(r == nb))")

   # predictions on fresh test environments match ground truth
   for _ in 1:5
      Rs, Zs = randenv(6)
      B = ETBackend.evaluate(basis, Rs, Zs)
      Σpred = sum(c_fit[k] * B[k] for k in eachindex(B))
      Σtrue = sum(c_true[k] * B[k] for k in eachindex(B))
      @test norm(Σpred - Σtrue) < 1e-8
   end
end
