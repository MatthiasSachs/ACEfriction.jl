using ACEfriction, Test, LinearAlgebra, StaticArrays

@testset "ACEfriction.jl" begin
    # EquivariantTensors backend unit suite (bases, models, serialization, fitting format)
    @testset "ET backend" begin include("etbackend/runtests.jl") end

    # public-API integration on the ET backend (constructors -> Gamma/Sigma -> Flux fit)
    @testset "cutover integration" begin include("test_cutover_integration.jl") end

    # I/O data round-trip
    @testset "I/O data" begin include("./test_IO_data.jl") end

    # NOTE: the legacy model-fit tests (test_ac_model_fit.jl, test_pwcsc_model_fit.jl,
    # test_pwcec_model_fit.jl) and their helper `create_frictionmodels.jl` use the old
    # ACEfrictionCore property types and constructor signatures (positional `evalcenter`,
    # `maxorder_on`/`rcut_on`, ...). They need porting to the ET-backend public API
    # (property markers exported by ACEfriction; single `maxorder`/`maxdeg`/`rcut`).
    # Superseded for now by `test_cutover_integration.jl`. TODO: port + re-enable.
end
