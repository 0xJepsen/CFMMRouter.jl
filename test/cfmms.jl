init_opt_cache(R) = (R⁺=similar(R), ∇ϕR=similar(R))

function optimality_conditions_met(c, Δ, Λ, cfmm; cache=nothing)
    R, γ = cfmm.R, cfmm.γ
    if isnothing(cache)
        cache = init_opt_cache(R)
    end
    R⁺ = cache.R⁺
    ∇ϕR = cache.∇ϕR
    n = length(c)

    # Primal feasibility
    R⁺ = R + γ * Δ - Λ
    pfeas = all(Δ .≥ 0) && all(Λ .≥ 0)
    ϕR = CR.ϕ(cfmm)
    ϕR⁺ = CR.ϕ(cfmm; R=R⁺)
    CR.∇ϕ!(∇ϕR, cfmm; R=R⁺)
    cfmm_sat = ≈(ϕR, ϕR⁺) && ϕR⁺ ≥ ϕR - sqrt(eps())

    opt = maximum(i -> γ * ∇ϕR[i] / c[i], 1:n) ≤ minimum(i -> ∇ϕR[i] / c[i], 1:n) + sqrt(eps())
    return pfeas && cfmm_sat && opt
end
@testset "CFMMs" begin
    @testset "arbitrage checks: two coins" begin
        Δ = MVector{2,Float64}(undef)
        Λ = MVector{2,Float64}(undef)

        n = 3
        Random.seed!(1234)
        γs = [rand() for i in 1:n]
        Rs = [rand(2) * 10 for i in 1:n]
        νs = [@MVector rand(2) for i in 1:n]
        d = Normal(0.95, 0.35)
        σs = rand(Truncated(d, 0.1, 1.4), n) # Standard deviation of price. Range gausian between [.6,1.3]
        # println(rand(Normal(i, j), n) for i in νs, j in σs)
        Ks = [@MVector rand(Normal(i, j)) for i in νs, j in σs] # near price vector
        println(Ks')
        τs = [((1 / 52) - (i - 1) * (1 / 154)) for i in 1:n] # max is 1/52 (week at most). Start max, reduce by step size
        cache = init_opt_cache(Rs[1])

        @testset "product" begin
            equal_pool = ProductTwoCoin([1, 1], 1, [1, 2])

            # No arb in the fee-less case
            v = MVector(1.0, 1.0)
            find_arb!(Δ, Λ, equal_pool, v)
            @test iszero(Δ) && iszero(Λ)

            v .*= 2
            find_arb!(Δ, Λ, equal_pool, v)
            @test iszero(Δ) && iszero(Λ)

            # Easy arb in the fee-less case
            v[2] = 1
            find_arb!(Δ, Λ, equal_pool, v)
            @test Δ[1] ≈ 0 && Δ[2] ≈ sqrt(2) - 1
            @test Λ[1] ≈ 1 - sqrt(1 / 2) && Λ[2] ≈ 0

            # No-arb in the fee case
            @test length(ProductTwoCoin([1, 1], 0.9, [1, 2])) == 2
            @test_throws ArgumentError ProductTwoCoin([1, 1], 0.9, [1])

            for R in Rs, γ in γs, ν in νs
                cfmm = ProductTwoCoin(R, γ, [1, 2])
                find_arb!(Δ, Λ, cfmm, ν)
                @test optimality_conditions_met(ν, Δ, Λ, cfmm; cache=cache)
            end

        end

        @testset "geo mean" begin
            ws = [(w1 = rand(); [w1; 1 - w1]) for _ in 1:n]
            for R in Rs, γ in γs, ν in νs, w in ws
                cfmm = GeometricMeanTwoCoin(R, w, γ, [1, 2])
                find_arb!(Δ, Λ, cfmm, ν)
                @test optimality_conditions_met(ν, Δ, Λ, cfmm; cache=cache)
            end
        end
        @testset "Primitive_RMM_01" begin
            for R in Rs, γ in γs, ν in νs, σ in σs, K in Ks, τ in τs
                cfmm = Primitive_RMM_01(R, γ, [1, 2], σ, τ, K)
                find_arb!(Δ, Λ, cfmm, ν)
                @test optimality_conditions_met(ν, Δ, Λ, cfmm; cache=cache)
            end
        end
    end

end