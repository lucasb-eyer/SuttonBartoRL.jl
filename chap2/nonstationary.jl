using Distributions
using PyPlot
plt[:style][:use]("ggplot")

include("utils.jl")

# The machine!
# ============
immutable RandomWalkMultiBandit
    μ::Vector{Float64}
    σ::Vector{Float64}
    v::Vector{Float64}  # "velocity" of the walks
end

rwmbandit(n) = RandomWalkMultiBandit(zeros(n), ones(n), ones(n))
rwmbandit(n, v::Real) = RandomWalkMultiBandit(zeros(n), ones(n), fill(float(v), n))

import Base.length
length(b::RandomWalkMultiBandit) = length(b.μ)

function play(b::RandomWalkMultiBandit, a)
    # Random walk!
    b.μ[:] += b.v .* randn(length(b))

    # New random reward.
    randn()*b.σ[a] + b.μ[a]
end
best(b::RandomWalkMultiBandit) = indmax(b.μ)

# The players!
# ============

abstract ValueEstimator

# All players use the same way of estimating value for this plot.
immutable SampleAverage <: ValueEstimator
    Qa::Vector{Float64}
    Na::Vector{Int64}

    SampleAverage(n) = new(zeros(n), zeros(Int64, n))
end

sampleavg_ve(b::RandomWalkMultiBandit) = SampleAverage(length(b))

function update!(v::SampleAverage, r, a)
    # Very simple "empirical mean" estimator.
    v.Na[a] += 1
    v.Qa[a] += 1/v.Na[a] * (r - v.Qa[a])
end

immutable DecayingSampleAverage <: ValueEstimator
    Qa::Vector{Float64}
    α::Vector{Float64}

    DecayingSampleAverage(n, α) = all(0 .< α .<= 1) ? new(zeros(n), α) : error("Invalid weight for moving sample average value-estimator")
end

decayavg_ve(b::RandomWalkMultiBandit, α::Vector{Float64}) = DecayingSampleAverage(length(b), α)
decayavg_ve(b::RandomWalkMultiBandit, α::Float64) = decayavg_ve(b, fill(α, length(b)))

function update!(v::DecayingSampleAverage, r, a)
    # Very simple "empirical mean" estimator.
    v.Qa[a] += v.α[a] * (r - v.Qa[a])
end

abstract Player

# Generic playing
function play!(p::Player, b::RandomWalkMultiBandit)
    a = choose_action(p)
    r = play(b, a)
    update!(p, r, a)
    return a, r
end

# Generic updating only updates value estimator
update!(p::Player, r, a) = update!(p.v, r, a)

# Mr. greedy!
# -----------

immutable GreedyPlayer <: Player
    v::ValueEstimator
end

choose_action(p::GreedyPlayer) = indmax(p.v.Qa)

# Mr. ϵ-greedy!
# -------------

immutable ϵGreedyPlayer <: Player
    v::ValueEstimator
    ϵ::Float64
end

# With a small probability, go randomly, else just GO GREEDY
choose_action(p::ϵGreedyPlayer) = rand() < p.ϵ ? rand(1:length(p.v.Qa)) : indmax(p.v.Qa)

# Mr. τ-greedy!  (my invention, but probably already exists and has a name!)
# -------------

immutable τGreedyPlayer <: Player
    v::ValueEstimator
    τ::Int
end

# If we're in the first n0 plays, go randomly, else just GO GREEDY
choose_action(p::τGreedyPlayer) = sum(p.v.Na) < p.τ ? rand(1:length(p.v.Qa)) : indmax(p.v.Qa)

# Mr. Softmax!
# ------------

immutable SoftMaxPlayer <: Player
    v::ValueEstimator
    τ::Float64
end

choose_action(p::SoftMaxPlayer) = rand(Categorical(softmax(p.v.Qa, p.τ)))

# Thug Aim!
# =========

const NROUNDS=2000
const NGAMES=1000

fig, (ax1, ax2) = subplots(2, 1)

for (mkplayer, name) in [
    ((b)->τGreedyPlayer(sampleavg_ve(b), 100), L"$\tau=100$ Greedy, static"),
    ((b)->ϵGreedyPlayer(sampleavg_ve(b), 0.1), L"$\epsilon=0.1$ Greedy, sample-avg"),
    ((b)->ϵGreedyPlayer(decayavg_ve(b, 0.1), 0.1), L"$\epsilon=0.1$ Greedy, $\alpha=0.1$-decaying"),
]
    println("Playing ", name, "...")

    rewards = zeros(NROUNDS, NGAMES)
    actions = zeros(Int64, NROUNDS, NGAMES)
    bestact = zeros(Int64, NROUNDS, NGAMES)
    for g=1:NGAMES
        bandit = rwmbandit(10)
        player = mkplayer(bandit)

        for r=1:NROUNDS
            actions[r,g], rewards[r,g] = play!(player, bandit)
            bestact[r,g] = best(bandit)
        end
    end

    ax1[:plot](mean(rewards, 2), label=name)
    ax2[:plot](mean(actions .== bestact, 2)*100, label=name)
end

fig[:suptitle]("10-armed bandit", fontsize=16)
ax1[:set_xlabel]("Epoch [plays]")
ax2[:set_xlabel]("Epoch [plays]")
ax1[:set_ylabel]("Average reward")
ax2[:set_ylabel]("Optimal action [%]")
ax1[:legend](loc="lower right")
ax2[:legend](loc="lower right")
show()
