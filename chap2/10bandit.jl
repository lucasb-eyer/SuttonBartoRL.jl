using Distributions
using PyPlot
plt[:style][:use]("ggplot")

# The machine!
# ============
immutable MultiBandit
    μ::Vector{Float64}
    σ::Vector{Float64}
end

multibandit(n) = MultiBandit(randn(n), ones(n))

import Base.length
length(b::MultiBandit) = length(b.μ)

play(b::MultiBandit, a) = randn()*b.σ[a] + b.μ[a]
best(b::MultiBandit) = indmax(b.μ)

# The players!
# ============

abstract ValueEstimator

# All players use the same way of estimating value for this plot.
immutable SampleAverage <: ValueEstimator
    Qa::Vector{Float64}
    Na::Vector{Int64}

    SampleAverage(n) = new(zeros(n), zeros(Int64, n))
end

sampleavg_ve(b::MultiBandit) = SampleAverage(length(b))

function update!(v::SampleAverage, r, a)
    # Very simple "empirical mean" estimator.
    v.Na[a] += 1
    v.Qa[a] += 1/v.Na[a] * (r - v.Qa[a])
end

abstract Player

# Generic playing
function play!(p::Player, b::MultiBandit)
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

# Compute softmax of `x` at temperature `τ`
function softmax(x, τ) ex = exp(x/τ) ; return ex / sum(ex) end

choose_action(p::SoftMaxPlayer) = rand(Categorical(softmax(p.v.Qa, p.τ)))

# Thug Aim!
# =========

const NROUNDS=2000
const NGAMES=1000

fig, (ax1, ax2) = subplots(2, 1)

for (mkplayer, name) in [
    ((ve)-> GreedyPlayer(ve), "Greedy"),
    ((ve)->ϵGreedyPlayer(ve, 0.1), L"$\epsilon$=0.1 Greedy"),
    ((ve)->ϵGreedyPlayer(ve, 0.01), L"$\epsilon$=0.01 Greedy"),
    ((ve)->τGreedyPlayer(ve, 10), L"$\tau=10$ Greedy"),
    ((ve)->τGreedyPlayer(ve, 100), L"$\tau=100$ Greedy"),
    ((ve)->SoftMaxPlayer(ve, 0.1), L"$\tau=0.1$ SoftMax"),
    ((ve)->SoftMaxPlayer(ve, 1), L"$\tau=1$ SoftMax"),
]
    println("Playing ", name, "...")

    rewards = zeros(NROUNDS, NGAMES)
    actions = zeros(Int64, NROUNDS, NGAMES)
    bestact = zeros(Int64, NGAMES)
    for g=1:NGAMES
        bandit = multibandit(10)
        player = mkplayer(sampleavg_ve(bandit))

        for r=1:NROUNDS
            actions[r,g], rewards[r,g] = play!(player, bandit)
        end

        bestact[g] = best(bandit)
    end

    ax1[:plot](mean(rewards, 2), label=name)
    ax2[:plot](mean(actions .== bestact', 2)*100, label=name)
end

fig[:suptitle]("10-armed bandit", fontsize=16)
ax1[:set_xlabel]("Epoch [plays]")
ax2[:set_xlabel]("Epoch [plays]")
ax1[:set_ylabel]("Average reward")
ax2[:set_ylabel]("Optimal action [%]")
ax1[:legend](loc="lower right")
ax2[:legend](loc="lower right")
show()
