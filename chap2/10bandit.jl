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
length(mb::MultiBandit) = length(mb.μ)

play(mb::MultiBandit, a) = randn()*mb.σ[a] + mb.μ[a]

# The players!
# ============

abstract ValueEstimater

# All players use the same way of estimating value for this plot.
immutable SampleAverage <: ValueEstimater
    Qa::Vector{Float64}
    Na::Vector{Int64}

    SampleAverage(n) = new(zeros(n), zeros(Int64, n))
end

sampleavg_ve(mb::MultiBandit) = SampleAverage(length(mb))

function update!(v::SampleAverage, r, a)
    # Very simple "empirical mean" estimator.
    v.Na[a] += 1
    v.Qa[a] += 1/v.Na[a] * (r - v.Qa[a])
end

abstract Player

# Generic playing
function play!(p::Player, mb::MultiBandit)
    a = choose_action(p)
    r = play(mb, a)
    update!(p, r, a)
    return r
end

# Generic updating only updates value estimator
update!(p::Player, r, a) = update!(p.v, r, a)

# Mr. greedy!
# -----------

immutable GreedyPlayer <: Player
    v::ValueEstimater
end

choose_action(p::GreedyPlayer) = indmax(p.v.Qa)

# Mr. ϵ-greedy!
# -------------

immutable ϵGreedyPlayer <: Player
    v::ValueEstimater
    ϵ::Float64
end

# With a small probability, go randomly, else just GO GREEDY
choose_action(p::ϵGreedyPlayer) = rand() < p.ϵ ? rand(1:length(p.v.Qa)) : indmax(p.v.Qa)

# Mr. τ-greedy!  (my invention, but probably already exists and has a name!)
# -------------

immutable τGreedyPlayer <: Player
    v::ValueEstimater
    τ::Int
end

# If we're in the first n0 plays, go randomly, else just GO GREEDY
choose_action(p::τGreedyPlayer) = sum(p.v.Na) < p.τ ? rand(1:length(p.v.Qa)) : indmax(p.v.Qa)

# Mr. Softmax!
# ------------

immutable SoftMaxPlayer <: Player
    v::ValueEstimater
    τ::Float64
end

# Compute softmax of `x` at temperature `τ`
function softmax(x, τ) ex = exp(x/τ) ; return ex / sum(ex) end
choose_action(p::SoftMaxPlayer) = rand(Categorical(softmax(p.v.Qa, p.τ)))

# Thug Aim!
# =========

const NROUNDS=2000
const NGAMES=2000

mbs = [multibandit(10) for i=1:NGAMES]

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
    players = [mkplayer(sampleavg_ve(mbs[i])) for i=1:length(mbs)]
    rewards = [play!(p, mb) for _=1:NROUNDS, (mb, p) in zip(mbs, players)]

    plot(mean(rewards, 2), label=name)
end

title("10-armed bandit")
xlabel("epochs (plays)")
ylabel("average reward")
legend(loc="lower right")
show()
