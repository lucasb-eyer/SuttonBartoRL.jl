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

# All players use the same way of estimating value for this plot.
immutable ValueEstimater
    Qa::Vector{Float64}
    Na::Vector{Int}

    ValueEstimater(n) = new(zeros(n), zeros(Int64, n))
end

valueestim(mb::MultiBandit) = ValueEstimater(length(mb))

function update!(v::ValueEstimater, r, a)
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
function softmax(x, τ) ex = exp(x/τ) ; return ex ./ sum(ex) end
# Sample from multinomial (softmax) distribution `p`.
multinomial(p) = findfirst(rand() .< cumsum(p))

choose_action(p::SoftMaxPlayer) = multinomial(softmax(p.v.Qa, p.τ))

# Thug Aim!
# =========

const NROUNDS=2000
const NGAMES=2000

mbs = [multibandit(10) for i=1:NGAMES]
g_players    = [GreedyPlayer(valueestim(mbs[i])) for i=1:length(mbs)]
ϵ01_players  = [ϵGreedyPlayer(valueestim(mbs[i]), 0.1) for i=1:length(mbs)]
ϵ001_players = [ϵGreedyPlayer(valueestim(mbs[i]), 0.01) for i=1:length(mbs)]
τ10_players  = [τGreedyPlayer(valueestim(mbs[i]), 10) for i=1:length(mbs)]
τ100_players = [τGreedyPlayer(valueestim(mbs[i]), 100) for i=1:length(mbs)]
sm01_players = [SoftMaxPlayer(valueestim(mbs[i]), 0.1) for i=1:length(mbs)]
sm1_players  = [SoftMaxPlayer(valueestim(mbs[i]), 1) for i=1:length(mbs)]

# Now y'all Play!
r_g    = [play!(p, mb) for e=1:NROUNDS, (mb, p) in zip(mbs, g_players)]
r_ϵ01  = [play!(p, mb) for e=1:NROUNDS, (mb, p) in zip(mbs, ϵ01_players)]
r_ϵ001 = [play!(p, mb) for e=1:NROUNDS, (mb, p) in zip(mbs, ϵ001_players)]
r_τ10  = [play!(p, mb) for e=1:NROUNDS, (mb, p) in zip(mbs, τ10_players)]
r_τ100 = [play!(p, mb) for e=1:NROUNDS, (mb, p) in zip(mbs, τ100_players)]
r_sm01 = [play!(p, mb) for e=1:NROUNDS, (mb, p) in zip(mbs, sm01_players)]
r_sm1  = [play!(p, mb) for e=1:NROUNDS, (mb, p) in zip(mbs, sm1_players)]

plot(mean(r_g, 2), label="Greedy")
plot(mean(r_ϵ01, 2), label=L"$\epsilon$=0.1 Greedy")
plot(mean(r_ϵ001, 2), label=L"$\epsilon$=0.01 Greedy")
plot(mean(r_τ10, 2), label=L"$\tau=10$ Greedy")
plot(mean(r_τ100, 2), label=L"$\tau=100$ Greedy")
plot(mean(r_sm01, 2), label=L"$\tau=0.1$ SoftMax")
plot(mean(r_sm1 , 2), label=L"$\tau=1$ SoftMax")
title("10-armed bandit")
xlabel("epochs (plays)")
ylabel("average reward")
legend(loc="lower right")
show()
