using Distributions
using PyPlot
plt[:style][:use]("ggplot")

include("utils.jl")
include("value_estimators.jl")
include("players.jl")

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

# Mr. Reinforcement comparison!
# ------------
# Finally, a slightly more complex player!

type ReinfCompPlayer <: Player
    p::Vector{Float64}  # Preferences for actions
    r̄::Float64          # Reference reward, aka baseline.

    α::Float64  # Momentum of baseline tracking.
    β::Float64  # Learning-rate of preferences.

    # Whether or not to decrease β for very likely actions.
    # This is an unnamed hack described in Exercise 2.11 which is supposed to
    # help the problem with too pessimistic r̂0.
    # Not sure what the name of this is.
    softer::Bool

    function ReinfCompPlayer(n::Int, r̄::Real, α::Real, β::Real, softer=false)
        0 < α <= 1 || error("ReinfCompPlayer needs α in (0,1], not $α")
        0 < β || error("ReinfCompPlayer needs positive β, not $β")
        new(zeros(n), r̄, α, β, softer)
    end
end

choose_action(p::ReinfCompPlayer) = rand(Categorical(softmax(p.p)))

function update!(p::ReinfCompPlayer, r, a)
    β = p.softer ? p.β*(1 - softmax(p.p)[a]) : p.β
    p.p[a] +=   β * (r - p.r̄)  # Update preferences.
    p.r̄    += p.α * (r - p.r̄)  # Update baseline.
end

# Thug Aim!
# =========

const NROUNDS=2000
const NGAMES=2000
const bandits = [multibandit(10) for _=1:NGAMES]

fig, (ax1, ax2) = subplots(2, 1, figsize=(8,6))

for (mkplayer, name) in [
    ((n)->ϵGreedyPlayer(SampleAverage(n), 0.1), L"$\epsilon=0.1$ Greedy"),
    ((n)->ReinfCompPlayer(n, 0, 0.1, 0.1), L"$\alpha=0.1, \beta=0.1$ Reinf-comp"),
    ((n)->ReinfCompPlayer(n,-5, 0.1, 0.1), L"$\alpha=0.1, \beta=0.1$ Reinf-comp, pessimist"),
    ((n)->ReinfCompPlayer(n, 0, 0.1, 0.1, true), L"$\alpha=0.1, \beta=0.1$ Reinf-comp, soft"),
    ((n)->ReinfCompPlayer(n,-5, 0.1, 0.1, true), L"$\alpha=0.1, \beta=0.1$ Reinf-comp, soft pessimist"),
]
    println("Playing ", name, "...")

    rewards = zeros(NROUNDS, NGAMES)
    actions = zeros(Int64, NROUNDS, NGAMES)
    bestact = zeros(Int64, NROUNDS, NGAMES)
    for g=1:NGAMES
        bandit = bandits[g]
        player = mkplayer(length(bandit))

        for r=1:NROUNDS
            actions[r,g], rewards[r,g] = play!(player, bandit)
            bestact[r,g] = best(bandit)
        end
    end

    ax1[:plot](mean(rewards, 2), label=name)
    ax2[:plot](mean(actions .== bestact, 2)*100, label=name)
end

fig[:suptitle]("10-armed bandit", fontsize=16)
ax2[:set_xlabel]("Epoch [plays]")
ax1[:set_ylabel]("Average reward")
ax2[:set_ylabel]("Optimal action [%]")
leg = fig[:legend](ax1[:get_legend_handles_labels]()..., loc="right", bbox_to_anchor=(0.9, 0.65), ncol=1)
fatlegend(leg)
# fig[:savefig]("plots/2.2.png", bbox_inches="tight", bbox_extra_artists=(leg,))
show()
