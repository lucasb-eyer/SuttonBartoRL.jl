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

# Thug Aim!
# =========

const NROUNDS=2000
const NGAMES=1000

fig, (ax1, ax2) = subplots(2, 1)

for (mkplayer, name) in [
    ((n)->ϵGreedyPlayer(DecayingSampleAverage(n, 0.1), 0.1), L"$\epsilon$=0.1 Greedy realist"),
    ((n)-> GreedyPlayer(DecayingSampleAverage(fill(5.0, n), 0.1)), "Greedy optimist"),
]
    println("Playing ", name, "...")

    rewards = zeros(NROUNDS, NGAMES)
    actions = zeros(Int64, NROUNDS, NGAMES)
    bestact = zeros(Int64, NROUNDS, NGAMES)
    for g=1:NGAMES
        bandit = multibandit(10)
        player = mkplayer(length(bandit))

        for r=1:NROUNDS
            actions[r,g], rewards[r,g] = play!(player, bandit)
            bestact[r,g] = best(bandit)
        end
    end

    ax1[:plot](mean(rewards, 2), label=name)
    ax2[:plot](mean(actions .== bestact, 2)*100, label=name)
end

fig[:suptitle]("10-armed bandit: Optimism", fontsize=16)
ax2[:set_xlabel]("Epoch [plays]")
ax1[:set_ylabel]("Average reward")
ax2[:set_ylabel]("Optimal action [%]")
leg = fig[:legend](ax1[:get_legend_handles_labels]()..., loc="lower right", bbox_to_anchor=(0.9, 0.54), ncol=1)
fatlegend(leg)
show()
