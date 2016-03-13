using Distributions
using PyPlot
plt[:style][:use]("ggplot")

include("utils.jl")
include("bandits.jl")
include("value_estimators.jl")
include("players.jl")

# Thug Aim!
# =========

const NROUNDS=1400
const NGAMES=2000

fig, (ax1, ax2) = subplots(2, 1, figsize=(8,6))

for (mkplayer, name) in [
    ((ve)->ϵGreedyPlayer(ve, 0.1), L"$\epsilon=0.1$ Greedy"),
    ((ve)->ϵGreedyPlayer(ve, 0.01), L"$\epsilon=0.01$ Greedy"),
    ((ve)->τGreedyPlayer(ve, 10), L"$\tau=10$ Greedy"),
    ((ve)->τGreedyPlayer(ve, 100), L"$\tau=100$ Greedy"),
    ((ve)->SoftMaxPlayer(ve, 0.1), L"$\tau=0.1$ SoftMax"),
    ((ve)->SoftMaxPlayer(ve, 1), L"$\tau=1$ SoftMax"),
    ((ve)-> GreedyPlayer(ve), "Greedy"),
]
    println("Playing ", name, "...")

    rewards = zeros(NROUNDS, NGAMES)
    actions = zeros(Int64, NROUNDS, NGAMES)
    bestact = zeros(Int64, NROUNDS, NGAMES)
    for g=1:NGAMES
        bandit = multibandit(10)
        player = mkplayer(SampleAverage(length(bandit)))

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
leg = fig[:legend](ax1[:get_legend_handles_labels]()..., loc="right", bbox_to_anchor=(0.9, 0.55), ncol=1)
fatlegend(leg)
# fig[:savefig]("plots/2.2.png", bbox_inches="tight", bbox_extra_artists=(leg,))
show()
