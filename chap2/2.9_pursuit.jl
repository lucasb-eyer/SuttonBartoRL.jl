using Distributions
using PyPlot
plt[:style][:use]("ggplot")

include("utils.jl")
include("bandits.jl")
include("value_estimators.jl")
include("players.jl")

# Thug Aim!
# =========

const NROUNDS=2000
const NGAMES=2000
const bandits = [multibandit(10) for _=1:NGAMES]

fig, (ax1, ax2) = subplots(2, 1, figsize=(8,6))

for (mkplayer, name) in [
    ((n)->    ϵGreedyPlayer(SampleAverage(n), 0.1), L"$\epsilon=0.1$ Greedy"),
    ((n)->  ReinfCompPlayer(n, 0, 0.1, 0.1), L"$\alpha=0.1, \beta=0.1$ Reinf-comp"),
    ((n)->    PursuitPlayer(SampleAverage(n), 0.01), L"$\beta=0.01$ Pursuit"),
    ((n)->PrefPursuitPlayer(SampleAverage(n), 0.1), L"$\beta=0.1$ Preference pursuit"),
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

fig[:suptitle]("10-armed bandit: Pursuit", fontsize=16)
ax2[:set_xlabel]("Epoch [plays]")
ax1[:set_ylabel]("Average reward")
ax2[:set_ylabel]("Optimal action [%]")
leg = fig[:legend](ax1[:get_legend_handles_labels]()..., loc="right", bbox_to_anchor=(0.9, 0.65), ncol=1)
fatlegend(leg)
show()
