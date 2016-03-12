using Distributions
using PyPlot
plt[:style][:use]("ggplot")

include("utils.jl")
include("value_estimators.jl")
include("players.jl")

# The machine!
# ============
immutable BinaryBandit
    p::Tuple{Float64,Float64} # Probability of win and lose, respectively.
    r::Tuple{Float64,Float64} # Reward for win and lose, respectively.

    function BinaryBandit(p::Tuple{Float64,Float64}, r::Tuple{Float64,Float64})
        if 0 <= p[1] <= 1 && 0 <= p[2] <= 1
            new(p, r)
        else
            error("Invalid probabilities $p passed to BinaryBandit")
        end
    end
end

bandit(psucc, pfail, rsucc=1, rfail=-1) = BinaryBandit((float(psucc),float(pfail)), (float(rsucc),float(rfail)))

import Base.length
length(::BinaryBandit) = 2

play(b::BinaryBandit, a::Int64) = b.r[2-Int(rand() < b.p[a])]
best(b::BinaryBandit) = indmax(b.p)

# Players specialized to the binary task!
# ===============

# Mr. Supervised!
# ---------------

immutable SupervisedPlayer <: Player
    v::ValueEstimator
end

choose_action(p::SupervisedPlayer) = best(p.v)

function update!(p::SupervisedPlayer, r, a)
    update!(p.v,  r, a)
    # Infer the opposite outcome for the other choice.
    # NOTE: This is only easily feasible in the binary case!
    update!(p.v, -r, [2,1][a])
end

# Linear Reward-Penalty
# ---------------------

immutable LRPPlayer <: Player
    π::Vector{Float64}
    α::Real

    LRPPlayer(α::Real, n::Int) = new(ones(n)/n, α)
end

choose_action(p::LRPPlayer) = rand(Categorical(p.π))

function update!(p::LRPPlayer, r, a)
    # Infer which choice would've been correct:
    right = (0 < r ? [1,2] : [2,1])[a]
    wrong = (0 < r ? [2,1] : [1,2])[a]

    # Adapt their probabilities.
    Δ = p.α * (1 - p.π[right])
    p.π[right] += Δ
    p.π[wrong] -= Δ

    # F*ck numerical stability :-/
    clamp!(p.π, 0, 1)
end

# Linear Reward-Inaction
# ----------------------

immutable LRIPlayer <: Player
    π::Vector{Float64}
    α::Real

    LRIPlayer(α::Real, n::Int) = new(ones(n)/n, α)
end

choose_action(p::LRIPlayer) = rand(Categorical(p.π))

function update!(p::LRIPlayer, r, a)
    # Only update probabilities on success!
    if 0 < r
        Δ = p.α * (1 - p.π[a])
        p.π[a] += Δ
        p.π[[2,1][a]] -= Δ

        clamp!(p.π, 0, 1)  # F*ck numerical stability :-/
    end
end

# Thug Aim!
# =========

const NROUNDS=2000
const NGAMES=1000

fig, (ax1, ax2) = subplots(2, 1, figsize=(8,8))

for (probs, ax) in (
    [(0.1, 0.2), ax1],
    [(0.8, 0.9), ax2]
)
    bs = [bandit(probs...) for i=1:NGAMES]

    for (mkplayer, name) in [
        ((n)->SupervisedPlayer(SampleAverage(n)), "Supervised"),
        ((n)->ϵGreedyPlayer(SampleAverage(n), 0.1), L"$\epsilon$=0.1 Greedy"),
        ((n)->ϵGreedyPlayer(SampleAverage(n), 0.01), L"$\epsilon$=0.01 Greedy"),
        ((n)->LRPPlayer(0.1, n), L"0.1 $L_{R-P}$"),
        ((n)->LRIPlayer(0.1, n), L"0.1 $L_{R-I}$"),
        ((n)->LRIPlayer(0.01, n), L"0.01 $L_{R-I}$"),
    ]
        println("Playing ", name, "...")
        actions = [play!(p, b)[1] for _=1:NROUNDS, (p, b) in zip(map(mkplayer, map(length, bs)), bs)]
        ax[:plot](mean(actions .== map(best, bs)', 2), label=name)
    end

    ax[:set_ylabel]("Optimal action [%]")
end

ax2[:set_xlabel]("Epochs [plays]")
ax1[:set_title]("Bandit A (mostly losing)")
ax2[:set_title]("Bandit B (mostly winning)")
fig[:suptitle]("Hard binary bandits", fontsize=16)
leg = fig[:legend](ax1[:get_legend_handles_labels]()..., loc="lower right", bbox_to_anchor=(0.9, 0.1), ncol=2)
show()
