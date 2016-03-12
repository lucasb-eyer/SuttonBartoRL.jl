abstract Player

# Generic playing
function play!(p::Player, game)
    a = choose_action(p)
    r = play(game, a)
    update!(p, r, a)
    return a, r
end

# By default, assume that any player has a value-estimator `v`
# and that the only learning is to update its value-estimator.
update!(p::Player, r, a) = update!(p.v, r, a)

# Mr. greedy!
# -----------

immutable GreedyPlayer <: Player
    v::ValueEstimator
end

choose_action(p::GreedyPlayer) = best(p.v)

# Mr. ϵ-greedy!
# -------------

immutable ϵGreedyPlayer <: Player
    v::ValueEstimator
    ϵ::Float64
end

# With a small probability, go randomly, else just GO GREEDY
choose_action(p::ϵGreedyPlayer) = rand() < p.ϵ ? rand(1:length(p.v.Qa)) : best(p.v)

# Mr. τ-greedy!  (my invention, but probably already exists and has a name!)
# -------------

immutable τGreedyPlayer <: Player
    v::ValueEstimator
    τ::Int
end

# If we're in the first τ plays, go randomly, else just GO GREEDY
choose_action(p::τGreedyPlayer) = sum(p.v.Na) < p.τ ? rand(1:length(p.v.Qa)) : best(p.v)

# Mr. Softmax!
# ------------

immutable SoftMaxPlayer <: Player
    v::ValueEstimator
    τ::Float64
end

choose_action(p::SoftMaxPlayer) = rand(Categorical(softmax(p.v.Qa, p.τ)))

