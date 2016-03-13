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

# Mr. Pursuit!
# ------------

type PursuitPlayer <: Player
    v::ValueEstimator
    π::Vector{Float64}  # Probabilities of actions

    β::Float64  # Learning-rate of action-probabilities.

    function PursuitPlayer(v::ValueEstimator, β::Real)
        0 < β || error("PursuitPlayer needs positive β, not $β")
        new(v, ones(length(v.Qa))/length(v.Qa), β)
    end
end

choose_action(p::PursuitPlayer) = rand(Categorical(p.π))

function update!(p::PursuitPlayer, r, a)
    update!(p.v, r, a)

    A = best(p.v)
    for i in eachindex(p.π)
        @inbounds p.π[i] += p.β * ((i==A) - p.π[i])
    end
end

# Mr. Preference pursuit!
# ------------
# My take on exercise 2.13/14: a pursuit adjusting preferences towards
# the greedy action according to the value estimator.

type PrefPursuitPlayer <: Player
    v::ValueEstimator
    p::Vector{Float64}  # Preferences for actions

    β::Float64  # Learning-rate of preferences.

    # Whether or not to decrease β for very likely actions.
    # This is an unnamed hack described in Exercise 2.11 which is supposed to
    # help the problem with too pessimistic r̂0.
    # Not sure what the name of this is.
    softer::Bool

    function PrefPursuitPlayer(v::ValueEstimator, β::Real, softer=false)
        0 < β || error("ReinfCompPlayer needs positive β, not $β")
        new(v, zeros(length(v.Qa)), β, softer)
    end
end

choose_action(p::PrefPursuitPlayer) = rand(Categorical(softmax(p.p)))

function update!(p::PrefPursuitPlayer, r, a)
    update!(p.v, r, a)

    A = best(p.v)

    β = p.softer ? p.β*(1 - softmax(p.p)[a]) : p.β
    p.p[A] += β * (p.v.Qa[A] - p.v.Qa[a])  # Update preferences towards greedy action.
end

