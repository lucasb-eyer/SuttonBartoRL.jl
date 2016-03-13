# The multi-bandit
# ================
immutable MultiBandit
    μ::Vector{Float64}
    σ::Vector{Float64}
end

multibandit(n) = MultiBandit(randn(n), ones(n))

import Base.length
length(b::MultiBandit) = length(b.μ)

play(b::MultiBandit, a) = randn()*b.σ[a] + b.μ[a]
best(b::MultiBandit) = indmax(b.μ)

# The moving multi-bandit
# ================
immutable RandomWalkMultiBandit
    μ::Vector{Float64}
    σ::Vector{Float64}
    v::Vector{Float64}  # "velocity" of the walks
end

rwmbandit(n) = RandomWalkMultiBandit(zeros(n), ones(n), ones(n))
rwmbandit(n, v::Real) = RandomWalkMultiBandit(zeros(n), ones(n), fill(float(v), n))

import Base.length
length(b::RandomWalkMultiBandit) = length(b.μ)

function play(b::RandomWalkMultiBandit, a)
    # Random walk!
    b.μ[:] += b.v .* randn(length(b))

    # New random reward.
    randn()*b.σ[a] + b.μ[a]
end
best(b::RandomWalkMultiBandit) = indmax(b.μ)
