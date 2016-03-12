abstract ValueEstimator

# Sample-average value estimator
# =================

immutable SampleAverage <: ValueEstimator
    Qa::Vector{Float64}
    Na::Vector{Int64}

    SampleAverage(n) = new(zeros(n), zeros(Int64, n))
end

best(v::SampleAverage) = indmax(v.Qa)

function update!(v::SampleAverage, r, a)
    v.Na[a] += 1
    v.Qa[a] += 1/v.Na[a] * (r - v.Qa[a])
end

# Exponential, recency-weighted (decaying) sample-average value estimator
# =================

immutable DecayingSampleAverage <: ValueEstimator
    Qa::Vector{Float64}
    α::Vector{Float64}

    DecayingSampleAverage(n, α) = all(0 .< α .<= 1) ? new(zeros(n), α) : error("Invalid weight for DecayingSampleAverage value-estimator: $α")
    DecayingSampleAverage(n, α::Float64) = DecayingSampleAverage(n, fill(α, n))
end

best(v::DecayingSampleAverage) = indmax(v.Qa)

function update!(v::DecayingSampleAverage, r, a)
    v.Qa[a] += v.α[a] * (r - v.Qa[a])
end
