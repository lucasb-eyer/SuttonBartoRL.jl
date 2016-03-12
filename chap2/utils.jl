# Largely based on Lin Dahua's NumericExtensions.jl implementation!
function softmax!{T<:Real}(dst::AbstractArray{T}, x::AbstractArray{T}, τ::T=1)
    !isempty(x) || error("softmax!: empty array is not allowed.")
    length(dst) == length(x) || error("Inconsistent argument dimensions.")

    u = maximum(x)
    τi = inv(τ)

    s = zero(T)
    @simd for i in eachindex(x)
        @inbounds s += dst[i] = exp((x[i] - u)*τi)
    end

    c = inv(s)
    @simd for i in eachindex(x)
        @inbounds dst[i] *= c
    end

    dst
end

softmax{T<:Real}(x::AbstractArray{T}, τ::T=1) = softmax!(Array(T, size(x)), x, τ)
