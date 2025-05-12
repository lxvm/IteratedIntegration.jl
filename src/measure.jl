"""
    measure(quad, limits)

Compute the (hyper)volume of the integration domain `limits` using iterated integration based on the quadrature routine `quad(f, segs)` where `f(x)` is a scalar function and `segs` is a segment `(a,b)` or containing breakpoints `(a,b,c)`.
"""
function measure(quad::T, l::AbstractIteratedLimits) where {T}
    if ndims(l) == 1
        a, b = segments(l, 1)
        return abs(b-a)
    else
        quad(x -> measure(quad, fixandeliminate(l, x, Val(ndims(l)))), segments(l, ndims(l)))
    end
end