using Test
using IteratedIntegration: CubicLimits, TetrahedralLimits, ProductLimits, TranslatedLimits
using IteratedIntegration: measure
using LinearAlgebra

function trapz(n::Integer)
    @assert n > 1
    r = range(-1, 1, length=n)
    x = collect(r)
    halfh = step(r)/2
    h = step(r)
    w = [ (i == 1) || (i == n) ? halfh : h for i in 1:n ]
    return (x, w)
end
quad = let n = 30, (x, w) = trapz(n)
    (f, segs) -> begin
        # composite trapezoidal rule
        for (a, b) in zip(segs[begin:end-1], segs[begin+1:end])
            s = (b-a)/2
            I = f((1+x[begin])*s+a) * w[begin]
            for i in 1:n-1
                _x = x[begin+i]
                _w = w[begin+i]
                I += f((1+_x)*s+a) * _w
            end
            return I * s
        end
    end
end

l = CubicLimits((-1,-1,-1), (1,1,1))
@test measure(quad, l) ≈ prod(l.b .- l.a)
l = TetrahedralLimits((1,2,3))
@test measure(quad, l) ≈ abs(det([0;0;3;;0;2;3;;1;2;3])) / factorial(3) atol=1e-3
l = ProductLimits(CubicLimits((-1, -1), (1, 1)), CubicLimits((3.0,), (5.0,)))
@test measure(quad, l) ≈ prod((1, 1) .- (-1, -1)) * 2
l = TranslatedLimits(TetrahedralLimits((1,2,3)), (-3,-4,-5))
@test measure(quad, l) ≈ abs(det([0;0;3;;0;2;3;;1;2;3])) / factorial(3) atol=1e-3