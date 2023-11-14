struct QuadNest{R,F,L,S,V,K,FX}
    routine::R
    f::F
    l::L
    state::S
    vol::V
    kwargs::K
    fx::FX
end

function (q::QuadNest)(x)
    next = limit_iterate(q.l, q.state, x)
    if next isa SVector
        return q.f(next)
    else
        segs, l, state = next
        g = remake_nest(q.routine, q.f, q.vol, segs, l, state, q.fx)
        I, = q.routine(g, segs; q.kwargs...)
        return I
    end
end

function (q::QuadNest)()
    segs, l, state = limit_iterate(q.l)
    g = remake_nest(q.routine, q.f, q.vol, segs, l, state, q.fx)
    q.routine(g, segs; q.kwargs...)
end

# we use function wrappers for improved compilation time and type stability
function remake_nest(::Function, n::QuadNest, vol, segs, l, state, fx)
    a, b = segs[1], segs[end]
    nextvol = vol * (b-a)   # we rescale the next tolerance by the outer volume to get the right units
    kws = haskey(n.kwargs, :atol) ? merge(n.kwargs, (atol=n.kwargs.atol/nextvol,)) : n.kwargs
    q = QuadNest(n.routine, n.f, l, state, nextvol, kws, n.fx)
    fw = FunctionWrapper{typeof(fx), Tuple{typeof((a+b)/2)}}(q)
    return fw
end

# there are some routines that need to accept real & complex input types, in which case the
# FunctionWrapper does not allow the dynamic behavior we need
# expect the compilation times to skyrocket with high-dimensional integrals
function remake_nest(::typeof(contquadgk), n::QuadNest, vol, segs, l, state, fx)
    a, b = segs[1], segs[end]
    nextvol = vol * (b-a)   # we rescale the next tolerance by the outer volume to get the right units
    kws = haskey(n.kwargs, :atol) ? merge(n.kwargs, (atol=n.kwargs.atol/nextvol,)) : n.kwargs
    return QuadNest(n.routine, n.f, l, state, nextvol, kws, n.fx)
end

function make_nest(f::F, l::AbstractIteratedLimits{d}, routines, kwargs) where {F,d}
    segs, l0, state = limit_iterate(l)
    x = (segs[1] + segs[2])/2
    vol = one(eltype(l))
    nest, fx = make_nest(f, l0, state, x, vol, Base.front(routines), Base.front(kwargs))
    return QuadNest(routines[d], nest, l0, state, vol, kwargs[d], fx)
end

function make_nest(f::F, l::AbstractIteratedLimits{1}, state, x, vol, routines, kwargs) where F
    next = limit_iterate(l, state, x)
    return QuadNest(nothing, f, nothing, nothing, nothing, NamedTuple(), nothing), f(next)
end
function make_nest(f::F, l::AbstractIteratedLimits{d}, state, x, vol, routines, kwargs) where {F,d}
    segs, lx, statex = limit_iterate(l, state, x)
    xx = (segs[1] + segs[2])/2
    volx = vol * oneunit(eltype(l))
    nest, fx = make_nest(f, lx, statex, xx, volx, Base.front(routines), Base.front(kwargs))
    fxx = fx*(segs[2]-segs[1])/2 # TODO, make this complex for the contquadgk routines
    return QuadNest(routines[d-1], nest, lx, state, one(eltype(lx)), kwargs[d-1], fx), fxx
end

"""
    nested_quad(f, a, b; kwargs...)
    nested_quad(f, l::AbstractIteratedLimits{d,T}; routine=quadgk, kwargs...) where {d,T}

Calls `QuadGK` to perform iterated 1D integration of `f` over a compact domain parametrized
by `AbstractIteratedLimits` `l`. In the case two points `a` and `b` are passed, the
integration region becomes the hypercube with those extremal vertices (which mimics
`hcubature`).

Returns a tuple `(I, E)` of the estimated integral and estimated error.

Keyword options include a relative error tolerance `rtol` (if `atol==0`, defaults to
`sqrt(eps)` in the precision of the norm of the return type), an absolute error tolerance
`atol` (defaults to 0), a maximum number of function evaluations `maxevals` for each nested
integral (defaults to `10^7`), and the `order` of the integration rule (defaults to 7).

The algorithm is an adaptive Gauss-Kronrod integration technique: the integral in each
interval is estimated using a Kronrod rule (`2*order+1` points) and the error is estimated
using an embedded Gauss rule (`order` points). The interval with the largest error is then
subdivided into two intervals and the process is repeated until the desired error tolerance
is achieved. This 1D procedure is applied recursively to each variable of integration in an
order determined by `l` to obtain the multi-dimensional integral.
"""
nested_quad(f, a, b; kwargs...) = nested_quad(f, CubicLimits(a, b); kwargs...)
function nested_quad(f::F, l::L; routine=auxquadgk, routines=nothing, kwargs...) where {F,d,T,L<:AbstractIteratedLimits{d,T}}
    routines_ = routines === nothing ? ntuple(i -> routine isa typeof(quadgk) ? myquadgk : routine, Val(d)) : map(r -> r isa typeof(quadgk) ? myquadgk : r, routines)
    kwargs_   = routines === nothing ? ntuple(i -> deepcopy(NamedTuple(kwargs)),  Val(d)) : nt_to_tup(NamedTuple(kwargs), Val(d))
    nest = make_nest(f, l, routines_, kwargs_)
    return nest()
end

function nt_to_tup(kws::NamedTuple, ::Val{d}) where {d}
    return ntuple(Val(d)) do n
        NamedTuple{keys(kws)}(ntuple(Val(length(kws))) do m
            kws[m] isa Tuple ? kws[m][n] : kws[m]
        end)
    end
end
