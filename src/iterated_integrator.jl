"""
    AbstractIntegrator{F} <: Function

Supertype of integration routines of a (collection of) function `F`.
"""
abstract type AbstractIntegrator{F} <: Function end

# interface

function limits end # return the limits object(s) used by the routine
function quad_integrand end # return the integrand used by the quadrature routine, optionally with arguments
quad_routine(f::AbstractIntegrator) = f.routine # integration routine
quad_args(f::AbstractIntegrator, ps...) = (quad_integrand(f, ps...), limits(f)...) # integration routine arguments
quad_kwargs(f::AbstractIntegrator; kwargs...) = (; f.kwargs..., kwargs...) # keyword arguments to routine

# abstract methods

(f::AbstractIntegrator)(ps...; kwargs...) =
    quad_routine(f)(quad_args(f, ps...)...; quad_kwargs(f; kwargs...)...)

# implementations

"""
    IteratedIntegrator(f, l, p...; ps=0.0, routine=iterated_integration, args=(), kwargs=(;))

!!! warning "Experimental"
    Intended to integrate all kinds of [`AbstractIteratedIntegrand`](@ref)
"""
struct IteratedIntegrator{F<:AbstractIteratedIntegrand,L<:AbstractLimits,P<:Tuple,R,K<:NamedTuple} <: AbstractIntegrator{F}
    f::F
    l::L
    p::P
    routine::R
    kwargs::K
end
function IteratedIntegrator(f::F, l, p...; ps=0.0, routine=iterated_integration, kwargs...) where F
    test = IteratedIntegrand{F}(p..., ps...)
    IteratedIntegrator(f, l, p, routine, quad_kwargs(routine, test, l; kwargs...))
end

limits(f::IteratedIntegrator) = (f.l,)
IteratedIntegrator{F}(args...; kwargs...) where {F<:Function} =
    IteratedIntegrator(F.instance, args...; kwargs...)

# default arguments

"""
    quad_args(routine, f, lims, [...])

Return the tuple of arguments needed by the quadrature `routine` other than the
test integrand  `f`, depending on the test integrand `f` and `lims`.
"""
quad_args(::typeof(quadgk), f, segs::T...) where T = segs
quad_args(::typeof(quadgk), f, lims::NTuple{N,T}) where {N,T} = lims
quad_args(::typeof(quadgk), f, lims::CubicLimits{1}) = endpoints(lims)
quad_args(::typeof(iterated_integration), f, lims::CubicLimits{1}) = (lims,)

"""
    quad_kwargs(routine, f, lims, kwargs::NamedTuple)

Supplies the default keyword arguments to the given integration `routine`
without over-writing those already provided in `kwargs`
"""
quad_kwargs(::typeof(quadgk), f, lims; kwargs...) = quad_kwargs(quadgk, f, quad_args(quadgk, f, lims)...; kwargs...)
function quad_kwargs(::typeof(quadgk), f, segs::T...;
    atol=zero(T), rtol=iszero(atol) ? sqrt(eps(T)) : zero(atol),
    order=7, maxevals=10^7, norm=norm, segbuf=nothing) where T
    F = Base.promote_op(f, T)
    segbuf_ = segbuf === nothing ? alloc_segbuf(T, F, Base.promote_op(norm, F)) : segbuf
    (rtol=rtol, atol=atol, order=order, maxevals=maxevals, norm=norm, segbuf=segbuf_)
end
quad_kwargs(::typeof(iterated_integration), f, l::AbstractLimits; kwargs...) =
    iterated_integration_kwargs(f, l; kwargs...)