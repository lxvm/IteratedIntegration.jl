"""
Package for auxiliary integration, i.e. integrating multiple functions at the same time
while ensuring that each converges to its own tolerance. This has a few advantages over
vector-valued integrands with custom norms in that the errors from different integrands can
be treated separated and the adaptive algorithm can decide which integrand to prioritize
based on whether others have already converged.
This results in conceptually simpler algorithms, especially when the various integrands may
differ in order of magnitude.

This module heavily reuses the source code of QuadGK.jl

# Statement of need

Calculating integrals of the form a^2/(f(x)^2+a^2)^2 is challenging
in the a -> 0 limit because they become extremely localized while also having vanishingly
small tails. I.e. the tails are O(a^2) however the integral is O(a^-1). Thus, setting an
absolute tolerance is impractical, since the tolerance also needs to be O(a^2) to resolve
the tails (otherwise the peaks will be missed) and that may be asking for many more digits
than desired. Another option is to specify a relative tolerance, but a failure mode is that
if there is more than one peak to integrate, the algorithm may only resolve the first one
because the errors in the tails to find the other peaks become eclipsed by the first peak
error magnitudes. When the peak positions are known a priori, the convential solution is to
pass in several breakpoints to the integration interval so that each interval has at most
one peak, but often finding breakpoints can be an expensive precomputation that is better
avoided. Instead, an integrand related to the original may more reliably find the peaks
without requiring excessive integrand evaluations or being expensive to compute. Returning
to the original example, an ideal auxiliary integrand would be 1/(f(x)+im*a)^2, which has
O(1) tails and a O(1) integral. Thus the tails will be resolved in order to find the peaks,
which don't need to be resolved to many digits of accuracy. However, since one wants to find
the original integral to a certain number of digits, it may be necessary to adapt further
after the auxiliary integrand has converged. This is the problem the package aims to solve.

# Example

    f(x)    = sin(x)/(cos(x)+im*1e-5)   # peaked "nice" integrand
    imf(x)  = imag(f(x))                # peaked difficult integrand
    f2(x)   = f(x)^2                    # even more peaked
    imf2(x) = imf(x)^2                  # even more peaked!

    x0 = 0.1    # arbitrary offset of between peak

    function integrand(x)
        re, im = reim(f2(x) + f2(x-x0))
        AuxValue(imf2(x) + imf2(x-x0), re)
    end

    using QuadGK    # plain adaptive integration

    quadgk(x -> imf2(x) + imf2(x-x0), 0, 2pi, atol = 1e-5)   # 1.4271103714584847e-7
    quadgk(x -> imf2(x) + imf2(x-x0), 0, 2pi, rtol = 1e-5)   # 235619.45750214785

    quadgk(x -> imf2(x), 0, 2pi, rtol = 1e-5)   # 78539.81901117883

    quadgk(x -> imf2(x-x0), 0, 2pi, rtol = 1e-5)   # 157079.63263294287

    using AuxQuad   # auxiliary integration

    auxquadgk(integrand, 0, 2pi, atol=1e-2) # 628318.5306881254
    auxquadgk(integrand, 0, 2pi, rtol=1e-2) # 628318.5306867635

As can be seen from the example, plain integration can completely fail to capture the
integral despite using stringent tolerances. With a well-chosen auxiliary integrand, often
arising naturally from the structure of the integrand, the integration is much more robust
to error because it can resolve the regions of interest with the more-easily adaptively
integrable problem.
"""
module RuleQuad

using QuadGK: handle_infinities, Segment, cachedrule, InplaceIntegrand, alloc_segbuf
using DataStructures, LinearAlgebra
import Base.Order.Reverse
import QuadGK: evalrule

export auxquadgk, AuxValue, Sequential, Parallel


struct AuxValue{T}
    val::T
    aux::T
end

Base.zero(::Type{AuxValue{T}}) where {T} = AuxValue(zero(T), zero(T))

struct KeyOrdering{T<:Base.Order.Ordering} <: Base.Order.Ordering
    o::T
    k::Symbol
end
Base.Order.lt(o::KeyOrdering, a::Number, b::AuxValue) =
    Base.Order.lt(o.o, a, getproperty(b, o.k))
Base.Order.lt(o::KeyOrdering, a::AuxValue, b::Number) =
    Base.Order.lt(o.o, getproperty(a, o.k), b)
Base.Order.lt(o::KeyOrdering, a::T, b::T) where {T<:AuxValue} =
    Base.Order.lt(o.o, getproperty(a, o.k), getproperty(b, o.k))

const IntegrandsSegment{TI,TE} = Segment{<:Any,<:AuxValue{TI},<:AuxValue{TE}}

Base.Order.lt(o::KeyOrdering, a::Number, b::IntegrandsSegment) =
    Base.Order.lt(o, a, b.E)
Base.Order.lt(o::KeyOrdering, a::IntegrandsSegment, b::Number) =
    Base.Order.lt(o, a.E, b)
Base.Order.lt(o::KeyOrdering, a::T, b::T) where {T<:IntegrandsSegment} =
    Base.Order.lt(o, a.E, b.E)

# first refine the auxiliary, then the true value
eachorder(::AuxValue) = (KeyOrdering(Reverse, :aux), KeyOrdering(Reverse, :val))
eachorder(::Any) = (Reverse,)   # fallback to normal quadgk ordering for other types

LinearAlgebra.norm(a::AuxValue) = AuxValue(norm(a.val), norm(a.aux))
Base.size(a::AuxValue) = size(a.val)
Base.eltype(::Type{AuxValue{T}}) where T = T
Base.:+(a::AuxValue, b::AuxValue) = AuxValue(a.val+b.val, a.aux+b.aux)
Base.:-(a::AuxValue, b::AuxValue) = AuxValue(a.val-b.val, a.aux-b.aux)
Base.:*(a::AuxValue, b::AuxValue) = AuxValue(a.val*b.val, a.aux*b.aux)
Base.:*(a::AuxValue, b) = AuxValue(a.val*b, a.aux*b)
Base.:*(a, b::AuxValue) = AuxValue(a*b.val, a*b.aux)
Base.:/(a::AuxValue, b) = AuxValue(a.val/b, a.aux/b)
Base.:/(a, b::AuxValue) = AuxValue(a/b.val, a/b.aux)
Base.:/(a::AuxValue, b::AuxValue) = AuxValue(a.val/b.val, a.aux/b.aux)
Base.zero(a::AuxValue) = AuxValue(zero(a.val), zero(a.aux))
Base.isinf(a::AuxValue) = isinf(a.val) || isinf(a.aux)
Base.isnan(a::AuxValue) = isnan(a.val) || isnan(a.aux)

Base.isless(a::AuxValue, b::AuxValue) = isless(a.aux, b.aux) && isless(a.val, b.val)
Base.isless(a::AuxValue, b) = isless(a.aux, b) && isless(a.val, b)
Base.isless(a, b::AuxValue) = isless(a, b.aux) && isless(a, b.val)

# strict error comparisons (De Morgan's Laws)
Base.:>(a::AuxValue, b::AuxValue) = >(a.val, b) || >(a.aux, b)
Base.:>(a::AuxValue, b) = >(a.val, b) || >(a.aux, b)
Base.:>(a, b::AuxValue) = >(a, b.val) || >(a, b.aux)

Base.:<(a::AuxValue, b::AuxValue) = <(a.val, b) && <(a.aux, b)
Base.:<(a::AuxValue, b) = <(a.val, b) && <(a.aux, b)
Base.:<(a, b::AuxValue) = <(a, b.val) && <(a, b.aux)

Base.isequal(a::AuxValue, b::AuxValue) = isequal(a.val, b.val) && isequal(a.aux, b.aux)
Base.isequal(a::AuxValue, b) = isequal(a.val, b) && isequal(a.aux, b)
Base.isequal(a, b::AuxValue) = isequal(a, b.val) && isequal(a, b.aux)
# Base.max(a::AuxValue, b::AuxValue) = AuxValue(max(a.val, b.val), max(a.aux, b.aux))
# Base.max(a::AuxValue, b) = AuxValue(max(a.val, b), max(a, b.aux))
# Base.max(a, b::AuxValue) = AuxValue(max(a, b.val), max(a, b.aux))


struct Sequential end

evalrule(::Nothing, f::F, a,b, x,w,gw, nrm) where F = evalrule(f, a,b, x,w,gw, nrm)

struct Parallel{T,S}
    f::Vector{T} # array to store function evaluations
    old_segs::Vector{S} # array to store segments popped off of heap
    new_segs::Vector{S} # array to store segments to add to heap
end


"""
    Parallel(domain_type=Float64, range_type=Float64, error_type=Float64; order=7)

This helper will allocate a buffer to parallelize `quadgk` calls across function evaluations
with a given `domain_type`, i.e. the type of the integration limits, `range_type`, i.e. the
type of the range of the integrand, and `error_type`, the type returned by the `norm` given
to `quadgk`. The keyword `order` allocates enough memory so that the Gauss-Kronrod rule of
that order can initially be evaluated without additional allocations. By passing this buffer
to multiple compatible `quadgk` calls, they can all be parallelized without repeated
allocation.
"""
function Parallel(TX=Float64, TI=Float64, TE=Float64; order=7)
    return Parallel(Vector{TI}(undef, 2*order+1), alloc_segbuf(TX,TI,TE), alloc_segbuf(TX,TI,TE, size=2))
end


function batcheval!(fx, f::F, x, a, s, l,n) where {F}
    Threads.@threads for i in 1:n
        z = i <= l ? x[i] : -x[n-i+1]
        fx[i] = f(a + (1 + z)*s)
    end
    return fx
end
function batcheval!(fx, f::InplaceIntegrand{F}, x, a, s, l,n) where {F}
    Threads.@threads for i in 1:n
        z = i <= l ? x[i] : -x[n-i+1]
        fx[i] = zero(f.fx) # allocate the output
        f.f!(fx[i], a + (1 + z)*s)
    end
    return fx
end

function parevalrule(fx, f::F, a,b, x,w,gw, nrm,l,n) where {F}
    n1 = 1 - (l & 1) # 0 if even order, 1 if odd order
    s = convert(eltype(x), 0.5) * (b-a)
    batcheval!(fx, f, x, a, s, l,n)
    if n1 == 0 # even: Gauss rule does not include x == 0
        Ik = fx[l] * w[end]
        Ig = zero(Ik)
    else # odd: don't count x==0 twice in Gauss rule
        f0 = fx[l]
        Ig = f0 * gw[end]
        Ik = f0 * w[end] + (fx[l-1] + fx[l+1]) * w[end-1]
    end
    for i = 1:length(gw)-n1
        fg = fx[2i] + fx[n-2i+1]
        fk = fx[2i-1] + fx[n-2i+2]
        Ig += fg * gw[i]
        Ik += fg * w[2i] + fk * w[2i-1]
    end
    Ik_s, Ig_s = Ik * s, Ig * s # new variable since this may change the type
    E = nrm(Ik_s - Ig_s)
    if isnan(E) || isinf(E)
        throw(DomainError(a+s, "integrand produced $E in the interval ($a, $b)"))
    end
    return Segment(oftype(s, a), oftype(s, b), Ik_s, E)
end

function evalrule(fx, f::F, a,b, x,w,gw, nrm) where {F}
    l = length(x)
    n = 2*l-1   # number of Kronrod points
    return parevalrule(fx, f, a,b, x,w,gw, nrm, l,n)
end

struct GaussKronrod{T<:Real}
    x::Vector{T}
    w::Vector{T}
    wg::Vector{T}
end

@inline GaussKronrod(T,n) = GaussKronrod(cachedrule(T, n)...)
Base.eltype(::Type{GaussKronrod{T}}) where T = T
countevals(g::GaussKronrod) = 2length(g.w)-1
@inline function (g::GaussKronrod)(f::F, s, nrm=norm, buffer=nothing) where {F}
    a, b = s
    return evalrule(buffer, f, a,b, g.x,g.w,g.wg, nrm)
end

# inlining this seems to help with inference
# function eval_segs(::Sequential, ::Nothing, s::NTuple{1,T}, f::F, rule, nrm::M) where {F,T,M}
#     return (rule(f, s[1], nrm),)
# end
function eval_segs(::Sequential, ::Nothing, s::NTuple{N,T}, f::F, rule, nrm::M) where {N,F,T,M}
    return ntuple(i -> rule(f, s[i], nrm), Val(N))
end
function eval_segs(p::Sequential, ::Vector, s::NTuple, f::F, rule, nrm) where {F}
    return eval_segs(p, nothing, s, f, rule, nrm)
end
function eval_segs(::Sequential, segbuf::Vector, segitr, f::F, rule, nrm) where {F}
    resize!(segbuf, length(segitr))
    for (i, s) in zip(eachindex(segbuf), segitr)
        segbuf[i] = rule(f, s, nrm)
    end
    return segbuf
end
function eval_segs(::Sequential, ::Nothing, segitr, f::F, rule, nrm) where {F}
    return map(s -> rule(f, s, nrm), segitr) # allocation for generic iterator
end
function eval_segs(p::Parallel{T,S}, segbuf, segitr, f::F, rule, nrm) where {F,T,S}
    n = countevals(rule)
    m = length(segitr)
    segs = segbuf === nothing ? Vector{S}(undef, m) : resize!(segbuf, m)
    (nm = n*m) <= length(p.f) || resize!(p.f, nm)
    Threads.@threads for i in 1:m
        s = segitr[i]
        fx = view(p.f, (1+(i-1)*n):(i*n))
        segs[i] = rule(f, s, nrm, fx)
    end
    return segs
end


# we have multiple codepaths
alloc_heap(::Parallel, _, segs) = segs # we already allocated for multi-threading
alloc_heap(::Sequential, _, segs) = segs  # input segments were an iterator, so we already allocated
alloc_heap(::Sequential, ::Nothing, segs::Tuple) = collect(segs)  # this & next same as quadgk (for tuple input segments)
function alloc_heap(::Sequential, segbuf::Vector{T}, segs::NTuple{N,T}) where {N,T}
    resize!(segbuf, N)
    segbuf .= segs
    return segbuf
end

# Internal routine: integrate f over an index-able iterator of segments
# [(a1, b1), (a2, b2), ...], (optimized for tuples) using h-adaptive
# integration with the order-n Kronrod rule and weights of type Tw,
# with absolute tolerance atol and relative tolerance rtol,
# with maxevals an approximate maximum number of f evaluations.
function do_auxquad(f::F, segitr::S, rule, atol, rtol, maxevals, nrm::M, segbuf, parallel) where {F,S,M}

    numevals = countevals(rule)
    numevals *= N = length(segitr)
    @assert N ≥ 1
    segs = eval_segs(parallel, segbuf, segitr, f, rule, nrm)
    if f isa InplaceIntegrand
        I = f.I .= segs[1].I
        for i = 2:length(segs)
            I .+= segs[i].I
        end
    else
        I = sum(s -> s.I, segs)
    end
    E = sum(s -> s.E, segs)

    # logic here is mainly to handle dimensionful quantities: we
    # don't know the correct type of atol115, in particular, until
    # this point where we have the type of E from f.  Also, follow
    # Base.isapprox in that if atol≠0 is supplied by the user, rtol
    # defaults to zero.
    T = float(real(eltype(eltype(segitr))))
    atol_ = something(atol, zero(E))
    rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(one(T))) : zero(T))

    # optimize common case of no subdivision
    if E ≤ atol_ || E ≤ rtol_ * nrm(I) || numevals ≥ maxevals
        return (I, E) # fast return when no subdivisions required
    end

    segheap = alloc_heap(parallel, segbuf, segs)
    for ord in eachorder(I)
        heapify!(segheap, ord)
        I, E, numevals = auxadapt(f, segheap, I, E, numevals, rule, atol_, rtol_, maxevals, nrm, ord, parallel)
        (E ≤ atol_ || E ≤ rtol_ * nrm(I) || numevals ≥ maxevals) && break
    end
    return (I, E)
end


# pop segments that contribute most to error
function pop_segs(::Sequential, segs, ord, E, tol)
    return (heappop!(segs, ord),)
end
function pop_segs(p::Parallel, segs, ord, E, tol)
    empty!(p.old_segs)
    while Base.Order.lt(ord, E, tol) && !isempty(segs)
        s = heappop!(segs, ord)
        E -= s.E
        push!(p.old_segs, s)
    end
    return p.old_segs
end

# bisect segments
function bisect_segs(p::Sequential, (s,), f::F, rule, nrm) where {F}
    mid = (s.a + s.b) / 2
    return eval_segs(p, nothing, ((s.a, mid), (mid, s.b)), f, rule, nrm)
end

function bisect_segs(p::Parallel, old_segs, f::F, rule, nrm) where {F}
    gen = ((s.a, (s.a+s.b)/2, s.b) for s in old_segs)
    lims = [(lb, ub) for (a, mid, b) in gen for (lb, ub) in ((a, mid), (mid, b))]
    eval_segs(p, p.new_segs, lims, f, rule, nrm)
end


# internal routine to perform the h-adaptive refinement of the integration segments (segs)
function auxadapt(f::F, segs::Vector{T}, I, E, numevals, rule, atol, rtol, maxevals, nrm, ord, parallel) where {F, T}
    # Pop the biggest-error segment and subdivide (h-adaptation)
    # until convergence is achieved or maxevals is exceeded.
    evalsperseg = countevals(rule)
    while (tol = max(atol, rtol*nrm(I)); Base.Order.lt(ord, E, tol)) && numevals < maxevals
        old_segs = pop_segs(parallel, segs, ord, E, tol)
        new_segs = bisect_segs(parallel, old_segs, f, rule, nrm)

        if f isa InplaceIntegrand
            for i = eachindex(old_segs)
                I .-= old_segs[i].I
            end
            for i = eachindex(new_segs)
                I .+= new_segs[i].I
            end
        else
            I = (I - sum(s -> s.I, old_segs)) + sum(s -> s.I, new_segs)
        end
        E = (E - sum(s -> s.E, old_segs)) + sum(s -> s.E, new_segs)
        numevals += length(new_segs)*evalsperseg

        # handle type-unstable functions by converting to a wider type if needed
        Tj = promote_type(T, map(typeof, new_segs)...)
        if Tj !== T
            segs_ = Vector{Tj}(segs)
            for s in new_segs
                heappush!(segs_, s, ord)
            end
            return auxadapt(f, segs_,
                         I, E, numevals, rule, atol, rtol, maxevals, nrm, ord, parallel)
        end

        for s in new_segs
            heappush!(segs, s, ord)
        end
    end

    # re-sum (paranoia about accumulated roundoff)
    if f isa InplaceIntegrand
        I .= segs[1].I
        E = segs[1].E
        for i in 2:length(segs)
            I .+= segs[i].I
            E += segs[i].E
        end
    else
        I = segs[1].I
        E = segs[1].E
        for i in 2:length(segs)
            I += segs[i].I
            E += segs[i].E
        end
    end
    return (I, E, numevals)
end

auxquadgk(f, segs...; kws...) =
    auxquadgk(f, promote(segs...)...; kws...)

initial_segs(s::NTuple{N}) where N = ntuple(i -> (s[i],s[i+1]), Val(N-1))
function auxquadgk(f, segs::T...; rule = nothing,
       atol=nothing, rtol=nothing, maxevals=10^7, order=7, norm=norm, segbuf=nothing, parallel=Sequential()) where {T}
    rule_ = (rule===nothing) ? GaussKronrod(T, order) : rule

    handle_infinities(f, segs) do g, s, _
        do_auxquad(g, initial_segs(s), rule_, atol, rtol, maxevals, norm, segbuf, parallel)
    end
end


end
