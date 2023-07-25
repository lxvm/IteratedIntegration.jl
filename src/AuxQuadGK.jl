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

    using AuxQuadGK # auxiliary integration

    auxquadgk(integrand, 0, 2pi, atol=1e-2) # 628318.5306881254
    auxquadgk(integrand, 0, 2pi, rtol=1e-2) # 628318.5306867635

As can be seen from the example, plain integration can completely fail to capture the
integral despite using stringent tolerances. With a well-chosen auxiliary integrand, often
arising naturally from the structure of the integrand, the integration is much more robust
to error because it can resolve the regions of interest with the more-easily adaptively
integrable problem.
"""
module AuxQuadGK

using QuadGK: Segment, cachedrule, InplaceIntegrand, alloc_segbuf, realone
using DataStructures, LinearAlgebra
import Base.Order.Reverse
import QuadGK: evalrule, handle_infinities

export auxquadgk, auxquadgk!, AuxValue, BatchIntegrand


struct AuxValue{T}
    val::T
    aux::T
end

for op in (:zero, :one, :oneunit)
    @eval Base.$op(a::AuxValue) = AuxValue($op(a.val), $op(a.aux))
    @eval Base.$op(::Type{AuxValue{T}}) where {T} = AuxValue($op(T), $op(T))
end

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
Base.max(a::AuxValue, b::AuxValue) = AuxValue(max(a.val, b.val), max(a.aux, b.aux))
Base.max(a::AuxValue, b) = AuxValue(max(a.val, b), max(a.aux, b))
Base.max(a, b::AuxValue) = AuxValue(max(a, b.val), max(a, b.aux))
function Base.isapprox(a::AuxValue, b::AuxValue; kwargs...)
    return isapprox(a.val, b.val; kwargs...) & isapprox(a.aux, b.aux; kwargs...)
end

"""
    BatchIntegrand(f!, y::AbstractArray, x::AbstractVector, max_batch=typemax(Int))

Constructor for a `BatchIntegrand` accepting an integrand of the form `f!(y,x) = y .= f.(x)`
that can evaluate the integrand at multiple quadrature nodes using, for example, threads,
the GPU, or distributed-memory. The `max_batch` keyword limits the number of nodes passed to
the integrand, and it must be at least `4*order+2` to evaluate two GK rules simultaneously.
The buffers `y,x` must both be `resize!`-able since the number of evaluation points may vary
between calls to `f!`.
"""
struct BatchIntegrand{F,Y,X}
    # in-place function f!(y, x) that takes an array of x values and outputs an array of results in-place
    f!::F
    y::Y
    x::X
    max_batch::Int # maximum number of x to supply in parallel
    function BatchIntegrand(f!, y::AbstractArray, x::AbstractVector, max_batch::Integer=typemax(Int))
        max_batch > 0 || throw(ArgumentError("maximum batch size must be positive"))
        return new{typeof(f!),typeof(y),typeof(x)}(f!, y, x, max_batch)
    end
end


"""
    BatchIntegrand(f!, y, x; max_batch=typemax(Int))

Constructor for a `BatchIntegrand` with pre-allocated buffers.
"""
BatchIntegrand(f!, y, x; max_batch::Integer=typemax(Int)) =
    BatchIntegrand(f!, y, x, max_batch)

"""
    BatchIntegrand(f!, y::Type, x::Type=Nothing; max_batch=typemax(Int))

Constructor for a `BatchIntegrand` whose range type is known. The domain type is optional.
Array buffers for those types are allocated internally.
"""
BatchIntegrand(f!, Y::Type, X::Type=Nothing; max_batch::Integer=typemax(Int)) =
    BatchIntegrand(f!, Y[], X[], max_batch)

function evalrule(f::BatchIntegrand{T}, a,b, x,w,gw, nrm) where {T}
    fx = f.y
    l = length(x)
    n = 2l - 1 # number of Kronrod points
    n1 = 1 - (l & 1) # 0 if even order, 1 if odd order
    s = convert(eltype(x), 0.5) * (b-a)
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

function evalrule(g::BatchIntegrand{F}, a,b, x,w,gw, nrm) where {F<:InplaceIntegrand}
    # Ik and Ig are integrals via Kronrod and Gauss rules, respectively
    s = convert(eltype(x), 0.5) * (b-a)
    l = length(x)
    n = 2l - 1 # number of Kronrod points
    n1 = 1 - (l & 1) # 0 if even order, 1 if odd order
    f = g.f!
    fx = g.y
    idx = CartesianIndices(Base.front(axes(fx)))
    fg, fk, Ig, Ik = f.fg, f.fk, f.Ig, f.Ik # pre-allocated temporary arrays
    if n1 == 0 # even: Gauss rule does not include x == 0
        Ik .= fx[idx,l] .* w[end]
        Ig .= zero.(Ik)
    else # odd: don't count x==0 twice in Gauss rule
        Ig .= fx[idx,l] .* gw[end]
        Ik .= fx[idx,l] .* w[end] .+ (fx[idx,l-1] .+ fx[idx,l+1]) .* w[end-1]
    end
    for i = 1:length(gw)-n1
        fg .= fx[idx,2i]   .+ fx[idx,n-2i+1]
        fk .= fx[idx,2i-1] .+ fx[idx,n-2i+2]
        Ig .+= fg .* gw[i]
        Ik .+= fg .* w[2i] .+ fk .* w[2i-1]
    end
    Ik_s = Ik * s # new variable since this may change the type
    f.Idiff .= Ik_s .- Ig .* s
    E = nrm(f.Idiff)
    if isnan(E) || isinf(E)
        throw(DomainError(a+s, "integrand produced $E in the interval ($a, $b)"))
    end
    return Segment(oftype(s, a), oftype(s, b), Ik_s, E)
end

function evalrules(f::BatchIntegrand{F}, s::NTuple{N}, x,w,gw, nrm) where {F,N}
    l = length(x)
    m = 2l-1    # evaluations per segment
    n = (N-1)*m # total evaluations
    resize!(f.x, n)
    resizelastdim!(f.y, n)
    for i in 1:(N-1)    # fill buffer with evaluation points
        a = s[i]; b = s[i+1]
        c = convert(eltype(x), 0.5) * (b-a)
        o = (i-1)*m
        f.x[l+o] = a + c
        for j in 1:l-1
            f.x[j+o] = a + (1 + x[j]) * c
            f.x[m+1-j+o] = a + (1 - x[j]) * c
        end
    end
    if f.f! isa InplaceIntegrand    # evaluate integrand
        f.f!.f!(f.y, f.x)
    else
        f.f!(f.y, f.x)
    end
    # ax = ntuple(_ -> (:), Val(ndims(v)-1))
    idx = CartesianIndices(Base.front(axes(f.y)))
    return ntuple(Val(N-1)) do i
        # g = BatchIntegrand(f.f!, view(f.y, ax..., (1+(i-1)*m):(i*m)), f.x, f.max_batch)
        g = BatchIntegrand(f.f!, view(f.y, idx, (1+(i-1)*m):(i*m)), f.x, f.max_batch)
        return evalrule(g, s[i], s[i+1], x,w,gw, nrm)
    end
end

function resizelastdim!(A, n)
    s = size(A)
    return resize!(A, ntuple(i -> i == ndims(A) ? n : s[i], Val(ndims(A)))...)
end

# we refine as many segments as we can fit into the buffer
function auxrefine(f::BatchIntegrand{F}, segs::Vector{T}, I, E, numevals, x,w,gw,n, tol, atol, rtol, maxevals, nrm, ord) where {F, T}
    nsegs = 0
    len = length(segs)
    l = length(x)
    m = 2l-1 # == 2n+1

    # collect as many segments that will have to be evaluated for the current tolerance
    # while staying under max_batch and maxevals
    while len > nsegs && 2m*(nsegs+1) <= f.max_batch && Base.Order.lt(ord, E, tol) && numevals < maxevals
        # same as heappop!, but moves segments to end of heap/vector to avoid allocations
        s = segs[1]
        y = segs[len-nsegs]
        segs[len-nsegs] = s
        nsegs += 1
        tol += s.E
        numevals += 2m
        len > nsegs && DataStructures.percolate_down!(segs, 1, y, Reverse, len-nsegs)
    end

    resize!(f.x, 2m*nsegs)
    resizelastdim!(f.y, 2m*nsegs)
    for i in 1:nsegs    # fill buffer with evaluation points
        s = segs[len-i+1]
        mid = (s.a+s.b)/2
        for (j,a,b) in ((2,s.a,mid), (1,mid,s.b))
            c = convert(eltype(x), 0.5) * (b-a)
            o = (2i-j)*m
            f.x[l+o] = a + c
            for k in 1:l-1
                f.x[k+o] = a + (1 + x[k]) * c
                f.x[m+1-k+o] = a + (1 - x[k]) * c
            end
        end
    end
    if f.f! isa InplaceIntegrand    # evaluate integrand
        f.f!.f!(f.y, f.x)
    else
        f.f!(f.y, f.x)
    end
    # ax = ntuple(_ -> (:), Val(ndims(v)-1))
    idx = CartesianIndices(Base.front(axes(f.y)))
    resize!(segs, len+nsegs)
    for i in 1:nsegs    # evaluate segments and update estimates & heap
        s = segs[len-i+1]
        mid = (s.a + s.b)/2
        # g1 = BatchIntegrand(f.f!, view(f.y, ax..., 1+2(i-1)*m:(2i-1)*m), f.x, f.max_batch)
        g1 = BatchIntegrand(f.f!, view(f.y, idx, 1+2(i-1)*m:(2i-1)*m), f.x, f.max_batch)
        s1 = evalrule(g1, s.a,mid, x,w,gw, nrm)
        # g2 = BatchIntegrand(f.f!, view(f.y, ax..., 1+(2i-1)*m:2i*m), f.x, f.max_batch)
        g2 = BatchIntegrand(f.f!, view(f.y, idx, 1+(2i-1)*m:2i*m), f.x, f.max_batch)
        s2 = evalrule(g2, mid,s.b, x,w,gw, nrm)
        if f.f! isa InplaceIntegrand
            I .= (I .- s.I) .+ s1.I .+ s2.I
        else
            I = (I - s.I) + s1.I + s2.I
        end
        E = (E - s.E) + s1.E + s2.E
        # the order of operations of placing segments onto the heap is different
        segs[len-i+1] = s1
        segs[len+i]   = s2
    end
    for i in 1:2nsegs
        DataStructures.percolate_up!(segs, len-nsegs+i, Reverse)
    end

    return I, E, numevals
end

function handle_infinities(workfunc, f::BatchIntegrand, s)
    s1, s2 = s[1], s[end]
    if realone(s1) && realone(s2) # check for infinite or semi-infinite intervals
        inf1, inf2 = isinf(s1), isinf(s2)
        if inf1 || inf2
            xtmp = f.x # buffer to store evaluation points
            ytmp = f.y # original integrand may have different units
            xbuf = similar(xtmp, typeof(one(eltype(f.x))))
            ybuf = similar(ytmp, typeof(oneunit(eltype(f.y))*oneunit(s1)))
            g! = f.f! isa InplaceIntegrand ? f.f!.f! : f.f!
            if inf1 && inf2 # x = t/(1-t^2) coordinate transformation
                g = (v, t) -> begin
                    resize!(xtmp, length(t)); resizelastdim!(ytmp, length(v))
                    g!(ytmp, xtmp .= oneunit(s1) .* t ./ (1 .- t .* t))
                    uscale!(v, ytmp, t, t -> (1 + t*t) * oneunit(s1) / (1 - t*t)^2)
                end
                h! = f.f! isa InplaceIntegrand ? InplaceIntegrand(g, f.f!.I, f.f!.fx * oneunit(s1)) : g
                return workfunc(BatchIntegrand(h!, ybuf, xbuf, f.max_batch),
                                map(x -> isinf(x) ? (signbit(x) ? -one(x) : one(x)) : 2x / (oneunit(x)+hypot(oneunit(x),2x)), s),
                                t -> oneunit(s1) * t / (1 - t^2))
            end
            let (s0,si) = inf1 ? (s2,s1) : (s1,s2) # let is needed for JuliaLang/julia#15276
                if si < zero(si) # x = s0 - t/(1-t)
                    g = (v, t) -> begin
                        resize!(xtmp, length(t)); resizelastdim!(ytmp, length(v))
                        g!(ytmp, xtmp .= s0 .- oneunit(s1) .* t ./ (1 .- t))
                        uscale!(v, ytmp, t, t -> oneunit(s1) / (1 - t)^2)
                    end
                    h! = f.f! isa InplaceIntegrand ? InplaceIntegrand(g, f.f!.I, f.f!.fx * oneunit(s1)) : g
                    return workfunc(BatchIntegrand(h!, ybuf, xbuf, f.max_batch),
                                    reverse(map(x -> 1 / (1 + oneunit(x) / (s0 - x)), s)),
                                    t -> s0 - oneunit(s1)*t/(1-t))
                else # x = s0 + t/(1-t)
                    g = (v, t) -> begin
                        resize!(xtmp, length(t)); resizelastdim!(ytmp, length(v))
                        g!(ytmp, xtmp .= s0 .+ oneunit(s1) .* t ./ (1 .- t))
                        uscale!(v, ytmp, t, t -> oneunit(s1) / (1 - t)^2)
                    end
                    h! = f.f! isa InplaceIntegrand ? InplaceIntegrand(g, f.f!.I, f.f!.fx * oneunit(s1)) : g
                    return workfunc(BatchIntegrand(h!, ybuf, xbuf, f.max_batch),
                                    map(x -> 1 / (1 + oneunit(x) / (x - s0)), s),
                                    t -> s0 + oneunit(s1)*t/(1-t))
                end
            end
        end
    end
    return workfunc(f, s, identity)
end

uscale!(v::AbstractVector, y, ts, f) = v .= y .* f.(ts)
function uscale!(v::AbstractArray, y, ts, f)
    ax = ntuple(_ -> (:), Val(ndims(v)-1))
    for (i,j,t) in zip(axes(v, ndims(v)), axes(y, ndims(y)), ts)
        v[ax...,i] .= view(y, ax..., j) .* f(t)
    end
    return v
end

# Internal routine: integrate f over the union of the open intervals
# (s[1],s[2]), (s[2],s[3]), ..., (s[end-1],s[end]), using h-adaptive
# integration with the order-n Kronrod rule and weights of type Tw,
# with absolute tolerance atol and relative tolerance rtol,
# with maxevals an approximate maximum number of f evaluations.
function do_auxquadgk(f::F, s::NTuple{N,T}, n, atol, rtol, maxevals, nrm, segbuf) where {T,N,F}
    x,w,gw = cachedrule(T,n)

    @assert N ≥ 2
    if f isa BatchIntegrand
        segs = evalrules(f, s, x,w,gw, nrm)
    else
        segs = ntuple(i -> evalrule(f, s[i], s[i+1], x,w,gw, nrm), Val(N-1))
    end
    if f isa InplaceIntegrand
        I = f.I .= segs[1].I
        for i = 2:length(segs)
            I .+= segs[i].I
        end
    else
        I = sum(s -> s.I, segs)
    end
    E = sum(s -> s.E, segs)
    numevals = (2n+1) * (N-1)

    # logic here is mainly to handle dimensionful quantities: we
    # don't know the correct type of atol115, in particular, until
    # this point where we have the type of E from f.  Also, follow
    # Base.isapprox in that if atol≠0 is supplied by the user, rtol
    # defaults to zero.
    atol_ = something(atol, zero(E))
    rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(one(eltype(x)))) : zero(eltype(x)))

    # optimize common case of no subdivision
    if E ≤ atol_ || E ≤ rtol_ * nrm(I) || numevals ≥ maxevals
        return (I, E) # fast return when no subdivisions required
    end

    segheap = segbuf === nothing ? collect(segs) : (resize!(segbuf, N-1) .= segs)
    for ord in eachorder(I)
        heapify!(segheap, ord)
        segheap = auxadapt(f, segheap, I, E, numevals, x,w,gw,n, atol_, rtol_, maxevals, nrm, ord)
        (E ≤ atol_ || E ≤ rtol_ * nrm(I) || numevals ≥ maxevals) && break
    end
    return resum(f, segheap)
end

# internal routine to perform the h-adaptive refinement of the integration segments (segs)
function auxadapt(f::F, segs::Vector{T}, I, E, numevals, x,w,gw,n, atol, rtol, maxevals, nrm, ord) where {F, T}
    # Pop the biggest-error segment and subdivide (h-adaptation)
    # until convergence is achieved or maxevals is exceeded.
    while (tol = max(atol, rtol*nrm(I)); Base.Order.lt(ord, E, tol)) && numevals < maxevals
        next = auxrefine(f, segs, I, E, numevals, x,w,gw,n, tol, atol, rtol, maxevals, nrm, ord)
        next isa Vector && return next # handle type-unstable functions
        I, E, numevals = next
    end
    return segs
end

# internal routine to refine the segment with largest error
function auxrefine(f::F, segs::Vector{T}, I, E, numevals, x,w,gw,n, tol, atol, rtol, maxevals, nrm, ord) where {F, T}
    s = heappop!(segs, Reverse)
    mid = (s.a + s.b) / 2
    s1 = evalrule(f, s.a, mid, x,w,gw, nrm)
    s2 = evalrule(f, mid, s.b, x,w,gw, nrm)
    if f isa InplaceIntegrand
        I .= (I .- s.I) .+ s1.I .+ s2.I
    else
        I = (I - s.I) + s1.I + s2.I
    end
    E = (E - s.E) + s1.E + s2.E
    numevals += 4n+2

    # handle type-unstable functions by converting to a wider type if needed
    Tj = promote_type(typeof(s1), promote_type(typeof(s2), T))
    if Tj !== T
        return auxadapt(f, heappush!(heappush!(Vector{Tj}(segs), s1, ord), s2, ord),
                     I, E, numevals, x,w,gw,n, atol, rtol, maxevals, nrm, ord)
    end

    # continue bisecting if the remaining error surpasses current tolerance
    tol += s1.E + s2.E
    if Base.Order.lt(ord, E, tol)
        next = auxrefine(f, segs, I,E, numevals, x, w,gw,n, tol, atol, rtol, maxevals, nrm, ord)
        next isa Vector && return heappush!(heappush!(next, s1, ord), s2, ord)
        I, E, numevals = next
    end

    heappush!(segs, s1, ord)
    heappush!(segs, s2, ord)

    return I, E, numevals
end

# re-sum (paranoia about accumulated roundoff)
function resum(f, segs)
    if f isa InplaceIntegrand
        I = f.I .= segs[1].I
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
    return (I, E)
end

# Gauss-Kronrod quadrature of f from a to b to c...

"""
    auxquadgk(f, a,b,c...; rtol=sqrt(eps), atol=0, maxevals=10^7, order=7, norm=norm, segbuf=nothing)

Numerically integrate the function `f(x)` from `a` to `b`, and optionally over additional
intervals `b` to `c` and so on. Keyword options include a relative error tolerance `rtol`
(if `atol==0`, defaults to `sqrt(eps)` in the precision of the endpoints), an absolute error tolerance
`atol` (defaults to 0), a maximum number of function evaluations `maxevals` (defaults to
`10^7`), and the `order` of the integration rule (defaults to 7).

Returns a pair `(I,E)` of the estimated integral `I` and an estimated upper bound on the
absolute error `E`. If `maxevals` is not exceeded then `E <= max(atol, rtol*norm(I))`
will hold. (Note that it is useful to specify a positive `atol` in cases where `norm(I)`
may be zero.)

Compared to `quadgk` from QuadGK.jl, `auxquadgk` implements a few more safeguards for
integration of difficult functions. It changes how adaptive refinement is done when using a
relative tolerance to refine all segments with an error above the tolerance (instead of just
the segment with the largest error). Additionally, if an integrand returns an `AuxValue`
then the heap first integrates the auxiliary value followed by the primary by resorting the
heap of segments.

The endpoints `a` et cetera can also be complex (in which case the integral is performed over
straight-line segments in the complex plane). If the endpoints are `BigFloat`, then the
integration will be performed in `BigFloat` precision as well.

!!! note
    It is advisable to increase the integration `order` in rough proportion to the
    precision, for smooth integrands.

More generally, the precision is set by the precision of the integration
endpoints (promoted to floating-point types).

The integrand `f(x)` can return any numeric scalar, vector, or matrix type, or in fact any
type supporting `+`, `-`, multiplication by real values, and a `norm` (i.e., any normed
vector space). Alternatively, a different norm can be specified by passing a `norm`-like
function as the `norm` keyword argument (which defaults to `norm`).

!!! note
    Only one-dimensional integrals are provided by this function. For multi-dimensional
    integration (cubature), there are many different algorithms (often much better than simple
    nested 1d integrals) and the optimal choice tends to be very problem-dependent. See the
    Julia external-package listing for available algorithms for multidimensional integration or
    other specialized tasks (such as integrals of highly oscillatory or singular functions).

The algorithm is an adaptive Gauss-Kronrod integration technique: the integral in each
interval is estimated using a Kronrod rule (`2*order+1` points) and the error is estimated
using an embedded Gauss rule (`order` points). The interval with the largest error is then
subdivided into two intervals and the process is repeated until the desired error tolerance
is achieved.

These quadrature rules work best for smooth functions within each interval, so if your
function has a known discontinuity or other singularity, it is best to subdivide your
interval to put the singularity at an endpoint. For example, if `f` has a discontinuity at
`x=0.7` and you want to integrate from 0 to 1, you should use `auxquadgk(f, 0,0.7,1)` to
subdivide the interval at the point of discontinuity. The integrand is never evaluated
exactly at the endpoints of the intervals, so it is possible to integrate functions that
diverge at the endpoints as long as the singularity is integrable (for example, a `log(x)`
or `1/sqrt(x)` singularity).

For real-valued endpoints, the starting and/or ending points may be infinite. (A coordinate
transformation is performed internally to map the infinite interval to a finite one.)

In normal usage, `auxquadgk(...)` will allocate a buffer for segments. You can
instead pass a preallocated buffer allocated using `alloc_segbuf(...)` as the
`segbuf` argument. This buffer can be used across multiple calls to avoid
repeated allocation.
"""
auxquadgk(f, segs...; kws...) =
    auxquadgk(f, promote(segs...)...; kws...)

function auxquadgk(f, segs::T...;
       atol=nothing, rtol=nothing, maxevals=10^7, order=7, norm=norm, segbuf=nothing) where {T}
    handle_infinities(f, segs) do g, s, _
        do_auxquadgk(g, s, order, atol, rtol, maxevals, norm, segbuf)
    end
end

"""
    auxquadgk!(f!, result, a,b,c...; rtol=sqrt(eps), atol=0, maxevals=10^7, order=7, norm=norm)

Like [`auxquadgk`](@ref), but make use of in-place operations for array-valued integrands (or other mutable
types supporting in-place operations).  In particular, there are two differences from `quadgk`:

1. The function `f!` should be of the form `f!(y, x) = y .= f(x)`.  That is, it writes the
   return value of the integand `f(x)` in-place into its first argument `y`.   (The return
   value of `f!` is ignored.)

2. Like `auxquadgk`, the return value is a tuple `(I,E)` of the estimated integral `I` and the
   estimated error `E`.   However, in `auxquadgk!` the estimated integral is written in-place
   into the `result` argument, so that `I === result`.

Otherwise, the behavior is identical to `auxquadgk`.

For integrands whose values are *small* arrays whose length is known at compile-time,
it is usually more efficient to use `quadgk` and modify your integrand to return
an `SVector` from the [StaticArrays.jl package](https://github.com/JuliaArrays/StaticArrays.jl).
"""
auxquadgk!(f!, result, segs...; kws...) =
    auxquadgk!(f!, result, promote(segs...)...; kws...)

function auxquadgk!(f!, result, a::T,b::T,c::T...; atol=nothing, rtol=nothing, maxevals=10^7, order=7, norm=norm, segbuf=nothing) where {T}
    fx = result / oneunit(T) # pre-allocate array of correct type for integrand evaluations
    f = InplaceIntegrand(f!, result, fx)
    return auxquadgk(f, a, b, c...; atol=atol, rtol=rtol, maxevals=maxevals, order=order, norm=norm, segbuf=segbuf)
end

"""
    auxquadgk(f::BatchIntegrand, a,b,c...; kws...)

Like [`auxquadgk`](@ref), but batches evaluation points for an in-place integrand to evaluate
simultaneously. In particular, there are two differences from `quadgk`

1. The function `f.f!` should be of the form `f!(y, x) = y .= f.(x)`.  That is, it writes
   the return values of the integand `f(x)` in-place into its first argument `y`. (The
   return value of `f!` is ignored.) See [`BatchIntegrand`](@ref) for how to define the
   integrand.

2. `f.max_batch` must be large enough to contain `4*order+2` points to evaluate two Kronrod
   rules simultaneously. Choosing `max_batch=4*order+2` will reproduce the result of
   `quadgk`, however if `max_batch=n*(4*order+2)` up to `2n` Kronrod rules will be evaluated
   together, which can produce different results for integrands with multiple peaks when
   used together with relative tolerances. For an example see the manual
"""
function auxquadgk(f::BatchIntegrand{F,Y,<:AbstractVector{Nothing}}, segs::T...; kws...) where {F,Y,T}
    FT = float(T) # the gk points are floating-point
    g = BatchIntegrand(f.f!, f.y, similar(f.x, FT), f.max_batch)
    return auxquadgk(g, segs...; kws...)
end

"""
    auxquadgk!(f::BatchIntegrand, result, a,b,c...; kws...)

Like [`auxquadgk!`](@ref), but batches evaluation points for an in-place integrand to evaluate
simultaneously. In particular, there are two differences from using `quadgk` with a
`BatchIntegrand`:

1. `f.y` must be an array of dimension `ndims(result)+1` whose first `axes` match those of
   `result`. The last dimension of `y` should be reserved for different Kronrod points, and
   the function `f.f!` should be of the form
   `f!(y,x) = foreach((v,z) -> v .= f(z), eachslice(y, dims=ndims(y)), x)` or

    function f!(y, x)
        idx = CartesianIndices(axes(y)[begin:end-1])
        for (j,i) in zip(axes(y)[end], eachindex(x))
            y[idx,j] .= f(x[i])
        end
    end

2. `f.y` must be `resize!`-able in the last dimension. Consider using
   [ElasticArrays.jl](https://github.com/JuliaArrays/ElasticArrays.jl) for this. Otherwise
   specialize `QuadGK.resizelastdim!(A::T, n)` for your array type `T`.
"""
function auxquadgk!(f::BatchIntegrand, result, a::T,b::T,c::T...; atol=nothing, rtol=nothing, maxevals=10^7, order=7, norm=norm, segbuf=nothing) where {T}
    fx = result / oneunit(T) # pre-allocate array of correct type for integrand evaluations
    @assert Base.front(axes(f.y)) == axes(result)
    g = BatchIntegrand(InplaceIntegrand(f.f!, result, fx), f.y, f.x, f.max_batch)
    return auxquadgk(g, a, b, c...; atol=atol, rtol=rtol, maxevals=maxevals, order=order, norm=norm, segbuf=segbuf)
end

end
