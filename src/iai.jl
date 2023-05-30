struct Node{V,K}
    val::V
    key::K
    idx::Int64
end
Base.getproperty(n::Node, name::Symbol) = getproperty(getfield(n, :val), name)
Base.setproperty!(n::Node, f::Symbol, v) = setproperty!(getfield(n, :val), f, v)

mutable struct HeapSegment{S,TX,TI,TE}
    h::BinaryMaxHeap{Node{S,TE}}
    a::TX
    b::TX
    Ik::TI
    Ig::TI
    E::TE   # total error on this segment
    Eo::TE  # outer error estimator on this segment
    Ek::TE  # inner error estimator on this segment
    # the rule on this panel
    x::Vector{TX}
    w::Vector{TX}
    gw::Vector{TX}
    # the estimates and errors of the inner quadratures at all nodes
    hI::Vector{TI}
    hE::Vector{TE}
end

segvalue(s::Segment) = s.I
segvalue(s::HeapSegment) = s.Ik
segvalue(n::Node) = segvalue(getfield(n, :val))

quaderror(s::Union{Segment,HeapSegment}) = s.E
quaderror(n::Node) = quaderror(getfield(n, :val))

Base.isless(i::Node, j::Node) = isless(getfield(i, :key)*quaderror(i), getfield(j, :key)*quaderror(j))
Base.isless(i::HeapSegment, j::HeapSegment) = isless(i.E, j.E)

# default to quadgk segments for leaf panels
evalrule(::Val{1}, f,l,::Val, a,b, x,w,gw, nrm) = evalrule(f, a,b, x,w,gw, nrm)

function evalsegs(::Val{d}, f::F,l::L,::Val{N},a,b, x,w,gw, nrm) where {d,F,L,N}
    s = iterated_segs(f, a, b, Val(N))
    ntuple(i -> evalrule(Val(d), f,l,Val(N),s[i],s[i+1], x,w,gw, nrm), Val(N))
end

# these functions just for testing
function evalsegs(f, a, b; kwargs...)
    l = CubicLimits(a,b)
    evalsegs(ThunkIntegrand{ndims(l)}(f), l; kwargs...)
end
function evalsegs(f::F, l::L; initdiv=1, order=7, norm=norm) where {F,L}
    x,w,gw = cachedrule(eltype(L),order)
    a,b = endpoints(l)
    evalsegs(Val(ndims(l)), f,l,Val(initdiv),a,b, x,w,gw, norm)
end

function evalnode(::Val{d}, f::F,l::L,::Val{N}, xn, x,w,gw,nrm) where {d,F,L,N}
    fn = iterated_pre_eval(f, xn, Val(d))
    ln = fixandeliminate(l, xn)
    an, bn = endpoints(ln)
    evalsegs(Val(d-1), fn,ln,Val(N),an,bn, x,w,gw, nrm)
end

function evalrule(::Val{d}, f::F,l::L,::Val{N}, a,b, x,w,gw, nrm) where {d,F,L,N}
    p = 2*length(x)-1 # number of points in Kronrod rule
    c = div(p,2)

    s = convert(eltype(x), 0.5) * (b-a)

    # deflated copy of the quadrature rules for convenience
    xs = Vector{eltype(x)}(undef, p)
    ws = Vector{eltype(w)}(undef, p)
    gws = Vector{eltype(gw)}(undef, p)

    n1 = 1 - (length(x) & 1) # 0 if even order, 1 if odd order

    xs[c+1] = a + s; ws[c+1] = w[end] * s; gws[c+1] = gw[end] * s
    f0 = evalnode(Val(d), f,l,Val(N), xs[c+1], x, w, gw, nrm)
    segheap = BinaryMaxHeap(collect(map(v -> Node(v, ws[c+1], c+1), f0)))
    sizehint!(segheap, N*p)

    hI = fill(sum(segvalue,  f0), p)
    hE = fill(sum(quaderror, f0), p)

    if n1 == 0 # even: Gauss rule does not include x == 0
        gws[c+1] = zero(eltype(gw))
    else # odd: don't count x==0 twice in Gauss rule
        # j = 2i; k = p-j+1
        xs[c] = a + (1+x[end-1])*s; ws[c] = w[end-1] * s; gws[c] = zero(eltype(gw))
        f1 = evalnode(Val(d), f,l,Val(N), xs[c], x, w, gw, nrm)
        foreach(v -> push!(segheap, Node(v, ws[c], c)), f1)
        hI[c] = sum(segvalue,  f1)
        hE[c] = sum(quaderror, f1)

        # j = 2i; k = p-j+1
        xs[c+2] = a + (1-x[end-1])*s; ws[c+2] = ws[c]; gws[c+2] = gws[c]
        f2 = evalnode(Val(d), f,l,Val(N), xs[c+2], x, w, gw, nrm)
        foreach(v -> push!(segheap, Node(v, ws[c+2], c+2)), f2)
        hI[c+2] = sum(segvalue,  f2)
        hE[c+2] = sum(quaderror, f2)
    end

    for i = 1:length(gw)-n1
        jk = (2i, p-2i+1)
        for (g, (j, k)) in ((gw[i] * s, jk), (zero(eltype(gw)), (jk[1]-1, jk[2]+1)))
            xs[j] = a + (1+x[j])*s; ws[j] = w[j] * s; gws[j] = g
            f1 = evalnode(Val(d), f,l,Val(N), xs[j], x, w, gw, nrm)
            foreach(v -> push!(segheap, Node(v, ws[j], j)), f1)
            hI[j] = sum(segvalue,  f1)
            hE[j] = sum(quaderror, f1)

            xs[k] = a + (1-x[j])*s; ws[k] = ws[j]; gws[k] = gws[j]
            f2 = evalnode(Val(d), f,l,Val(N), xs[k], x, w, gw, nrm)
            foreach(v -> push!(segheap, Node(v, ws[k], k)), f2)
            hI[k] = sum(segvalue,  f2)
            hE[k] = sum(quaderror, f2)
        end
    end

    # Ik and Ig are integrals via Kronrod and Gauss rules, respectively

    Ik = mapreduce(*, +, ws, hI)
    Ig = mapreduce(*, +, gws, hI)
    Eo = norm(Ik - Ig)
    if isnan(Eo) || isinf(Eo)
        throw(DomainError(a+s, "integrand produced $rE in the interval ($a, $b)"))
    end
    Ek = mapreduce(*, +, ws, hE)
    Eg = mapreduce(*, +, gws, hE)
    E = Eo + Ek

    return HeapSegment(segheap, oftype(s,a),oftype(s,b),Ik,Ig, E,Eo,Ek, xs,ws,gws, hI,hE)
end


# unlike nested_quadgk, this routine won't support iterated_integrand since that
# could change the type of the result and mess up the error estimation
"""
    iai(f, a, b; kwargs...)
    iai(f::AbstractIteratedIntegrand{d}, l::AbstractLimits{d}; order=7, atol=0, rtol=sqrt(eps()), norm=norm, maxevals=typemax(Int), segbuf=nothing) where d

Multi-dimensional globally-adaptive quadrature via iterated integration using
Gauss-Kronrod rules. Interface is similar to [`nested_quadgk`](@ref).
"""
function iai(f, a, b; kwargs...)
    l = CubicLimits(a, b)
    iai(ThunkIntegrand{ndims(l)}(f), l; kwargs...)
end
iai(f::F, l::L; order=7, atol=nothing, rtol=nothing, norm=norm, maxevals=typemax(Int), initdiv=1, segbuf=nothing) where {F,L} =
    do_iai(f, l, Val(something(initdiv, 1)), order, atol, rtol, maxevals, norm, segbuf)

function do_iai(f::F, l::L, ::Val{N}, n, atol, rtol, maxevals, nrm, buf) where {F,L,N}
    T = eltype(l); d = ndims(l); a,b = endpoints(l)
    x,w,gw = cachedrule(T,n)
    p = 2n+1

    @assert N ≥ 1
    (numevals = (N*p)^d) <= maxevals || throw(ArgumentError("maxevals exceeded on initial evaluation"))
    segs = evalsegs(Val(d), f,l,Val(N),a,b, x,w,gw, nrm)
    I = sum(s -> segvalue(s), segs)
    E = sum(s -> quaderror(s), segs)

    # logic here is mainly to handle dimensionful quantities: we
    # don't know the correct type of atol, in particular, until
    # this point where we have the type of E from f. Also, follow
    # Base.isapprox in that if atol≠0 is supplied by the user, rtol
    # defaults to zero.
    atol_ = something(atol, zero(E))
    rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(one(eltype(x)))) : zero(eltype(x)))

    # optimize common case of no subdivision
    if E ≤ atol_ || E ≤ rtol_ * nrm(I) || numevals ≥ maxevals
        return (I, E, 0) # fast return when no subdivisions required
    end

    segheap = buf===nothing ? BinaryMaxHeap(collect(segs)) : (empty!(buf.valtree); push!(buf, segs...); buf)
    adapt!(segheap, Val(d), f,l,a,b, I,E,numevals, x,w,gw,p, atol_, rtol_, maxevals, nrm)
end

# internal routine to perform the h-adaptive refinement of the multi-dimensional integration segments
function adapt!(segheap::BinaryMaxHeap{T}, ::Val{d}, f::F, l::L,a,b, I,E,numevals, x,w,gw,p, atol, rtol, maxevals, nrm) where {T,d,F,L}
    # Pop the biggest-error segment (in any dimension) and subdivide (h-adaptation)
    # until convergence is achieved or maxevals is exceeded.
    while (tol = max(atol, rtol * nrm(I))) < E
        if numevals > maxevals
            @warn "maxevals exceeded"
            break
        end

        s = pop!(segheap)
        I, E, numevals = refine!(segheap, s, Val(d), f,l,a,b, I,E,numevals, x,w,gw,p, nrm,atol,rtol)
    end

    # TODO: re-sum (paranoia about accumulated roundoff)
    return (I, E)
end

function divide(::Val{d}, s, f,l, x,w,gw, nrm) where d
    mid = (s.a + s.b) / 2
    s1 = evalrule(Val(d), f,l,Val(1), s.a,mid, x,w,gw, nrm)
    s2 = evalrule(Val(d), f,l,Val(1), mid,s.b, x,w,gw, nrm)
    return (s1, s2)
end
function divide(::Val{d}, n::Node, f,l, x,w,gw, nrm) where d
    val = getfield(n, :val)
    key = getfield(n, :key)
    idx = getfield(n, :idx)
    s1, s2 = divide(Val(d), val, f,l, x,w,gw, nrm)
    return (Node(s1, key, idx), Node(s2, key, idx))
end

# internal routine to descend adaptive tree and refine it
function refine!(segheap, s, ::Val{d}, f,l,a,b, I,E,numevals, x,w,gw,p, nrm,atol,rtol) where d

#=
    Takeaways
    - multiple integration smooths out functions - penalty dimension dependence?
    - when requesting fewer digits, make the thrashing penalty higher
    - penalty needs to be at least linear and depth^(2 to 3) seems to work well
    - penalty maxima could be dangerous and need to be carefully thought out
=#
    penalty = 1 # thrashing
    # penalty = (b-a) / (s.b-s.a)
    # penalty = log2((b-a) / (s.b-s.a))#^d#^(2-1/log10(max(atol/nrm(I), rtol)))
    # penalty = min((b-a) / (s.b-s.a), 16)
    I -= segvalue(s)
    E -= quaderror(s)
    if d == 1 || s.Eo > s.Ek * penalty
        # refine outer panel
        s1, s2 = divide(Val(d), s, f,l, x,w,gw, nrm)

        numevals += 2*p^d
        I += segvalue(s1) + segvalue(s2)
        E += quaderror(s1) + quaderror(s2)

        push!(segheap, s1, s2)
    else
        # refine inner panel

        n = pop!(s.h)
        i  = getfield(n, :idx)

        xn = s.x[i]
        wn = s.w[i]
        gwn = s.gw[i]

        s.Ik -= wn * s.hI[i]
        s.Ig -= gwn * s.hI[i]
        s.Ek -= wn * s.hE[i]

        fn = iterated_pre_eval(f, xn, Val(d))
        ln = fixandeliminate(l, xn)
        an, bn = endpoints(ln)

        s.hI[i], s.hE[i], numevals = refine!(s.h, n, Val(d-1), fn,ln,an,bn, s.hI[i],s.hE[i],numevals, x,w,gw,p, nrm,atol,rtol)

        s.Ik += wn * s.hI[i]
        s.Ig += gwn * s.hI[i]
        s.Ek += wn * s.hE[i]

        s.Eo = nrm(s.Ik - s.Ig)
        s.E = s.Eo + s.Ek
        I += segvalue(s)
        E += quaderror(s)

        push!(segheap, s)
    end
    return (I, E, numevals)
end

"""
    iai_buffer(ndim, DT, RT, NT)

Allocate a buffer for use by [`iai`](@ref) for an integrand with integration
over `ndim` dimensions with domain type `DT`, range type `RT` and norm type `NT`
"""
iai_buffer(ndim, DT, RT, NT) = BinaryMaxHeap{iai_buffer_(Val(ndim), DT, RT, NT)}()
iai_buffer_(::Val{1}, DT, RT, NT) = Segment{DT,RT,NT}
iai_buffer_(::Val{d}, DT, RT, NT) where d =
    HeapSegment{iai_buffer_(Val(d-1),DT,RT,NT),DT,RT,NT}
