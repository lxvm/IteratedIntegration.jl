# this file is kept for proof of concept, however I do not intend to maintain it
mutable struct HeapNode{T,TX,TI,TE}
    h::BinaryMaxHeap{T}
    I::TI
    E::TE
    x::TX
    w::TX
    g::TX
end
mutable struct HeapSegment{T,TX,TI,TE}
    h::BinaryMaxHeap{HeapNode{T,TX,TI,TE}}
    a::TX
    b::TX
    Ik::TI
    Ig::TI
    rE::TE
    qE::TE
    sE::TE
end

segvalue(s::Segment) = s.I
segvalue(s::HeapSegment) = s.Ik

# distinguish between the GK rule error and the multidimensional quadrature error
ruleerror(s::Segment) = s.E
quaderror(s::Segment) = s.E
sorterror(s::Segment) = s.E

ruleerror(s::HeapSegment) = s.rE
quaderror(s::HeapSegment) = s.qE
sorterror(s::HeapSegment) = s.sE

quaderror(n::HeapNode) = n.w * n.E
sorterror(n::HeapNode) = sorterror(first(n.h)) # sort by the largest segment error

Base.isless(i::HeapNode, j::HeapNode) = isless(sorterror(i), sorterror(j))
Base.isless(i::HeapSegment, j::HeapSegment) = isless(sorterror(i), sorterror(j))

# default to quadgk segments for leaf panels
evalrule(::Val{1}, f,l,::Val, a,b, x,w,gw, nrm) = evalrule(f, a,b, x,w,gw, nrm)

function evalsegs(::Val{d}, f::F,l::L,::Val{N}, x,w,gw, nrm) where {d,F,L,N}
    a, b = endpoints(l)
    s = iterated_segs(f, l, a, b, Val(N))
    ntuple(i -> evalrule(Val(d), f,l,Val(N),s[i],s[i+1], x,w,gw, nrm), Val(N))
end

# these functions just for testing
function evalsegs(f, a, b; kwargs...)
    l = CubicLimits(a,b)
    evalsegs(ThunkIntegrand{ndims(l)}(f), l; kwargs...)
end
function evalsegs(f::F, l::L; initdiv=1, order=7, norm=norm) where {F,L}
    x,w,gw = cachedrule(eltype(L),order)
    evalsegs(Val(ndims(l)), f,l,Val(initdiv), x,w,gw, norm)
end

function evalnode(::Val{d}, f::F,l::L,::Val{N}, xn, wn, gwn, x,w,gw,nrm) where {d,F,L,N}
    g = iterated_pre_eval(f, xn, Val(d))
    m = fixandeliminate(l, xn)
    segs = evalsegs(Val(d-1), g,m,Val(N), x,w,gw, nrm)
    segheap = BinaryMaxHeap(collect(segs))
    I = sum(segvalue,  segs)
    E = sum(quaderror, segs)
    HeapNode(segheap, I,E, xn,wn,gwn)
end

function evalrule(::Val{d}, f::F,l::L,::Val{N}, a,b, x,w,gw, nrm) where {d,F,L,N}
    p = 2*length(x)-1 # number of points in Kronrod rule

    s = convert(eltype(x), 0.5) * (b-a)
    n1 = 1 - (length(x) & 1) # 0 if even order, 1 if odd order

    f0 = evalnode(Val(d), f,l,Val(N), a + s, w[end] * s, gw[end] * s, x, w, gw, nrm)
    nodeheap = BinaryMaxHeap{typeof(f0)}()
    sizehint!(nodeheap, p)

    if n1 == 0 # even: Gauss rule does not include x == 0
        f0.g = zero(f0.g)
        push!(nodeheap, f0)

        Ik = f0.I * w[end]
        Ig = zero(Ik)
        qE = f0.E * w[end]

    else # odd: don't count x==0 twice in Gauss rule
        fk1 = evalnode(Val(d), f,l,Val(N), a + (1+x[end-1])*s, w[end-1] * s, zero(gw[end]), x, w, gw, nrm)
        fk2 = evalnode(Val(d), f,l,Val(N), a + (1-x[end-1])*s, w[end-1] * s, zero(gw[end]), x, w, gw, nrm)
        push!(nodeheap, f0, fk1, fk2)

        Ig = f0.I * gw[end]
        Ik = f0.I * w[end] + (fk1.I + fk2.I) * w[end-1]
        qE = f0.E * w[end] + (fk1.E + fk2.E) * w[end-1]
    end

    # Ik and Ig are integrals via Kronrod and Gauss rules, respectively
    # qE is the quadrature of the error

    for i = 1:length(gw)-n1
        fg1 = evalnode(Val(d), f,l,Val(N), a + (1+x[2i])*s, w[2i] * s, gw[i] * s, x, w, gw, nrm)
        fg2 = evalnode(Val(d), f,l,Val(N), a + (1-x[2i])*s, w[2i] * s, gw[i] * s, x, w, gw, nrm)
        fk1 = evalnode(Val(d), f,l,Val(N), a + (1+x[2i-1])*s, w[2i-1] * s, zero(gw[i]), x, w, gw, nrm)
        fk2 = evalnode(Val(d), f,l,Val(N), a + (1-x[2i-1])*s, w[2i-1] * s, zero(gw[i]), x, w, gw, nrm)
        push!(nodeheap, fg1, fg2, fk1, fk2)

        fg = fg1.I + fg2.I
        fk = fk1.I + fk2.I
        Eg = fg1.E + fg2.E
        Ek = fk1.E + fk2.E

        Ig += fg * gw[i]
        Ik += fg * w[2i] + fk * w[2i-1]
        qE += Eg * w[2i] + Ek * w[2i-1]
    end

    Ik_s, Ig_s = Ik * s, Ig * s # new variable since this may change the type
    rE = nrm(Ik_s - Ig_s)
    if isnan(rE) || isinf(rE)
        throw(DomainError(a+s, "integrand produced $rE in the interval ($a, $b)"))
    end
    qE_s = rE + qE * s
    sE = max(rE, sorterror(first(nodeheap)))

    return HeapSegment(nodeheap, oftype(s,a),oftype(s,b), Ik_s,Ig_s, rE,qE_s,sE)
end

treesum(s::Segment) = (segvalue(s), quaderror(s))
treesum(n::HeapNode) = n.w .* reduce((a,b) -> a .+ b, treesum.(n.h.valtree))
treesum(s::HeapSegment) = reduce((a,b) -> a .+ b, treesum.(s.h.valtree))


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
    do_iai(f, l, Val(initdiv), order, atol, rtol, maxevals, norm, segbuf)

function do_iai(f::F, l::L, ::Val{N}, n, atol, rtol, maxevals, nrm, buf) where {F,L,N}
    T = eltype(l); d = ndims(l)
    x,w,gw = cachedrule(T,n)
    p = 2n+1

    @assert N ≥ 1
    (numevals = N*p^d) <= maxevals || throw(ArgumentError("maxevals exceeded on initial evaluation"))
    segs = evalsegs(Val(d), f,l,Val(N), x,w,gw, nrm)
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
        return (I, E) # fast return when no subdivisions required
    end

    segheap = buf===nothing ? BinaryMaxHeap(collect(segs)) : (empty!(buf.valtree); push!(buf, segs...); buf)
    adapt!(segheap, Val(d), f,l, I, E, numevals, x,w,gw,p, atol_, rtol_, maxevals, nrm)
end

# internal routine to perform the h-adaptive refinement of the multi-dimensional integration segments
function adapt!(segheap::BinaryMaxHeap{T}, ::Val{d}, f::F, l::L, I, E, numevals, x,w,gw,p, atol, rtol, maxevals, nrm) where {T,d,F,L}
    # Pop the biggest-error segment (in any dimension) and subdivide (h-adaptation)
    # until convergence is achieved or maxevals is exceeded.
    while (tol = max(atol, rtol * nrm(I))) < E
        if numevals > maxevals
            @warn "maxevals exceeded"
            break
        end

        s = pop!(segheap)
        I -= segvalue(s)
        E -= quaderror(s)

        # find the segment with the largest error by percolating down the tree
        branch = (); g = f; m = l; depth = d
        while sorterror(s) > ruleerror(s)
            n = pop!(s.h) # node in segment with largest error
            branch = ((n,s), branch...)
            # remove the contribution of the node to the segment
            s.qE -= ruleerror(s) + quaderror(n)
            s.Ik -= n.w * n.I
            s.Ig -= n.g * n.I
            s = pop!(n.h) # segment in node with largest error
            # remove the contribution of the segment to the node
            n.I -= segvalue(s)
            n.E -= quaderror(s)
            # pre-evaluate the integrand/limits of integration
            g = iterated_pre_eval(g, n.x, Val(depth))
            m = fixandeliminate(m, n.x)
            depth -= 1
        end

        mid = (s.a + s.b) / 2
        s1 = evalrule(Val(depth), g,m,Val(1), s.a,mid, x,w,gw, nrm)
        s2 = evalrule(Val(depth), g,m,Val(1), mid,s.b, x,w,gw, nrm)
        segs = (s1, s2)
        numevals += 2*p^depth

        # percolate up the tree to update the quadratures and heaps
        for (n,s) in branch
            # replace the removed segment(s)
            push!(n.h, segs...)
            n.I += sum(segvalue,  segs)
            n.E += sum(quaderror, segs)
            # replace the removed node
            push!(s.h, n)
            s.Ik += n.w * n.I
            s.Ig += n.g * n.I
            s.rE =  nrm(s.Ik - s.Ig)
            s.qE += ruleerror(s) + quaderror(n)
            s.sE =  max(ruleerror(s), sorterror(first(s.h)))
            segs = (s,)
        end

        push!(segheap, segs...)
        I += sum(segvalue,  segs)
        E += sum(quaderror, segs)
    end

    # re-sum (paranoia about accumulated roundoff)
    I, E = treesum(segheap.valtree[1])
    for i in 2:length(segheap)
        I_, E_ = treesum(segheap.valtree[i])
        I += I_
        E += E_
    end
    return (I, E)
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
