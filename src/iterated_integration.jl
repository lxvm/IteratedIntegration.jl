
struct Node{T,TX}
    v::T
    x::TX
    w::TX
    g::TX
end
mutable struct HeapSegment{T,TX,TI,TE}
    h::BinaryMaxHeap{Node{T,TX}}
    a::TX
    b::TX
    Ik::TI
    Ig::TI
    E::TE
    Emax::TE
end
mutable struct HeapTree{T,TX,TI,TE}
    h::BinaryMaxHeap{HeapSegment{T,TX,TI,TE}}
    I::TI
    E::TE
    Emax::TE
end


AbstractTrees.children(t::HeapTree) = Iterators.flatten(map(children, t.h.valtree))
AbstractTrees.nodevalue(t::HeapTree) = t.I
quaderror(t::HeapTree) = t.E
sorterror(t::HeapTree) = t.Emax

AbstractTrees.children(t::HeapSegment) = t.h.valtree
AbstractTrees.nodevalue(t::HeapSegment) = t.Ik

quaderror(s::HeapSegment) = s.E
sorterror(s::HeapSegment) = s.Emax

AbstractTrees.nodevalue(t::Segment) = t.I

quaderror(s::Segment) = s.E
sorterror(s::Segment) = s.E

AbstractTrees.children(n::Node{<:HeapTree}) = children(n.v)
AbstractTrees.children(n::Node) = (n,)
AbstractTrees.nodevalue(n::Node{<:HeapTree}) = nodevalue(n.v)
quaderror(n::Node{<:HeapTree}) = quaderror(n.v)
sorterror(n::Node{<:HeapTree}) = sorterror(n.v) * n.w

AbstractTrees.nodevalue(n::Node) = n.v
quaderror(n::Node) = zero(n.x)
sorterror(n::Node) = zero(n.x)

Base.isless(i::Node, j::Node) = isless(sorterror(i), sorterror(j))
Base.isless(i::HeapSegment, j::HeapSegment) = isless(sorterror(i), sorterror(j))

function treepop!(t::HeapTree)
    s = first(t.h) # segment
    t.I -= nodevalue(s)
    t.E -= quaderror(s)
    if sorterror(s) > norm(s.Ik - s.Ig)
        n = first(s.h) # node
        s.E  -= norm(s.Ik - s.Ig) + n.w * quaderror(n)
        s.Ik -= n.w * nodevalue(n)
        s.Ig -= n.g * nodevalue(n)
        s2, nodelist = treepop!(n.v)
        s2, (nodelist..., n)
    else
        pop!(t.h)
        t.Emax = length(t.h) == 0 ? zero(t.Emax) : sorterror(first(t.h))
        s, ()
    end
end

function treepush!(t, nodelist, s...)
    treepush!(t, nodelist, s)
    return (nodevalue(t), quaderror(t))
end
function treepush!(t::HeapTree, ::Tuple{}, segs::NTuple)
    t.I += sum(nodevalue, segs)
    t.E += sum(quaderror, segs)
    t.Emax = max(maximum(sorterror, segs), t.Emax)
    foreach(s -> push!(t.h, s), segs)
    t
end
function treepush!(t::HeapTree, nodelist::Tuple, segs::NTuple)
    s = pop!(t.h) # segment
    n = pop!(s.h) # node
    treepush!(n.v, Base.front(nodelist), segs)
    s.Ik += n.w * nodevalue(n)
    s.Ig += n.g * nodevalue(n)
    s.E  += norm(s.Ik - s.Ig) + n.w * quaderror(n)
    t.I += nodevalue(s)
    t.E += quaderror(s)
    
    push!(s.h, n)
    s.Emax = max(norm(s.Ik - s.Ig), sorterror(first(s.h)))
    push!(t.h, s)
    t.Emax = max(t.Emax, sorterror(first(t.h)))
    t
end

function evalnode(::Val{1}, f, l, xn, wn, gwn, x,w,gw,nrm)
    Node(f(xn), xn, wn, gwn)
end

function evalnode(::Val{d}, f, l, xn, wn, gwn, x,w,gw,nrm) where d
    g = iterated_pre_eval(f, xn, Val(d))
    m = fixandeliminate(l, xn)
    a, b = endpoints(m)
    s = evalsegment(Val(d-1), g, m,a,b, x,w,gw, nrm)
    segheap = BinaryMaxHeap{typeof(s)}()
    push!(segheap, s)
    tree = HeapTree(segheap, nodevalue(s), quaderror(s), sorterror(s))
    Node(tree, xn, wn, gwn)
end

function evalsegment(f::F, l::L; order=7, norm=norm) where {F,L}
    # this function is just for testing
    x,w,gw = cachedrule(eltype(L),order)
    a, b = endpoints(l)
    evalsegment(Val(ndims(l)), f, l,a,b, x,w,gw, norm)
end
# function evalsegment(::Val{1}, f::F, l::L, a, b, x,w,gw, nrm) where {F,L}
#     evalrule(f, a,b, x,w,gw, nrm) # default to quadgk segments for leaf panels
# end

function evalsegment(::Val{d}, f::F, l::L, a, b, x,w,gw, nrm) where {d,F,L}
    p = 2*length(x)-1 # number of points in Kronrod rule

    s = convert(eltype(x), 0.5) * (b-a)
    n1 = 1 - (length(x) & 1) # 0 if even order, 1 if odd order

    f0 = evalnode(Val(d), f, l, a + s, w[end] * s, gw[end] * s, x, w, gw, nrm)
    nodeheap = BinaryMaxHeap{typeof(f0)}()
    sizehint!(nodeheap, p)

    if n1 == 0 # even: Gauss rule does not include x == 0
        push!(nodeheap, Node(f0.v, f0.x, f0.w, zero(f0.g)))

        Ik = nodevalue(f0) * w[end]
        Ig = zero(Ik)
        qE = quaderror(f0) * w[end]

    else # odd: don't count x==0 twice in Gauss rule
        fk1 = evalnode(Val(d), f, l, a + (1+x[end-1])*s, w[end-1] * s, zero(gw[end]), x, w, gw, nrm)
        fk2 = evalnode(Val(d), f, l, a + (1-x[end-1])*s, w[end-1] * s, zero(gw[end]), x, w, gw, nrm)
        push!(nodeheap, f0, fk1, fk2)

        Ig = nodevalue(f0) * gw[end]
        Ik = nodevalue(f0) * w[end] + (nodevalue(fk1) + nodevalue(fk2)) * w[end-1]
        qE = quaderror(f0) * w[end] + (quaderror(fk1) + quaderror(fk2)) * w[end-1]
    end

    # Ik and Ig are integrals via Kronrod and Gauss rules, respectively
    # qE is the quadrature of the error
    
    for i = 1:length(gw)-n1
        fg1 = evalnode(Val(d), f, l, a + (1+x[2i])*s, w[2i] * s, gw[i] * s, x, w, gw, nrm)
        fg2 = evalnode(Val(d), f, l, a + (1-x[2i])*s, w[2i] * s, gw[i] * s, x, w, gw, nrm)
        fk1 = evalnode(Val(d), f, l, a + (1+x[2i-1])*s, w[2i-1] * s, zero(gw[i]), x, w, gw, nrm)
        fk2 = evalnode(Val(d), f, l, a + (1-x[2i-1])*s, w[2i-1] * s, zero(gw[i]), x, w, gw, nrm)
        push!(nodeheap, fg1, fg2, fk1, fk2)

        fg = nodevalue(fg1) + nodevalue(fg2)
        fk = nodevalue(fk1) + nodevalue(fk2)
        Eg = quaderror(fg1) + quaderror(fg2)
        Ek = quaderror(fk1) + quaderror(fk2)

        Ig += fg * gw[i]
        Ik += fg * w[2i] + fk * w[2i-1]
        qE += Eg * w[2i] + Ek * w[2i-1]
    end
    
    Ik_s, Ig_s = Ik * s, Ig * s # new variable since this may change the type
    E0 = nrm(Ik_s - Ig_s)
    E = E0 + qE * s
    if isnan(E) || isinf(E)
        throw(DomainError(a+s, "integrand produced $E in the interval ($a, $b)"))
    end
    Emax = max(E0, sorterror(first(nodeheap)))

    return HeapSegment(nodeheap, oftype(s, a), oftype(s, b), Ik_s, Ig_s, E, Emax)
end

pre_evaluate(::Val, f, l, ::Tuple{}) = f, l
function pre_evaluate(::Val{d}, f, l, xs::NTuple{N}) where {d,N}
    pre_evaluate(Val(d-1), iterated_pre_eval(f, xs[N], Val(d)), fixandeliminate(l, xs[N]), Base.front(xs))
end

# unlike nested_quadgk, this routine won't support iterated_integrand since that
# could change the type of the result and mess up the error estimation
"""
    iai(f, a, b; kwargs...)
    iai(f::AbstractIteratedIntegrand{d}, l::AbstractLimits{d}; order=7, atol=0, rtol=sqrt(eps()), norm=norm, maxevals=10^7, segbuf=nothing) where d
"""
function iai(f, a, b; kwargs...)
    l = CubicLimits(a, b)
    iai(ThunkIntegrand{ndims(l)}(f), l; kwargs...)
end
iai(f::F, l::L; order=7, atol=nothing, rtol=nothing, norm=norm, maxevals=typemax(Int), initdiv=1, segbuf=nothing) where {F,L} =
    do_iai(f, l, order, Val(initdiv), atol, rtol, maxevals, norm, segbuf)

function do_iai(f::F, l::L, n, ::Val{N}, atol, rtol, maxevals, nrm, segbuf) where {F,L,N}
    T = eltype(l); d = ndims(l)
    x,w,gw = cachedrule(T,n)
    
    @assert N ≥ 1
    (numevals = N*(2n+1)^d) <= maxevals || throw(ArgumentError("maxevals exceeded on initial evaluation"))
    s = iterated_segs(f, l, Val(N))
    segs = ntuple(i -> evalsegment(Val(d), f, l,s[i],s[i+1], x,w,gw, nrm), Val(N))
    I = sum(s -> nodevalue(s), segs)
    E = sum(s -> quaderror(s), segs)

    # logic here is mainly to handle dimensionful quantities: we
    # don't know the correct type of atol, in particular, until
    # this point where we have the type of E from f. Also, follow
    # Base.isapprox in that if atol≠0 is supplied by the user, rtol
    # defaults to zero.
    atol_ = something(atol, zero(E))
    rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(one(eltype(x)))) : zero(eltype(x)))

    # optimize common case of no subdivision
    # if E ≤ atol_ || E ≤ rtol_ * nrm(I) || numevals ≥ maxevals
    #     return (I, E) # fast return when no subdivisions required
    # end
    
    tree = if segbuf===nothing
        segheap = BinaryMaxHeap(collect(segs))
        HeapTree(segheap, I, E, sorterror(first(segheap)))
    else
        empty!(segbuf.h.valtree)
        push!(segbuf.h, segs...)
        segbuf.I = I; segbuf.E = E; segbuf.Emax = sorterror(first(segbuf.h))
        segbuf
    end
    adapt!(tree, Val(d), f, l, I, E, numevals, x,w,gw,n, atol_, rtol_, maxevals, nrm)
end

# internal routine to perform the h-adaptive refinement of the multi-dimensional integration segments
function adapt!(tree::T, ::Val{d}, f::F, l::L, I, E, numevals, x,w,gw,n, atol, rtol, maxevals, nrm) where {T,d,F,L}
    # Pop the biggest-error segment (in any dimension) and subdivide (h-adaptation)
    # until convergence is achieved or maxevals is exceeded.
    while max(atol, rtol * nrm(I)) < E
        if numevals > maxevals
            @warn "maxevals exceeded"
            break
        end

        s, nodelist = treepop!(tree)
        depth = d - length(nodelist)
        g, m = pre_evaluate(Val(d), f, l, map(n -> n.x, nodelist))
        mid = (s.a + s.b) / 2
        s1 = evalsegment(Val(depth), g, m, s.a, mid, x,w,gw, nrm)
        s2 = evalsegment(Val(depth), g, m, mid, s.b, x,w,gw, nrm)
        numevals += 2*(2n+1)^depth
        # @show depth I

        I, E = treepush!(tree, nodelist, s1, s2)
    end
    # re-sum (paranoia about accumulated roundoff)
    # I = zero(I)
    # E = zero(E)
    # for leaf in Leaves(tree) # TODO, depth-first summation accumulating the node weights
    #     I += nodevalue(leaf)
    #     E += quaderror(leaf)
    # end
    return (I, E)
end

function iai_count(f, a, b; kwargs...)
    numevals = 0
    I, E = iai(a, b; kwargs...) do x
        numevals += 1
        f(x)
    end
    return (I, E, numevals)
end

iai_print(io::IO, f, args...; kws...) = iai_count(args...; kws...) do x
    y = f(x)
    println(io, "f(", x, ") = ", y)
    y
end
iai_print(f, args...; kws...) = iai_print(stdout, f, args...; kws...)

function alloc_tree(DT, RT, NT, ndims)
    T = alloc_tree_(DT, RT, NT, Val(ndims))
    HeapTree(BinaryMaxHeap{HeapSegment{T,DT,RT,NT}}(), zero(RT), zero(NT), zero(NT))
end
function alloc_tree_(DT, RT, NT,::Val{1})
    RT
end
function alloc_tree_(DT, RT, NT,::Val{d}) where d
    HeapTree{alloc_tree_(DT,RT,NT, Val(d-1)),DT,RT,NT}
end

griddump(n::Node) = n.x
griddump(n::Node{<:HeapTree}) = griddump(n.v)
function griddump(t::HeapTree)
    [n.x for n in children(t)]
end
function griddump(t::HeapTree{<:HeapTree})
    vcat([map(x -> (x..., n.x), griddump(n)) for n in children(t)]...)
end