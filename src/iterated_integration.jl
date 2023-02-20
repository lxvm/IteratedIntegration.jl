struct Panel{TX,TI,TE<:Real,TN}
    a::TX
    b::TX
    I::TI
    E::TE
    n::TN
    Panel(a::TX,b::TX,I::TI,E::TE,n::TN) where {TX,TI,TE,TN} =
        new{TX,TI,TE,TN}(a,b,I,E,n)
end
Base.isless(i::Panel, j::Panel) = isless(i.E, j.E)


struct Node{TV,TX,TP}
    v::TV
    x::TX
    w::TX
    g::TX
    p::TP
    Node(v::TV,x::TX,w::TX,g::TX,p::TP) where {TV,TX,TP} =
        new{TV,TX,TP}(v,x,w,g,p)
end

evalnode(::Val{1}, f, _, xn, wn, gwn, _, _, _, _, ::Val) = Node(f(xn), xn, wn, gwn, nothing)
function evalnode(::Val{d}, f, l, xn, wn, gwn, x,w,gw,nrm, ::Val{N}) where {d,N}
    g = iterated_pre_eval(f, xn, Val(d))
    m = fixandeliminate(l, xn)
    s = iterated_segs(g, m, Val(N))
    segs = ntuple(n -> evalpanel(Val(d-1), g, m, s[n], s[n+1], x,w,gw, nrm, Val(N)), Val(N))
    Node(sum(s -> s.I, segs), xn, wn, gwn, collect(segs))
end

function evalpanel(f::F, l::L, ::Val{N}=Val(1); order=7, norm=norm) where {F,L,N}
    # this function is just for testing
    x,w,gw = cachedrule(eltype(L),order)
    a,b = endpoints(l)
    evalpanel(f, l,a,b, x,w,gw, norm, Val(N))
end
evalpanel(f::F, l::L, a, b, x,w,gw, nrm, ::Val{N}) where {F,L,N} =
    evalpanel(Val(ndims(l)), f, l,a,b, x,w,gw, nrm, Val(N))
function evalpanel(::Val{d}, f::F, l::L, a, b, x,w,gw, nrm, ::Val{N}) where {d,F,L,N}
    p = 2*length(x)-1
    T = Base.promote_op(evalnode, Val{d}, F, L, eltype(x),eltype(w),eltype(gw), typeof(x),typeof(w),typeof(gw),typeof(nrm), Val{N})
    node = Vector{T}(undef, p)
    # Ik and Ig are integrals via Kronrod and Gauss rules, respectively
    s = convert(eltype(x), 0.5) * (b-a)
    n1 = 1 - (length(x) & 1) # 0 if even order, 1 if odd order
    # unroll first iteration of loop to get correct type of Ik and Ig
    node[2] = fg1 = evalnode(Val(d), f, l, a + (1+x[2])*s, w[2] * s, gw[1] * s, x, w, gw, nrm, Val(N))
    # node[2] = fg1 = f(a + (1+x[2])*s, w[2], gw[1])
    node[p-1] = fg2 = evalnode(Val(d), f, l, a + (1-x[2])*s, w[2] * s, gw[1] * s, x, w, gw, nrm, Val(N))
    # node[p-1] = fg2 = f(a + (1-x[2])*s, w[2], gw[1])
    fg = fg1.v + fg2.v
    node[1] = fk1 = evalnode(Val(d), f, l, a + (1+x[1])*s, w[1] * s, zero(gw[1]), x, w, gw, nrm, Val(N))
    # node[1] = fk1 = f(a + (1+x[1])*s, w[1], zero(gw[1]))
    node[p] = fk2 = evalnode(Val(d), f, l, a + (1-x[1])*s, w[1] * s, zero(gw[1]), x, w, gw, nrm, Val(N))
    # node[p] = fk2 = f(a + (1-x[1])*s, w[1], zero(gw[1]))
    fk = fk1.v + fk2.v
    Ig = fg * gw[1]
    Ik = fg * w[2] + fk * w[1]
    for i = 2:length(gw)-n1
        node[2i] = fg1 = evalnode(Val(d), f, l, a + (1+x[2i])*s, w[2i] * s, gw[i] * s, x, w, gw, nrm, Val(N))
        # node[2i] = fg1 = f(a + (1+x[2i])*s, w[2i], gw[i])
        node[p-2i+1] = fg2 = evalnode(Val(d), f, l, a + (1-x[2i])*s, w[2i] * s, gw[i] * s, x, w, gw, nrm, Val(N))
        # node[p-2i+1] = fg2 = f(a + (1-x[2i])*s, w[2i], gw[i])
        fg = fg1.v + fg2.v
        node[2i-1] = fk1 = evalnode(Val(d), f, l, a + (1+x[2i-1])*s, w[2i-1] * s, zero(gw[i]), x, w, gw, nrm, Val(N))
        # node[2i-1] = fk1 = f(a + (1+x[2i-1])*s, w[2i-1], zero(gw[i]))
        node[p-2i+2] = fk2 = evalnode(Val(d), f, l, a + (1-x[2i-1])*s, w[2i-1] * s, zero(gw[i]), x, w, gw, nrm, Val(N))
        # node[p-2i+2] = fk2 = f(a + (1-x[2i-1])*s, w[2i-1], zero(gw[i]))
        fk = fk1.v + fk2.v
        Ig += fg * gw[i]
        Ik += fg * w[2i] + fk * w[2i-1]
    end
    c = div(p,2)
    if n1 == 0 # even: Gauss rule does not include x == 0
        node[c+1] = fk0 = evalnode(Val(d), f, l, a + s, w[end] * s, zero(w[end]), x, w, gw, nrm, Val(N))
        # node[c+1] = fk0 = f(a + s, w[end], zero(w[end]))
        fk = fk0.v
        Ik += fk * w[end]
    else # odd: don't count x==0 twice in Gauss rule
        node[c+1] = fg0 = evalnode(Val(d), f, l, a + s, w[end] * s, gw[end] * s, x, w, gw, nrm, Val(N))
        # node[c+1] = fg0 = f(a + s, w[end], gw[end])
        fg = fg0.v
        Ig += fg * gw[end]
        node[c] = fk1 = evalnode(Val(d), f, l, a + (1+x[end-1])*s, w[end-1] * s, zero(gw[end]), x, w, gw, nrm, Val(N))
        # node[c] = fk1 = f(a + (1+x[end-1])*s, w[end-1], zero(gw[end]))
        node[c+2] = fk2 = evalnode(Val(d), f, l, a + (1-x[end-1])*s, w[end-1] * s, zero(gw[end]), x, w, gw, nrm, Val(N))
        # node[c+2] = fk2 = f(a + (1-x[end-1])*s, w[end-1], zero(gw[end]))
        fk = fk1.v + fk2.v
        Ik += fg * w[end] + fk * w[end-1]
    end
    Ik_s, Ig_s = Ik * s, Ig * s # new variable since this may change the type
    E = nrm(Ik_s - Ig_s)
    if isnan(E) || isinf(E)
        throw(DomainError(a+s, "integrand produced $E in the interval ($a, $b)"))
    end
    return Panel(oftype(s, a), oftype(s, b), Ik_s, E, node)
end

# get the max error at all levels of tree
function tree_err(seg::Panel)
    E = seg.E
    if seg.n isa Vector{<:Node{<:Any,<:Any,Nothing}}
        (E,)
    else
        (p0, rest) = Iterators.peel(seg.p)
        err = nderr(p0)
        for p in rest
            err = map(max, err, nderr(p))
        end
        (err..., seg.E)
    end
end

function nodedump(n::Node)
    if isnothing(n.p)
        (f=[n.v], x=[(n.x,)], w=[n.w], gw=[n.g])
    else
        data = map(paneldump, n.p) # grouped by panel
        (;
        f = vcat(map(d -> d.f, data)...),
        x = vcat(map(d -> map(xs -> (xs..., n.x), d.x), data)...),
        w = vcat(map(d -> n.w * d.w, data)...),
        gw = vcat(map(d -> n.g * d.gw, data)...),
        )
    end
end

function paneldump(p::Panel)
    data = map(nodedump, p.n)
    map(s -> vcat(map(d -> getfield(d, s), data)...), (f=:f, x=:x, w=:w, gw=:gw))
end
function paneldump(p::Vector{<:Panel})
    data = map(paneldump, p)
    map(s -> vcat(map(d -> getfield(d, s), data)...), (f=:f, x=:x, w=:w, gw=:gw))
end

# unlike nested_quadgk, this routine won't support iterated_integrand since that
# could change the type of the result and mess up the error estimation
"""
    iterated_integration(f, a, b; kwargs...)
    iterated_integration(f::AbstractIteratedIntegrand{d}, l::AbstractLimits{d}; order=7, atol=0, rtol=sqrt(eps()), norm=norm, maxevals=10^7, segbuf=nothing) where d
"""
function iterated_integration(f, a, b; kwargs...)
    l = CubicLimits(a, b)
    iterated_integration(ThunkIntegrand{ndims(l)}(f), l; kwargs...)
end
iterated_integration(f::F, l::L; order=7, atol=nothing, rtol=nothing, norm=norm, maxevals=10^7, initdiv=1, segbuf=nothing) where {F,L} =
    do_iai(f, l, order, Val(initdiv), atol, rtol, maxevals, norm, segbuf)

function do_iai(f::F, l::L, n, ::Val{N}, atol, rtol, maxevals, nrm, segbuf) where {F,L,N}
    T = eltype(l); d = ndims(l)
    x,w,gw = cachedrule(T,n)
    
    @assert N ≥ 1
    (numevals = N*(2n+1)^d) <= maxevals || throw(ArgumentError("maxevals exceeded on initial evaluation"))
    s = iterated_segs(f, l, Val(N))
    segs = ntuple(n -> evalpanel(Val(d), f, l, s[n], s[n+1], x,w,gw, nrm, Val(N)), Val(N))
    I = sum(s -> s.I, segs)
    E = sum(s -> s.E, segs)
    # This error estimate is for the outer integral
    # It might help to be more rigorous and check that all the inner integrals
    # are converged as well, e.g. using nderr

    # logic here is mainly to handle dimensionful quantities: we
    # don't know the correct type of atol, in particular, until
    # this point where we have the type of E from f. Also, follow
    # Base.isapprox in that if atol≠0 is supplied by the user, rtol
    # defaults to zero.
    atol_ = something(atol, zero(E))
    rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(one(eltype(x)))) : zero(eltype(x)))

    # optimize common case of no subdivision
    if E ≤ atol_ || E ≤ rtol_ * nrm(I) || numevals ≥ maxevals
        return (I, E, numevals) # fast return when no subdivisions required
    end

    segheap = segbuf === nothing ? collect(segs) : (resize!(segbuf, N) .= segs)
    heapify!(segheap, Reverse)
    @show adaptpanel!(segheap, Val(d), f, l, I, E, numevals, x,w,gw,n, atol_, rtol_, maxevals, nrm, Val(N))
    return segheap
end

# internal routine to perform the h-adaptive refinement of the multi-dimensional integration segments (segs)
# for now the goal is to replicate nested_quadgk, but modifications such as
# breadth-first search would be interesting
function adaptpanel!(segs::Vector{T}, ::Val{d}, f::F, l::L, I, E, numevals, x,w,gw,n, atol, rtol, maxevals, nrm, ::Val{N}) where {T,d,F,L,N}
    # Pop the biggest-error segment and subdivide (h-adaptation)
    # until convergence is achieved or maxevals is exceeded.
    while (tol = max(atol, rtol * nrm(I))) < E
        if numevals > maxevals
            @warn "maxevals exceeded"
            break
        end

        s = heappop!(segs, Reverse)

        if d > 1
            # adapt in the inner integral
            sI, sE, numevals = resolvepanel!(s, Val(d), f, l, s.I, s.E, numevals, x,w,gw,n, atol, rtol, maxevals, nrm, Val(N))
            E = (E - s.E) + sE
            I = (I - s.I) + sI
            
            heappush!(segs, Panel(s.a, s.b, sI, sE, s.n), Reverse)
            s = heappop!(segs, Reverse)
            tol = max(atol, rtol * nrm(I))
        end

        if d == 1 || E > tol
            # adapt in the current integral
            mid = (s.a + s.b) / 2
            s1 = evalpanel(Val(d), f, l, s.a, mid, x,w,gw, nrm, Val(N))
            s2 = evalpanel(Val(d), f, l, mid, s.b, x,w,gw, nrm, Val(N))
            I = (I - s.I) + s1.I + s2.I
            E = (E - s.E) + s1.E + s2.E
            numevals += 2*(2n+1)^d

            heappush!(segs, s1, Reverse)
            heappush!(segs, s2, Reverse)
        elseif d > 1
            heappush!(segs, s, Reverse)
        end
    end

    # re-sum (paranoia about accumulated roundoff)
    I = segs[1].I
    E = segs[1].E
    for i in 2:length(segs)
        I += segs[i].I
        E += segs[i].E
    end
    return (I, E, numevals)
end

# internal routine to raise the accuracy of the nodes in a panel
function resolvepanel!(p::Panel, ::Val{d}, f::F, l::L, I, E, numevals, x,w,gw,n, atol, rtol, maxevals, nrm, ::Val{N}) where {d,F,L,N}
    for (i,node) in enumerate(p.n)
        if true # E > max(atol, rtol*nrm(I)) # nested quadgk always goes
            g = iterated_pre_eval(f, node.x, Val(d))
            m = fixandeliminate(l, node.x)
            heapify!(node.p, Reverse)
            pI, pE, numevals = adaptpanel!(node.p, Val(d-1), g, m, node.v, sum(n -> n.E, node.p), numevals, x,w,gw,n, atol, rtol, maxevals, nrm, Val(N))
            # I = (I - node.w*node.v) + node.w*pI
            # E = (E - sum(p -> p.E, node.p)) + pE
            p.n[i] = Node(pI, node.x, node.w, node.g, node.p)
            if numevals > maxevals
                @warn "maxevals exceeded"
                break
            end
        end
    end
    # compute Ik, Ig for this panel
    Ik = p.n[1].w * p.n[1].v
    Ig = p.n[1].g * p.n[1].v
    for i in 2:length(p.n)
        Ik += p.n[i].w * p.n[i].v
        Ig += p.n[i].g * p.n[i].v
    end
    E = nrm(Ik - Ig)
    return (Ik, E, numevals)
end
