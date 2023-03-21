struct Node2d{TX,TI,TE}
    s::Segment{TX,TI,TE}
    E::TE
    i::Int
end
Base.isless(i::Node2d, j::Node2d) = isless(i.E, j.E)

mutable struct Segment2d{TX,TI,TE}
    h::BinaryMaxHeap{Node2d{TX,TI,TE}}
    a::TX
    b::TX
    Ik::TI
    Ig::TI
    E::TE
    Eo::TE
    Ek::TE
    Eg::TE
    # the rule on this panel
    x::Vector{TX}
    w::Vector{TX}
    gw::Vector{TX}
    # the estimates and errors of the 1d quadratures at all nodes
    hI::Vector{TI}
    hE::Vector{TE}
end

# segvalue(s::Segment) = s.I
segvalue(s::Segment2d) = s.Ik

# distinguish between the GK rule error and the multidimensional quadrature error
# ruleerror(s::Segment) = s.E
# quaderror(s::Segment) = s.E

ruleerror(s::Segment2d) = s.Eo
quaderror(s::Segment2d) = s.E

Base.isless(i::Segment2d, j::Segment2d) = isless(quaderror(i), quaderror(j))

# default to quadgk segments for leaf panels
function evalnode2d(f::F,l::L,::Val{N}, xn, x,w,gw,nrm) where {F,L,N}
    g = iterated_pre_eval(f, xn, Val(2))
    m = fixandeliminate(l, xn)
    s = iterated_segs(g, m, Val(N))
    ntuple(i -> evalrule(g,s[i],s[i+1], x,w,gw, nrm), Val(N))
end

function evalrule2d(f::F,l::L,::Val{N}, a,b, x,w,gw, nrm) where {F,L,N}
    p = 2*length(x)-1 # number of points in Kronrod rule
    c = div(p,2)

    s = convert(eltype(x), 0.5) * (b-a)

    # deflated copy of the quadrature rules for convenience
    xs = Vector{eltype(x)}(undef, p)
    ws = Vector{eltype(w)}(undef, p)
    gws = Vector{eltype(gw)}(undef, p)

    n1 = 1 - (length(x) & 1) # 0 if even order, 1 if odd order

    xs[c+1] = a + s; ws[c+1] = w[end] * s; gws[c+1] = gw[end] * s
    f0 = evalnode2d(f,l,Val(N), xs[c+1], x, w, gw, nrm)
    # segheap = BinaryMaxHeap(collect(map(v -> Node2d(v, quaderror(v), c+1), f0)))
    segheap = BinaryMaxHeap(collect(map(v -> Node2d(v, ws[c+1]*quaderror(v), c+1), f0)))
    sizehint!(segheap, N*p)

    hI = fill(sum(segvalue,  f0), p)
    hE = fill(sum(quaderror, f0), p)

    if n1 == 0 # even: Gauss rule does not include x == 0
        gws[c+1] = zero(eltype(gw))
    else # odd: don't count x==0 twice in Gauss rule
        # j = 2i; k = p-j+1
        xs[c] = a + (1+x[end-1])*s; ws[c] = w[end-1] * s; gws[c] = zero(eltype(gw))
        f1 = evalnode2d(f,l,Val(N), xs[c], x, w, gw, nrm)
        # foreach(v -> push!(segheap, Node2d(v, quaderror(v), c)), f1)
        foreach(v -> push!(segheap, Node2d(v, ws[c]*quaderror(v), c)), f1)
        hI[c] = sum(segvalue,  f1)
        hE[c] = sum(quaderror, f1)
        
        # j = 2i; k = p-j+1
        xs[c+2] = a + (1-x[end-1])*s; ws[c+2] = ws[c]; gws[c+2] = gws[c]
        f2 = evalnode2d(f,l,Val(N), xs[c+2], x, w, gw, nrm)
        # foreach(v -> push!(segheap, Node2d(v, quaderror(v), c+2)), f2)
        foreach(v -> push!(segheap, Node2d(v, ws[c+2]*quaderror(v), c+2)), f2)
        hI[c+2] = sum(segvalue,  f2)
        hE[c+2] = sum(quaderror, f2)
    end

    for i = 1:length(gw)-n1
        jk = (2i, p-2i+1)
        for (g, (j, k)) in ((gw[i] * s, jk), (zero(eltype(gw)), (jk[1]-1, jk[2]+1)))
            xs[j] = a + (1+x[j])*s; ws[j] = w[j] * s; gws[j] = g
            f1 = evalnode2d(f,l,Val(N), xs[j], x, w, gw, nrm)
            # foreach(v -> push!(segheap, Node2d(v, quaderror(v), j)), f1)
            foreach(v -> push!(segheap, Node2d(v, ws[j]*quaderror(v), j)), f1)
            hI[j] = sum(segvalue,  f1)
            hE[j] = sum(quaderror, f1)

            xs[k] = a + (1-x[j])*s; ws[k] = ws[j]; gws[k] = gws[j]
            f2 = evalnode2d(f,l,Val(N), xs[k], x, w, gw, nrm)
            # foreach(v -> push!(segheap, Node2d(v, quaderror(v), k)), f2)
            foreach(v -> push!(segheap, Node2d(v, ws[k]*quaderror(v), k)), f2)
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

    return Segment2d(segheap, oftype(s,a),oftype(s,b), Ik,Ig, E,Eo,Ek,Eg, xs,ws,gws, hI,hE)
end

function evalsegs2d(f::F,l::L,::Val{N}, x,w,gw, nrm) where {F,L,N}
    s = iterated_segs(f, l, Val(N))
    ntuple(i -> evalrule2d(f,l,Val(N),s[i],s[i+1], x,w,gw, nrm), Val(N))
end

# these functions just for testing
function evalsegs2d(f, a, b; kwargs...)
    l = CubicLimits(a,b)
    evalsegs2d(ThunkIntegrand{ndims(l)}(f), l; kwargs...)
end
function evalsegs2d(f::F, l::L; initdiv=1, order=7, norm=norm) where {F,L}
    x,w,gw = cachedrule(eltype(L),order)
    evalsegs2d(f,l,Val(initdiv), x,w,gw, norm)
end


function leafsum2d(s::Segment2d{TX,TI,TE}) where {TX,TI,TE}
    I = sum(segvalue∘first,  s.h.valtree; init=zero(TI))
    E = sum(quaderror∘first, s.h.valtree; init=zero(TE))
    return (I, E)
end

# unlike nested_quadgk, this routine won't support iterated_integrand since that
# could change the type of the result and mess up the error estimation
"""
    iai2d(f, a, b; kwargs...)
    iai2d(f::AbstractIteratedIntegrand{2}, l::AbstractLimits{2}; order=7, atol=0, rtol=sqrt(eps()), norm=norm, maxevals=typemax(Int), segbuf=nothing)
    
Two-dimensional globally-adaptive quadrature via iterated integration using
Gauss-Kronrod rules. Interface is similar to [`nested_quadgk`](@ref).
"""
function iai2d(f, a, b; kwargs...)
    l = CubicLimits(a, b)
    iai2d(ThunkIntegrand{ndims(l)}(f), l; kwargs...)
end
iai2d(f::F, l::L; order=7, atol=nothing, rtol=nothing, norm=norm, maxevals=typemax(Int), initdiv=1, segbuf=nothing) where {F,L} =
    do_iai2d(f, l, Val(initdiv), order, atol, rtol, maxevals, norm, segbuf)

function do_iai2d(f::F, l::L, ::Val{N}, n, atol, rtol, maxevals, nrm, buf) where {F,L,N}
    T = eltype(l)
    x,w,gw = cachedrule(T,n)
    p = 2n+1
    
    @assert N ≥ 1
    (numevals = (N*p)^2) <= maxevals || throw(ArgumentError("maxevals exceeded on initial evaluation"))
    segs = evalsegs2d(f,l,Val(N), x,w,gw, nrm)
    I = sum(segvalue, segs)
    E = sum(quaderror, segs)

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

    segheap = buf===nothing ? BinaryMaxHeap(collect(segs)) : (empty!(buf.valtree); push!(buf, segs...); buf)
    adapt2d!(segheap, f,l, I, E, numevals, x,w,gw,p, atol_, rtol_, maxevals, nrm)
end

# internal routine to perform the h-adaptive refinement of the multi-dimensional integration segments
function adapt2d!(segheap::BinaryMaxHeap{Segment2d{TX,TI,TE}}, f::F, l::L, I, E, numevals, x,w,gw,p, atol, rtol, maxevals, nrm) where {TX,TI,TE,F,L}
    # Pop the biggest-error segment (in any dimension) and subdivide (h-adaptation)
    # until convergence is achieved or maxevals is exceeded.
    refineinner = true

    while (tol = max(atol, rtol * nrm(I))) < E
        if numevals > maxevals
            @warn "maxevals exceeded"
            break
        end
        
        s = pop!(segheap)

        # refine inner panel till we enter the convergence regime
        while refineinner || any(max.(atol, rtol*nrm.(s.hI)) .< s.hE) #|| (s.Eo <= s.Ek && max(atol, rtol * nrm(I)) < E) # && ((s.Ek >= s.Eo && s.E >= E2) || qE <= abs(s.Ek-s.Eg))
            if numevals > maxevals
                @warn "maxevals exceeded"
                break
            end
            # println(s.E, "\t", s.Eo, "\t", s.Ek,"\t", nrm(s.Ik), "\t", numevals)
            refineinner = false
            
            I -= segvalue(s)
            E -= quaderror(s)

            n = pop!(s.h)
            ss = n.s
            i  = n.i
            
            xn = s.x[i]
            wn = s.w[i]
            gwn = s.gw[i]

            s.Ik -= wn * s.hI[i]
            s.Ig -= gwn * s.hI[i]
            s.hI[i] -= segvalue(ss)

            s.Ek -= wn * s.hE[i]
            s.Eg -= gwn * s.hE[i]
            s.hE[i] -= quaderror(ss)

            g = iterated_pre_eval(f, xn, Val(2))

            mid = (ss.a + ss.b) / 2
            ss1 = evalrule(g, ss.a,mid, x,w,gw, nrm)
            ss2 = evalrule(g, mid,ss.b, x,w,gw, nrm)
            numevals += 2*p

            # push!(s.h, Node2d(ss1, quaderror(ss1), i), Node2d(ss2, quaderror(ss2), i))
            push!(s.h, Node2d(ss1, wn*quaderror(ss1), i), Node2d(ss2, wn*quaderror(ss2), i))

            s.hI[i] += segvalue(ss1) + segvalue(ss2)
            s.Ik += wn * s.hI[i]
            s.Ig += gwn * s.hI[i]

            s.hE[i] += quaderror(ss1) + quaderror(ss2)
            s.Ek += wn * s.hE[i]
            s.Eg += gwn * s.hE[i]

            #=
            whenever s.Eo -nrm(s.Ik - s.Ig) ~ wn * nrm(s.h[i]) is a sign that we
            updated a Kronrod point with a much larger value that may cause
            outer thrashing. we could force the algorithm to continue refining
            until the new Eo goes below the original or the nex
            =#
            s.Eo = nrm(s.Ik - s.Ig)
            s.E = s.Eo + s.Ek
            # @show s.Eo s.Ek s.Eg
            I += segvalue(s)
            E += quaderror(s)
        end

        if s.Eo > s.Ek
            # refine outer panel
            I -= segvalue(s)
            E -= quaderror(s)

            mid2d = (s.a + s.b) / 2
            s1 = evalrule2d(f,l,Val(1), s.a,mid2d, x,w,gw, nrm)
            s2 = evalrule2d(f,l,Val(1), mid2d,s.b, x,w,gw, nrm)
            numevals += 2*p^2

            I += segvalue(s1) + segvalue(s2)
            E += quaderror(s1) + quaderror(s2)

            push!(segheap, s1, s2)
        else
            push!(segheap, s)
        end
        refineinner = true
    end

    # re-sum (paranoia about accumulated roundoff) TODO
    return I, E
end

"""
    iai2d_buffer(DT, RT, NT)

Allocate a buffer for use by [`iai2d`](@ref) for an integrand with with domain
type `DT`, range type `RT` and norm type `NT`
"""
iai2d_buffer(DT, RT, NT) = BinaryMaxHeap{Segment2d{DT, RT, NT}}()
