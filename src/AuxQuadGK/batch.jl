
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
