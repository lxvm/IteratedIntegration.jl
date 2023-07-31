
function applypolesub!(gvals::AbstractArray, ginvals::AbstractArray, a::Number,
                      b::Number, x,w,gw,n, rho, fac, rootmeth, nrm, maxpolesubint=Inf)
# pole-correcting version of applygkrule. Changes the input ginvals array.
# no local g evals; just pass in all vals and reciprocal vals.
# Barnett 6/30/23
    @assert length(gvals)==length(ginvals)
    s = evalrule(ginvals, a,b, x,w,gw, nrm) # create Segment w/ plain GK ans for (a,b)
    if b-a > maxpolesubint           # save a few pole-sub considerations? no
        return s
    end
    # now work in local coords wrt std seg [-1,1]...
    # get roots, g'(roots)  ...n seems good max # roots to pole-sub @ 2n+1 pts
    zr, dgdt = find_near_roots(gvals,x,rho,fac,rootmeth)
    if length(zr)==0 || length(zr)>4  #length(r.gw)  # or 3, captures most
        return s        # either nothing to do, or don't pole-sub too much!
    end
    Ipoles = zero(s.I)
    for (i,z) in enumerate(zr)     # loop over roots of g, change user's ginvals
        Res_ginv = inv(dgdt[i])     # residue of 1/g
        ginvals[n+1] -= Res_ginv / (x[n+1] - z)
        for j in 1:n
            ginvals[j] -= Res_ginv / (x[j] - z)
            ginvals[2n+2-j] -= Res_ginv / (-x[j] - z)
        end
        # @. ginvals -= Res_ginv/(r.x - z)    # subtract each pole off 1/g vals
        Ipoles += Res_ginv * log((1.0-z)/(-1.0-z))  # add analytic pole integral
    end
    sp = evalrule(ginvals, a,b, x,w,gw, nrm) # GK on corr 1/g vals
    if sp.E > s.E
        return s                # error not improved, revert to plain GK
    else
        sca = (b-a)/2
        return Segment(a,b, sp.I+sca*Ipoles, sp.E)
    end
end

"compute the vandermonde matrix for a symmetric Kronrod rule"
function kronrod_vandermonde(x::Vector{T}) where {T}
    (n = length(x)-1) > 0 || throw(ArgumentError("Kronrod rule not large enough"))
    V = Matrix{T}(undef, 2n+1, 2n+1)
    V[:,1] .= one(T)
    V[n+1,2] = x[n+1]
    for i in 1:n
        V[i,2] = x[i]
        V[2n+2-i,2] = -x[i]
    end
    v = view(V, :, 2)
    for j in 3:2n+1
        w = view(V, :, j-1)
        V[:,j] .= v .* w
    end
    return V
end

const lucache = Dict{Type,Dict}(
    Float64 => Dict{Int,LU{Float64, Matrix{Float64}, Vector{Int64}}}(
        7 => lu!(kronrod_vandermonde(xd7))))


@generated function _cachedlu(::Type{TF}, n::Int) where {TF}
    cache = haskey(lucache, TF) ? lucache[TF] : (lucache[TF] = Dict{Int,LU{TF,Matrix{TF},Vector{Int64}}}())
    :(haskey($cache, n) ? $cache[n] : ($cache[n] = lu!(kronrod_vandermonde(cachedrule($TF, n)[1]))))
end

function cachedlu(::Type{T}, n::Integer) where {T<:Number}
    _cachedlu(typeof(float(real(one(T)))), Int(n))
end

"""
    roots, derivs = find_near_roots(vals, nodes;
                                    rho=1.0, fac=nothing, meth="PR")

Returns complex-valued roots of unique polynomial approximant g(z)
matching the vector of `vals` at the vector `nodes`.  The nodes
are assumed to be well-chosen for interpolation on [-1,1].
'roots' are returned in order of increasing (Bernstein) distance
from the interval [-1,1]. It also computes 'derivs', the
corresponding values of g' at each kept root.

`rho > 0.0` sets the Bernstein ellipse parameter within which to keep
roots. Recall that the ellipse for the standard segment `[-1,1]` has
semiaxes `cosh(rho)` horizontally and `sinh(rho)` vertically.

`fac` allows user to pass in a pre-factorized (eg LU) object for
the Vandermonde matrix. This accelerates things by 3us for 15 nodes.

`meth` controls method for polynomial root-finding:
        "PR" - PolynomialRoots.roots()
        "PR5" - PolynomialRoots.roots5() degree-5 only (worse perf)
        "F" - few_poly_roots local attempt

To do:
1) template so compiles for known n (speed up roots? poly eval?)
2) compare Boyd version using Cheby points (needs twice the degree)

Alex Barnett 6/29/23 - 7/4/23
"""
function find_near_roots(vals::Vector, nodes::Vector, rho, fac, meth)
    n = length(nodes)
    c = fac \ vals       # solve via passed-in LU factorization of V (1.5us)
    roots = get_roots(meth, c, vals, nodes)

    # now solve roots = (t+1/t)/2 to get t (Joukowsky map) values (1 us)
    t = @. roots + sqrt(roots^2 - 1.0)
    rhos = abs.(log.(abs.(t)))        # Bernstein param for each root
    nkeep = sum(rhos .< rho)          # then keep t with e^-rho < t < e^rho
    inds = sortperm(rhos)[1:nkeep]    # indices to keep
    roots = roots[inds]
    derivs = zero(roots)              # initialize deriv vals
    derc = Vector{typeof(c[1])}(undef,n-1)    # alloc
    for (i,r) in enumerate(roots)     # (1us for 14 roots degree 14)
        for k=1:n-1
            derc[k] = k*c[k+1]        # coeffs of deriv of poly, no alloc loop
        end
        derivs[i] = Base.evalpoly(r,derc)   # eval at root (14 ns)
    end
    return roots, derivs
end

abstract type RootAlgorithm end

"""
    roots,rvals = few_poly_roots(c::Vector, vals::Vector, nodes::Vector, n::Int; verb=0)

Return `nr` polynomial roots `roots` given by coefficients `c`,
and `rvals` corresponding polynomial values.

Speed goal is 1 us for nr about 3 and degree-14. May use
`vals` function values at `nodes` which should fill out [-1,1].
Alternates Newton for next root, then deflation to factor it out.

No failure reporting yet. User should use `rvals` as quality check.
"""
function few_poly_roots(c::Vector{T}, vals::Vector{T}, nodes::Vector,
                        nr::Int=3) where T
    # Barnett 7/15/23
    roots = similar(c,nr); rvals = similar(c,nr)     # size-nr output arrays
    cl = copy(c)                 # alloc local copy of coeffs
    cp = similar(c,length(c)-1)           # alloc coeffs of deriv
    for jr = 1:nr                # loop over roots to find
        p = length(c)-jr         # degree of current poly
        resize!(cp,p)            # no alloc
        for k=1:p; cp[k] = k*cl[k+1]; end  # coeffs of deriv
        drok = 1e-8     # Newton params (expected err is drok^2; quadr conv)
        itermax = 10
        k = 0
        dr = 1.0
        if jr==1    # r init method
            idx = argmin(i -> abs(vals[i]), eachindex(vals))
            r = complex(nodes[idx > length(nodes) ? length(vals)-idx+1 : idx])    # init at node w/ min val?
        else
            r = complex(0.0)    # too crude init?
            # *** would need update all vals via evalpoly, O(p^2.nr) tot cost?
        end
        while dr>drok && k<itermax
            rold = r
            r -= evalpoly(r,cl) / evalpoly(r,cp)   # lengths determines degrees
            dr = abs(r-rold)
            k += 1
        end
        # IDEA: if dr>tol; return roots[1:jr-1]  # failed
        for k=1:p; cp[k]=cl[k]; end       # overwrite cp as workspace
        # deflate poly from cp workspace back into cl coeffs (degree p-1)
        cl[p] = cl[p+1]          # start deflation downwards recurrence
        resize!(cl,p)            # trunc len by one, no alloc
        for k=p-1:-1:1; cl[k] = cp[k+1] + r*cl[k+1]; end
        rvals[jr] = cp[1]+r*cl[1]     # final recurrence evals the poly at r
        roots[jr] = r            # copy out answer
    end
    return roots, rvals
end

# could store the newton algorithm parameters in this struct in order to pass them to routine
struct NewtonDeflation <: RootAlgorithm end

function get_roots(::NewtonDeflation, c, vals, nodes)
    roots, rvals = few_poly_roots(c,vals,nodes,3)
    return roots
end

struct CompanionMatrix <: RootAlgorithm end

"""
    roots_companion(a)

    find all complex roots of polynomial a[1]*z^n + a[2]*z^(n-1) + ... + a[n+1]
    via companion matrix EVP in O(n^3) time. Similar to MATLAB roots.
    Note poly coeffs are in reverse order that in many Julia pkgs.
    If the entire C plane is a root, returns [complex(NaN)].

    Local reference implementation; superceded by other pkgs.
"""
function roots_companion(a::AbstractVector{<:Number})
    # does not allow dims>1 arrays
    a = complex(a)          # idempotent, unlike Complex{T} for T a type...
    T = eltype(a)
    while length(a)>1 && a[1]==0.0         # gobble up any zero leading coeffs
        a = a[2:end]
    end
    if isempty(a) || (a==[0.0])            # done, meaningless
        return [complex(NaN)]     # array, for type stability. signifies all C
    end
    deg = length(a)-1       # a is now length>1 with nonzero 1st entry
    if deg==0
        return T[]          # done: empty list of C-#s
    end
    a = reshape(a[deg+1:-1:2] ./ a[1],(deg,1))    # make monic, col and flip
    C = [ [zeros(T,1,deg-1); Matrix{T}(I,deg-1,deg-1)] -a ]   # stack companion mat
    # at this point we want case of real C to be possible, faster than complex
    complex(eigvals!(C))    # overwrite C, and we don't want the vectors
end
# Note re fact that we don't need evecs: see also LinearAlgebra.LAPACK.geev!

function get_roots(::CompanionMatrix, c, vals, nodes)
    return roots_companion(reverse(c))   # also 10x slower (~100us)
end

#=
# plan to offer the following methods through a package extension

struct PR <: RootAlgorithm end
function get_roots(::PR, c, vals, nodes)
    return PolynomialRoots.roots(c)       # find all roots (typ 7-10us)
end

struct PR5 <: RootAlgorithm end
function get_roots(::PR5, c, vals, nodes)
    return PolynomialRoots.roots5(c[1:6]) # find roots only degree-5 (4us)
end

=#
