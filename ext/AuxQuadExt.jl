module AuxQuadExt

using LinearAlgebra: norm
using IteratedIntegration: iterated_segs, endpoints, QuadNest, iterated_outer_tol, CubicLimits, ThunkIntegrand, alloc_segbufs
import IteratedIntegration: nested_auxquadgk
using AuxQuad: do_auxquadgk, Sequential
import AuxQuad: Parallel

function do_nested_auxquadgk(q::QuadNest{1})
    segs = iterated_segs(q.f, q.l, q.a, q.b, q.initdivs[1])
    do_auxquadgk(q.f, segs, q.order, q.atol, q.rtol, q.maxevals, q.norm, q.segbufs[1]...)
end

function do_nested_auxquadgk(q::QuadNest{d}) where d
    segs = iterated_segs(q.f, q.l, q.a, q.b, q.initdivs[d])
    atol = iterated_outer_tol(q.atol, q.a, q.b)
    do_auxquadgk(q, segs, q.order, atol, q.rtol, q.maxevals, q.norm, q.segbufs[d]...)
end

function nested_auxquadgk(f, a, b; kwargs...)
    l = CubicLimits(a, b)
    nested_auxquadgk(ThunkIntegrand{ndims(l)}(f), l; kwargs...)
end
function nested_auxquadgk(f::F, l::L; order=7, atol=nothing, rtol=nothing, norm=norm, maxevals=10^7, initdivs=nothing, segbufs=nothing, parallels=nothing) where {F,L}
    initdivs_ = initdivs === nothing ? ntuple(i -> Val(1), Val(ndims(l))) : initdivs
    segbufs_ = segbufs === nothing ? alloc_segbufs(f, l) : segbufs
    parallels_ = parallels === nothing ? ntuple(i -> Sequential(), Val(ndims(l))) : parallels
    atol_ = something(atol, zero(eltype(l)))
    rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(one(eltype(l)))) : zero(eltype(l)))
    a, b = endpoints(l)
    q = QuadNest(Val(ndims(l)), f, l,a,b, order, atol_, rtol_, maxevals, norm, initdivs_, Tuple(zip(segbufs_, parallels_)), do_nested_auxquadgk)
    do_nested_auxquadgk(q)
end

"""
    Parallel(domain_type, range_type, error_type, ndim::Int; order=7)

Allocate `ndim` parallelization buffers for use in `nested_auxquadgk`.
"""
function Parallel(TX, TI, TE, ndim::Int; order=7)
    ntuple(n -> Parallel(TX, TI, TE; order=order), ndim)
end

end
