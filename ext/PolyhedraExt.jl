module PolyhedraExt

using Polyhedra: Polyhedron, VRepresentation, vrep, points, fulldim, hasallrays, coefficient_type, Polyhedra
import IteratedIntegration: fixandeliminate, segments, AbstractIteratedLimits, load_limits

"""
    PolyhedralLimits(::Polyhedron)

Integration endpoints from a convex hull.
"""
struct PolyhedralLimits{d,T,P} <: AbstractIteratedLimits{d,T}
    p::P
    PolyhedralLimits{d}(p::P) where {d,P<:Polyhedron} = new{d,coefficient_type(p),P}(p)
end
PolyhedralLimits(p::Polyhedron) = PolyhedralLimits{fulldim(p)}(p)

# TODO: compute vrep at the same time
function fixandeliminate(l::PolyhedralLimits{d}, x, ::Val{dim}) where {d,dim}
    p = Polyhedra.fixandeliminate(l.p, dim, x)
    points(p)
    vrep(p) # compute/save the vrep
    return PolyhedralLimits{d-1}(p)
end

function segments_(v::VRepresentation, dim::Integer)
    hasallrays(v) && error("Infinite limits not implemented: found ray in V representation")
    (d = fulldim(v)) >= dim >= 1 || error("V representation of fulldim $d doesn't have index $dim")
    pts = points(v)
    rtol = atol = sqrt(eps(eltype(eltype(pts))))
    segs=Vector{NTuple{2,eltype(eltype(pts))}}(undef, length(pts))
    numpts = 0
    for p in pts
        vert = p[dim]
        test = isapprox(vert, atol=atol, rtol=rtol)
        if !any(x -> test(x[1]), @view(segs[begin:begin+numpts-1]))
            numpts += 1
            segs[numpts] = (vert,vert)
        end
    end
    @assert numpts >= 2 segs
    resize!(segs,numpts)
    sort!(segs)
    top = pop!(segs)
    for i in numpts-1:-1:1
        segs[i] = top = (segs[i][2],top[1])
    end
    return segs
end
function segments(l::PolyhedralLimits{d}, dim=d) where d
    segments_(vrep(l.p), dim)
end

load_limits(p::Polyhedron) = PolyhedralLimits(p)

end
