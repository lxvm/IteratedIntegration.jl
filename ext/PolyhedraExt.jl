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
    vrep(p) # compute/save the vrep
    return PolyhedralLimits{d-1}(p)
end

function segments_(v::VRepresentation, dim::Integer)
    hasallrays(v) && error("Infinite limits not implemented: found ray in V representation")
    (d = fulldim(v)) >= dim >= 1 || error("V representation of fulldim $d doesn't have index $dim")
    segs = unique((x=p[end]; (x,x)) for p in points(v))
    sort!(segs)
    for i in eachindex(@view(segs[begin:end-1]))
        segs[i] = (segs[i][1], segs[i+1][1])
    end
    deleteat!(segs, lastindex(segs))
    return segs
end
segments(l::PolyhedralLimits{d}, dim=d) where d = segments_(vrep(l.p), dim)

load_limits(p::Polyhedron) = PolyhedralLimits(p)

end
