module PolyhedraExt

using Polyhedra: Polyhedron, VRepresentation, vrep, points, fulldim, hasallrays, coefficient_type, Polyhedra
import IteratedIntegration: fixandeliminate, endpoints, AbstractIteratedLimits, iterated_segs, load_limits

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
function fixandeliminate(l::PolyhedralLimits{d}, x) where d
    p = Polyhedra.fixandeliminate(l.p, d, x)
    vrep(p) # compute/save the vrep
    return PolyhedralLimits{d-1}(p)
end

function endpoints(v::VRepresentation, dim::Integer)
    hasallrays(v) && error("Infinite limits not implemented: found ray in V representation")
    (d = fulldim(v)) >= dim >= 1 || error("V representation of fulldim $d doesn't have index $dim")
    return extrema(v -> v[dim], points(v))
end
endpoints(l::PolyhedralLimits{d}, dim=d) where d = endpoints(vrep(l.p), dim)

# we need to add breakpoints at the projections of vertices because there may be kinks in
# the volume at these points
iterated_segs(_, l::PolyhedralLimits{1}, a, b, ::Val{initdivs}) where initdivs = (a, b)
function iterated_segs(_, l::PolyhedralLimits, a, b, ::Val{initdivs}) where initdivs
    vert = unique(v[end] for v in points(l.p))
    sort!(vert)
    return tuple(vert...)
end

load_limits(p::Polyhedron) = PolyhedralLimits(p)

end
