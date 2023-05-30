module PolyhedraExt

using Polyhedra: Polyhedron, VRepresentation, vrep, points, fulldim, hasallrays, coefficient_type, Polyhedra
import IteratedIntegration: fixandeliminate, endpoints, AbstractIteratedLimits, load_limits

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
fixandeliminate(l::PolyhedralLimits{d}, x) where d =
    PolyhedralLimits{d-1}(Polyhedra.fixandeliminate(l.p, d, x))

endpoints(l::PolyhedralLimits{d}, dim=d) where d =
    endpoints(vrep(l.p), dim)
function endpoints(v::VRepresentation, dim::Integer)
    hasallrays(v) && error("Infinite limits not implemented: found ray in V representation")
    (d = fulldim(v)) >= dim >= 1 || error("V representation of fulldim $d doesn't have index $dim")
    extrema(v -> v[dim], points(v))
end

load_limits(p::Polyhedron) = PolyhedralLimits(p)

end