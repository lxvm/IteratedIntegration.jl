module ContQuadGK

using QuadGK: Segment, cachedrule, InplaceIntegrand, alloc_segbuf, realone
using DataStructures, LinearAlgebra
import Base.Order.Reverse
import QuadGK: evalrule, handle_infinities

# use 2 heaps, one real and one complex

export contquadgk, contquadgk!

include("adapt.jl")

end
