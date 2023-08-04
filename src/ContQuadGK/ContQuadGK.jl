module ContQuadGK

using QuadGK: Segment, cachedrule, handle_infinities, alloc_segbuf, xd7
using DataStructures, LinearAlgebra
import Base.Order.Reverse
import QuadGK: evalrule

# use 2 heaps, one real and one complex

export contquadgk, contquadgk!

include("roots.jl")
include("evalrule.jl")
include("adapt.jl")

end
