module MeroQuadGK

using LinearAlgebra
import QuadGK: Segment, handle_infinities, xd7, cachedrule
using DataStructures
import Base.Order.Reverse

export meroquadgk, BatchIntegrand

include("evalrule.jl")
include("polesub.jl")
include("adapt.jl")

end
