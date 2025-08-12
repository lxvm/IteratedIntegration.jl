module IteratedIntegrationQuadGKExt

using QuadGK: quadgk

using IteratedIntegration: AbstractIteratedLimits
import IteratedIntegration: measure

function measure(::typeof(quadgk), l::AbstractIteratedLimits; kws...)
    measure(l) do f, segs
        I, E = quadgk(f, segs...; kws...)
        return I
    end
end

end