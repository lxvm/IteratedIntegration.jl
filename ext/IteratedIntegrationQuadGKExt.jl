module IteratedIntegrationQuadGKExt

using QuadGK: quadgk

using IteratedIntegration: AbstractIteratedLimits
import IteratedIntegration: measure

function measure(::typeof(quadgk), l::AbstractIteratedLimits)
    measure(l) do f, segs
        I, E = quadgk(f, segs...)
        return I
    end
end

end