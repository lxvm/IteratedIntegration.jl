function measure(l::AbstractIteratedLimits)
    if ndims(l) == 1
        a, b = segments(l, 1)
        return abs(b-a)
    else
        quadgk(x -> measure(fixandeliminate(l, x, Val(ndims(l)))), segments(l, ndims(l))...)[1]
    end
end