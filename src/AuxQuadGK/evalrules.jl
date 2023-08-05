@inline function evalrules(f, s, x,w,gw, nrm)
    N = length(s)
    if s isa NTuple
        return ntuple(i -> evalrule(f, s[i],s[i+1], x,w,gw, nrm), Val(N-1))
    else
        return map(i -> evalrule(f, s[i],s[i+1], x,w,gw, nrm), 1:N-1)
    end
end
