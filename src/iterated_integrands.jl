"""
    ThunkIntegrand{d}(f)

Store `f` to evaluate `f(x)` at a later time. Employed by
`iterated_integration` for generic integrands that haven't been specialized to
use `iterated_pre_eval`. Note that `x isa SVector`, so the function arguments
needs to expect the behavior. `d` is a parameter specifying the number of
variables in the vector input of `f`. This is good for integrands like `∫∫∫f`.
"""
struct ThunkIntegrand{d,F,X} <: AbstractIteratedIntegrand{d}
    f::F
    x::X
    ThunkIntegrand{d}(f::F, x::X=()) where {d,F,X<:NTuple} =
        new{d,F,X}(f, x)
end

iterated_integrand(_::ThunkIntegrand{d}, x, ::Val{d}) where d = x
iterated_integrand(f::ThunkIntegrand{1}, x, ::Val{1}) =
    f.f(SVector{1+length(f.x)}(x, f.x...))

iterated_pre_eval(f::ThunkIntegrand{d}, x, ::Val{d}) where d =
    ThunkIntegrand{d-1}(f.f, (x, f.x...))


"""
    IteratedIntegrand(fs...)

Represents a nested integral of the form `∫dxN fN( ... ∫dx1 f1([x1, ..., xN]))`
so the functions need to know the arguments and their layout, since the
variables are passed as a vectors.
"""
struct IteratedIntegrand{d,F,X} <: AbstractIteratedIntegrand{d}
    f::F
    x::X
    IteratedIntegrand{d}(f::F, x::X) where {d,F<:Tuple{Vararg{Any,d}},X<:NTuple} =
        new{d,F,X}(f, x)
end
IteratedIntegrand(f...) = IteratedIntegrand{length(f)}(f, ())

iterated_integrand(f::IteratedIntegrand{d}, x, ::Val{d}) where d = f.f[d](x)
iterated_integrand(f::IteratedIntegrand{1}, x, ::Val{1}) =
    f.f[1](SVector{1+length(f.x)}(x, f.x...))

iterated_pre_eval(f::IteratedIntegrand{d}, x, ::Val{d}) where d =
    IteratedIntegrand{d-1}(f.f[1:d-1], (x, f.x...))

iterated_vars(f::Union{ThunkIntegrand,IteratedIntegrand}) = SVector{length(f.x)}(f.x...)


struct ProductIteratedIntegrand{d,idx,F,S} <:AbstractIteratedIntegrand{d}
    f::F
    fs::S
    ProductIteratedIntegrand{d,idx}(f::F, fs::S) where {d,idx,F,S<:Tuple} =
        new{d,idx,F,S}(f,fs)
end

ProductIteratedIntegrand(f, fs::AbstractIteratedIntegrand...) = ProductIteratedIntegrand(f, fs)
function ProductIteratedIntegrand(f, fs::Tuple{Vararg{AbstractIteratedIntegrand}})
    dims = map(ndims, fs)
    d = sum(dims)
    m = 0; k = 1
    idx = ntuple(d) do n
        if n-m < dims[k]
            return k
        else
            m += dims[k]
            k += 1
            return k-1
        end
    end
    ProductIteratedIntegrand{d,reverse(idx)}(f, fs)
end


function iterated_integrand(f::ProductIteratedIntegrand{d,idx}, x, ::Val{d}) where {d,idx}
    iterated_integrand(f.fs[idx[d]], x)
end
function iterated_integrand(f::ProductIteratedIntegrand{1}, x, ::Val{1})
    y = (iterated_vars(f.fs[end], x), f.fs[1:end-1]...)
    f.f(y...)
end

function iterated_pre_eval(f::ProductIteratedIntegrand{d,idx}, x, ::Val{d}) where {d,idx}
    i = idx[d]
    g = iterated_pre_eval(f.fs[i], x)
    ProductIteratedIntegrand{d-1,idx}(f.f, setindex(f.fs, g, i))
end
function iterated_pre_eval(f::ProductIteratedIntegrand{d,idx,F,<:Tuple{<:AbstractIteratedIntegrand{1},Vararg{AbstractIteratedIntegrand}}}, x, ::Val{d}) where {d,idx,F}
    i = idx[d]
    y = iterated_vars(f.fs[i], x)
    ProductIteratedIntegrand{d-1,idx}(f.f, setindex(f.fs, y, i))
end


# IDEA: separate vars from integrand so that ProductIteratedIntegrand is a single function
# and a collection of vars, whereas it is currently a function and collection of vars.
# or somehow find a way to link vars and lims
