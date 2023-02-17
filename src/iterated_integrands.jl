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
