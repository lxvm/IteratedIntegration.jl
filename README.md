# IteratedIntegration.jl

[Documentation](https://lxvm.github.io/IteratedIntegration.jl/dev/)

This package provides experimental algorithms for globally h-adaptive iterated
integration.

Here is an example comparing `nested_quadgk` to the globally-adaptive `iai` for
a challenging localized 3D integrand of a form typically found in Brillouin zone
integration
```
julia> using IteratedIntegration

julia> f(x) = inv(0.01im+sum(sin, x))

julia> @time I1, = nested_quadgk_count(f, (0,0,0), (2pi,2pi,2pi); atol=1e-5)
 20.840139 seconds (270.43 M allocations: 4.030 GiB, 3.90% gc time)
(3.0500657555165844e-11 - 221.4503334456331im, 6.513167789370278e-6, 270412665)

julia> @time I2, = iai_count(f, (0,0,0), (2pi,2pi,2pi); atol=1e-5)
 34.775353 seconds (885.34 M allocations: 19.114 GiB, 7.68% gc time)
(-2.0161650127192843e-13 - 221.4503334459543im, 9.99997776208797e-6, 481793715)

julia> abs(I1-I2) # check how closely solutions agree to within tolerance
3.050987164194743e-10
```
It is interesting that `iai` has about 20% fewer function evaluations than
`nested_quadgk`, however both routines appear to be returning a solution with
nearly 5 digits of accuracy beyond what was requested

Additionally, here is a comparison of the same integral to
[HCubature.jl](https://github.com/JuliaMath/HCubature.jl) (note that I've
requested a larger tolerance so that the calculation finishes relatively quickly)
```
julia> using HCubature

julia> function hcubature_count(f, a, b; kwargs...)
       numevals=0
       I, E = hcubature(a, b; kwargs...) do x
           numevals += 1
           f(x)
       end
       return (I, E, numevals)
       end
hcubature_count (generic function with 1 method)

julia> @time I3, = hcubature_count(k, (0,0,0), (2pi,2pi,2pi); atol=1e-2)
 36.371073 seconds (406.85 M allocations: 6.776 GiB, 10.80% gc time)
(-4.367439804114779e-8 - 221.45050701410435im, 0.009999996360083807, 406846935)

julia> abs(I1-I3)
0.00017356830234685803
```
It is worth noting here that the reason `hcubature` uses far more function
evaluations is due to the geometry of the integrand and its interplay with
multi-dimensional quadratures, although `hcubature` returns a result much closer
to the requested tolerance.

## Algorithm

See the `notes` folder for a description of the IAI algorithm.
The implementation is based on [QuadGK.jl](https://github.com/JuliaMath/QuadGK.jl)

## Author and Copyright

IteratedIntegration.jl was written by [Lorenzo Van
Muñoz](https://web.mit.edu/lxvm/www/), and is free/open-source software under
the MIT license.

## Related packages
- [QuadGK.jl](https://github.com/JuliaMath/QuadGK.jl)
- [HCubature.jl](https://github.com/JuliaMath/HCubature.jl)
- [Integrals.jl](https://github.com/SciML/Integrals.jl)
