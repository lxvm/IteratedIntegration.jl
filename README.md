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
 13.810075 seconds (187.29 M allocations: 2.791 GiB, 4.05% gc time)
(3.014355431929516e-11 - 221.4503334458075im, 6.5132132917468245e-6, 187271895)

julia> @time I2, = iai_count(f, (0,0,0), (2pi,2pi,2pi); atol=1e-5)
 85.821637 seconds (521.70 M allocations: 13.221 GiB, 2.34% gc time)
(1.9539925233402755e-14 - 221.45033344593668im, 8.248907526206854e-6, 151635225)

julia> abs(I1-I2) # check how closely solutions agree to within tolerance
1.32642633231035e-10
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
