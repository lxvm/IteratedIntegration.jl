# IteratedIntegration.jl

| Documentation | Build Status | Coverage | Version |
| :-: | :-: | :-: | :-: |
| [![][docs-stable-img]][docs-stable-url] | [![][action-img]][action-url] | [![][codecov-img]][codecov-url] | [![ver-img]][ver-url] |
| [![][docs-dev-img]][docs-dev-url] | [![][pkgeval-img]][pkgeval-url] | [![][aqua-img]][aqua-url] | [![deps-img]][deps-url] |

This package provides experimental algorithms for globally h-adaptive iterated
integration.

The main takeaways of these experiments are that:
- pole-subtraction is superior to quadgk for integration of meromorphic functions
- The peak missing problem, which occurs for near singular functions, can be
  elegantly solved with auxiliary integrands
- nested integrals can be implemented reliably when absolute tolerances for
  inner integrals are scaled by the measure of the outer integration domains
- nested integrals cannot be reliably replaced by globally-adaptive schemes using
  a treap (tree-heap) and any adaptation scheme other than depth-first

## Algorithm

See the `notes` folder for a description of the IAI algorithm.
The implementation is based on [QuadGK.jl](https://github.com/JuliaMath/QuadGK.jl)

## Author and Copyright

IteratedIntegration.jl was written by [Lorenzo Van
Muñoz](https://web.mit.edu/lxvm/www/), and is free/open-source software under
the MIT license.

The algorithms in `IteratedIntegration.MeroQuadGK` and
`IteratedIntegration.ContQuadGK` were developed by [Alex
Barnett](https://github.com/ahbarnett/bz-integral) and are distributed
under the Apache 2.0 License

## Related packages
- [QuadGK.jl](https://github.com/JuliaMath/QuadGK.jl)
- [HCubature.jl](https://github.com/JuliaMath/HCubature.jl)
- [Integrals.jl](https://github.com/SciML/Integrals.jl)


<!-- badges -->

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://lxvm.github.io/IteratedIntegration.jl/stable/

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://lxvm.github.io/IteratedIntegration.jl/dev/

[action-img]: https://github.com/lxvm/IteratedIntegration.jl/actions/workflows/CI.yml/badge.svg?branch=main
[action-url]: https://github.com/lxvm/IteratedIntegration.jl/actions/?query=workflow:CI

[pkgeval-img]: https://juliahub.com/docs/General/IteratedIntegration/stable/pkgeval.svg
[pkgeval-url]: https://juliahub.com/ui/Packages/General/IteratedIntegration

[codecov-img]: https://codecov.io/github/lxvm/IteratedIntegration.jl/branch/main/graph/badge.svg
[codecov-url]: https://app.codecov.io/github/lxvm/IteratedIntegration.jl

[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl

[ver-img]: https://juliahub.com/docs/IteratedIntegration/version.svg
[ver-url]: https://juliahub.com/ui/Packages/IteratedIntegration/UDEDl

[deps-img]: https://juliahub.com/docs/General/IteratedIntegration/stable/deps.svg
[deps-url]: https://juliahub.com/ui/Packages/General/IteratedIntegration?t=2