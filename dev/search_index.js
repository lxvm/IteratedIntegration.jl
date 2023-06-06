var documenterSearchIndex = {"docs":
[{"location":"methods/#Reference","page":"Manual","title":"Reference","text":"","category":"section"},{"location":"methods/","page":"Manual","title":"Manual","text":"Modules = [IteratedIntegration]\nOrder   = [:type, :function]","category":"page"},{"location":"methods/#IteratedIntegration.AbstractIteratedIntegrand","page":"Manual","title":"IteratedIntegration.AbstractIteratedIntegrand","text":"AbstractIteratedIntegrand{d}\n\nSupertype for integrands compatible with iterated integration of d variables.\n\n(::AbstractIteratedIntegrand)(x)\n\nEvaluate the integrand at the point x\n\n\n\n\n\n","category":"type"},{"location":"methods/#IteratedIntegration.AbstractIteratedLimits","page":"Manual","title":"IteratedIntegration.AbstractIteratedLimits","text":"AbstractIteratedLimits{d,T}\n\nSupertype for limits of integration over a domain with elements of type SVector{d,T}. In order to work with iterated integration, the following methods must be implemented\n\nInterface\n\nnextdim: returns Val(n) where n is the next variable of integration\nsegments: returns an iterator over intervals to integrate in the current dimension\nfixandeliminate: return another limit object with one of the variables of integration eliminated\n\n\n\n\n\n","category":"type"},{"location":"methods/#IteratedIntegration.CubicLimits","page":"Manual","title":"IteratedIntegration.CubicLimits","text":"CubicLimits(a, b)\n\nStore integration limit information for a hypercube with vertices a and b. which can be can be real numbers, tuples, or AbstractVectors. The outermost variable of integration corresponds to the last entry.\n\n\n\n\n\n","category":"type"},{"location":"methods/#IteratedIntegration.ProductLimits","page":"Manual","title":"IteratedIntegration.ProductLimits","text":"ProductLimits(lims::AbstractIteratedLimits...)\n\nConstruct a collection of limits which yields the first limit followed by the second, and so on. The inner limits are not allowed to depend on the outer ones. The outermost variable of integration should be placed first, i.e. int_Omega int_Gamma should be ProductLimits(Ω, Γ). Although changing the order of the limits should not change the results, putting the shortest limits first may save nested_quadgk some work.\n\n\n\n\n\n","category":"type"},{"location":"methods/#IteratedIntegration.TetrahedralLimits","page":"Manual","title":"IteratedIntegration.TetrahedralLimits","text":"TetrahedralLimits(a::NTuple{d}) where d\n\nA parametrization of the integration limits for a tetrahedron whose vertices are\n\n( 0.0,  0.0, ...,  0.0)\n( 0.0,  0.0, ..., a[d])\n…\n( 0.0, a[2], ..., a[d])\n(a[1], a[2], ..., a[d])\n\n\n\n\n\n","category":"type"},{"location":"methods/#IteratedIntegration.TranslatedLimits","page":"Manual","title":"IteratedIntegration.TranslatedLimits","text":"TranslatedLimits(lims::AbstractIteratedLimits{d}, t::NTuple{d}) where d\n\nReturns the limits of lims translated by offsets in t.\n\n\n\n\n\n","category":"type"},{"location":"methods/#IteratedIntegration.alloc_segbufs-Tuple{Any, Any, Any, Int64}","page":"Manual","title":"IteratedIntegration.alloc_segbufs","text":"alloc_segbufs(coefficient_type, range_type, norm_type, ndim)\n\nThis helper function will allocate enough segment buffers as are needed for an nested_quadgk call of integrand f and integration limits l. coefficient_type should be eltype(l), typesof_fx should be the return type of the integrand f for each iteration of integration, typesof_nfx should be the types of the norms of a value of f for each iteration of integration, and ndim should be ndims(l).\n\n\n\n\n\n","category":"method"},{"location":"methods/#IteratedIntegration.fixandeliminate","page":"Manual","title":"IteratedIntegration.fixandeliminate","text":"fixandeliminate(l::AbstractIteratedLimits, x)\n\nFix the outermost variable of integration and return the inner limits.\n\nnote: For developers\nRealizations of type T<:AbstractIteratedLimits only have to implement a method with signature fixandeliminate(::T, ::Number). The result must also have dimension one less than the input, and this should only be called when ndims= 1\n\n\n\n\n\n","category":"function"},{"location":"methods/#IteratedIntegration.iterated_integrand","page":"Manual","title":"IteratedIntegration.iterated_integrand","text":"iterated_integrand(f::AbstractIteratedIntegrand{d}, x, ::Val{d}) where d\n\nEvaluate a function on the inner integral x. When d==1, x is the argument of the innermost integral that is passed by the integration routine via f(x).\n\n\n\n\n\n","category":"function"},{"location":"methods/#IteratedIntegration.iterated_pre_eval","page":"Manual","title":"IteratedIntegration.iterated_pre_eval","text":"iterated_pre_eval(f::AbstractIteratedIntegrand{d}, x, ::Val{d}) where d\n\nPerform a precomputation on f using the value of a variable of integration, x. Certain types of functions, such as Fourier series, take can use x to precompute a new integrand for the remaining variables of integration that is more computationally efficient. Otherwise, the type can store x and delay the evaluation to the inner integral. This function must return the integrand for the subsequent integral, which should be an AbstractIteratedIntegrand{d-1}\n\n\n\n\n\n","category":"function"},{"location":"methods/#IteratedIntegration.iterated_vars","page":"Manual","title":"IteratedIntegration.iterated_vars","text":"iterated_vars(f::AbstractIteratedIntegrand)\niterated_vars(f::AbstractIteratedIntegrand, x)\n\nDump the variables of integration stored by f, optionally including x\n\n\n\n\n\n","category":"function"},{"location":"methods/#IteratedIntegration.load_limits","page":"Manual","title":"IteratedIntegration.load_limits","text":"load_limits(obj)\n\nLoad integration limits from an object. Serves as an api hook for package extensions with specialized limit types.\n\n\n\n\n\n","category":"function"},{"location":"methods/#IteratedIntegration.nested_quad-Tuple{Any, Any, Any}","page":"Manual","title":"IteratedIntegration.nested_quad","text":"nested_quad(f, a, b; kwargs...)\nnested_quad(f, l::AbstractIteratedLimits{d,T}; order=7, atol=nothing, rtol=nothing, norm=norm, maxevals=typemax(Int), initdivs=ntuple(i -> Val(1), Val{d}()), segbufs=nothing, parallels=nothing, rule=NestedGaussKronrod) where {d,T}\n\nCalls QuadGK to perform iterated 1D integration of f over a compact domain parametrized by AbstractIteratedLimits l. In the case two points a and b are passed, the integration region becomes the hypercube with those extremal vertices (which mimics hcubature).\n\nReturns a tuple (I, E) of the estimated integral and estimated error.\n\nKeyword options include a relative error tolerance rtol (if atol==0, defaults to sqrt(eps) in the precision of the norm of the return type), an absolute error tolerance atol (defaults to 0), a maximum number of function evaluations maxevals for each nested integral (defaults to 10^7), and the order of the integration rule (defaults to 7).\n\nThe algorithm is an adaptive Gauss-Kronrod integration technique: the integral in each interval is estimated using a Kronrod rule (2*order+1 points) and the error is estimated using an embedded Gauss rule (order points). The interval with the largest error is then subdivided into two intervals and the process is repeated until the desired error tolerance is achieved. This 1D procedure is applied recursively to each variable of integration in an order determined by l to obtain the multi-dimensional integral.\n\nUnlike quadgk, this routine does not allow infinite limits of integration nor unions of intervals to avoid singular points of the integrand. However, the initdivs keyword allows passing a tuple of integers which specifies the initial number of panels in each quadgk call at each level of integration.\n\nIn normal usage, nested_quadgk will allocate segment buffers. You can instead pass a preallocated buffer allocated using alloc_segbufs as the segbuf argument. This buffer can be used across multiple calls to avoid repeated allocation.\n\n\n\n\n\n","category":"method"},{"location":"methods/#IteratedIntegration.segments","page":"Manual","title":"IteratedIntegration.segments","text":"segments(::AbstractLimits)\n\nReturn a segitr, i.e. an iterator over interval\n\n\n\n\n\n","category":"function"},{"location":"#IteratedIntegration.jl","page":"Home","title":"IteratedIntegration.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"IteratedIntegration","category":"page"},{"location":"#IteratedIntegration","page":"Home","title":"IteratedIntegration","text":"A package for iterated adaptive integration (IAI) based on QuadGK.jl. Its main exports are nested_quadgk, a routine which performs multidimensional adaptive integration with nested quadgk calls, iai, which performs globally-adaptive iterated integration, and the AbstractIteratedLimits abstraction to evaluate parametrizations of limits of integration.\n\n\n\n\n\n","category":"module"}]
}
