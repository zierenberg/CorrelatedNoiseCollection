cd(@__DIR__)
using DocStringExtensions
using Statistics
using StatsBase
using LinearAlgebra
using DataFrames
using FastTransforms
using Distributions
using Random
# Base.:|>(args::Tuple, f) = f(args...)

C(r, a = 2.0) = (1 + r^2)^(-a / 2)

periodic_dist(x, L) = x ≤ L / 2 ? x : L - x

"""
    generate_corr_matrix(dims::Tuple, C::Function; T::Type = Float64)

Calculate the function `C` for each distance `r` and returns an array where each C(r) is stored.
The distnace `r` is measured from upper left corner and implies periodic boundary conditions.

# Arguments:
- `dims::Tuple`: dimensions of the desired lattice
- `C::Function`: desired correlation function `C(r)`
- `T::Type = Float64`: type of the retured array
"""
function generate_corr_matrix(dims::Tuple, C::Function; T::Type = Float64)
	Λ = Array{T}(undef, dims)
	fill_periodic_distances!(Λ, C)
	return Λ
end

"""
    fill_periodic_distances!(Λ::AbstractArray, C::Function)

Helper function for the `generate_corr_matrix` function.
The type of `Λ` is known on call and therefore can this inner function works faster.

# Arguments:
- `Λ::AbstractArray`: array to save the `C(r)` to
- `C::Function`: desired correlation function `C(r)`
"""
function fill_periodic_distances!(Λ::AbstractArray, C::Function)
	for i ∈ CartesianIndices(Λ)
		r = norm(periodic_dist.(Tuple(i) .- 1, size(Λ)))
		Λ[i] = C(r)
	end
end

"""
    generate_continuous_disorder(dims::Tuple, S::AbstractArray)

Produces an array with dimensions `dims` with correlated disorder where the correlation follows the function `C(r)`

# Arguments:
- `dims::Tuple`: dimensions of the desired lattice
- `C::Function`: desired correlation function `C(r)`
"""
function generate_continuous_disorder(dims::Tuple, S::AbstractArray)
	φq = sqrt.(max.(S, 0)) .* rand.(Normal(0, √(2 * prod(dims)))) # 2V = 2prod(dims) is used because the random numbers are produced only in real space
	φ = real.(ifft(φq))
	return φ
end

"""
    generate_S(dims::Tuple, C::Function)

Produces a spectral density array with dimensions `dims` of the array with `C(r)` entries

# Arguments:
- `dims::Tuple`: dimensions of the desired lattice
- `C::Function`: desired correlation function `C(r)`
"""
generate_S(dims::Tuple, C::Function) = real.(fft(generate_corr_matrix(dims, C)))

"""
    transform_to_discrete_disorder(Λ::AbstractArray, p::Number; T = Int8(1), F = Int8(0))

Produces a truncated array similar to `Λ` but only with `T` and `F` values.
The concentration of `T` is given through `p`.

# Arguments:
- `Λ::AbstractArray`: expect an array with continuous correlated numbers
- `p::Number`: concentration of `T` values
- `T = Int8(1)`: values with concentration `p` to fill into the returned array
- `F = Int8(0)`: values with concentration `1 - p` to fill into the returned array
"""
transform_to_discrete_disorder(Λ::AbstractArray, p::Number; T = Int8(1), F = Int8(0)) = [i ≤ quantile(Normal(0, 1), p) ? T : F for i ∈ Λ]

######################### TEST #########################
Random.seed!(1)
L1 = 1000
L2 = 2000
a = Inf
p = 0.8

S = generate_S((L1, L2), r -> C(r, a))
Λc = generate_continuous_disorder((L1, L2), S)
Λd = transform_to_discrete_disorder(Λc, p)
@show mean(Λc)
@show std(Λc)
@show check_p = counts(Λd, 1:1)[1] / prod(size(Λd))

using PyPlot
clf()
imshow(Λc; interpolation = "none")
savefig("$(L1)x$(L2)_a$(a)_p$(p)_continuous_disorder.pdf")

clf()
imshow(Λd, interpolation = "none")
savefig("$(L1)x$(L2)_a$(a)_p$(p)_discrete_disorder.pdf")

# ######################### TEST SPEED #########################
# using BenchmarkTools

# @btime generate_continuous_disorder((L1, L2), r -> C(r, a))
# @btime transform_to_discrete_disorder(Λc, p)

A = -1.38

Int(round(abs(A))) # reads: perform Int of round of abs of A
round(abs(A)) |> Int
abs(A) |> round |> Int
A |> abs |> round |> Int # reads: take A, do abs, round it, make it an Int

### dot version

A = [0.1, -1.3, 2.5]

Int.(round.(abs.(A)))
round.(abs.(A)) .|> Int
abs.(A) .|> round .|> Int
A .|> abs .|> round .|> Int

### absurd

(1 + 3)^(1 / 2)
(1 + 3) |> x -> ^(x, 1 / 2)


############### in-place operations and |>

A = rand(10000)

B = round.(abs.(A))
A .= round.(abs.(A)) # in-place
A .= A .|> abs .|> round # in-place
