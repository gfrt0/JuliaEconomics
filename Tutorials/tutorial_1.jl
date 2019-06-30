# Bradley J. Setzler
# JuliaEconomics.com
# Tutorial 1: Introductory Example: Ordinary Least Squares (OLS) Estimation in Julia
# Passed test on Julia 1.1 (Giuseppe Forte edit 200619)

using Pkg

Pkg.add("Distributions")
Pkg.add("StatsBase")

using Random, Distributions, LinearAlgebra, GLM, DataFrames

Random.seed!(2)

N=1000
K=3

genX = MvNormal(Matrix{Float64}(I, K, K))
X = rand(genX,N)
X = X'
X_noconstant = X
constant = ones(N)
X = [constant X]

genEpsilon = Normal(0, 1)
epsilon = rand(genEpsilon,N)
trueParams = [0.1,0.5,-0.3,0.]
Y = X*trueParams + epsilon

function OLSestimator(y,x)
    betaols  = inv(x'*x)*(x'*y)

    residual = y - x * betaols
    sigma2c  = sum(residual.^2)/(size(x)[1] - size(x)[2])
    vcovdiag = sqrt.(diag(sigma2c .* inv(x' * x)))
    tvalues  = (betaols) ./ vcovdiag
    pvalues  = 2*(1 .- cdf.(Normal(), abs.(tvalues)))

    colnames = ["Estimate", "Std. Err.", "T Stat.", "P Value"]
    estimate = convert(DataFrame, [betaols vcovdiag tvalues pvalues])
    names!(estimate, Symbol.(colnames))
    return estimate
end

estimates = OLSestimator(Y,X)

A = convert(DataFrame, [Y X[:, [2,3,4]]])
colnames = ["y", "x1", "x2", "x3"]
names!(A, Symbol.(colnames))

GLM_estimates = lm(@formula(y ~ x1 + x2 + x3), A)

# Newton - Raphson One-Step Update

function NR1step(y,x, beta1s)
    betatilde  = beta1s + inv((x' * x)/size(x)[1])*x'*(y - x * beta1s)/size(x)[1]

    return betatilde
end

# As expected, the two estimators are equal up to computing error
betaols = estimates[1]
betanr  = NR1step(Y,X,betaols)
