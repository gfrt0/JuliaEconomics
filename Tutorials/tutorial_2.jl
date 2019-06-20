# Bradley J. Setzler
# JuliaEconomics.com
# Tutorial 2: Maximum Likelihood Estimation (MLE) in Julia: The OLS Example
# Passed test on Julia 1.1.1 (Giuseppe Forte edit 200619)

using Pkg, Distributions, Random, LinearAlgebra, Optim, ForwardDiff

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

function loglike(rho)
    beta = rho[1:4]
    sigma2 = exp(rho[5])
    residual = Y-X*beta
    dist = Normal(0, sigma2)
    contributions = logpdf.(dist,residual)
    loglikelihood = sum(contributions)
    #println(-loglikelihood)
    return -loglikelihood
end

# Optimisation
params0 = [.1,.2,.3,.4,.5]
optimum = optimize(loglike,params0,ConjugateGradient())
MLE_CG = Optim.minimizer(optimum)
MLE_CG[5] = exp(MLE_CG[5])
println(MLE_CG)

# Trying another algorithm
optimum = Optim.optimize(loglike,params0,NelderMead())
MLE_NM = Optim.minimizer(optimum)
MLE_NM[5] = exp(MLE_NM[5])
println(MLE_NM)

# Generating the Variance-Covariance matrix as the inverse of the numerical H(β_MLE)
#ForwardDiff.gradient(loglike, MLE)
vcovmat_NM = inv(ForwardDiff.hessian(loglike, MLE_NM))
vcovmat_CG = inv(ForwardDiff.hessian(loglike, MLE_CG))
println(vcovmat_CG)
println(vcovmat_NM)

# Asymptotically N(0,1) t-tests

t_NM = MLE_NM ./ sqrt.(diag(vcovmat_NM))
t_CG = MLE_CG ./ sqrt.(diag(vcovmat_CG))

p_NM = 2*(1 .- cdf.(Normal(), abs.(t_NM)))
p_CG = 2*(1 .- cdf.(Normal(), abs.(t_CG)))
