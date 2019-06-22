# Bradley J. Setzler
# JuliaEconomics.com
# Tutorial 3: Bootstrapping and Non-parametric p-values in Julia
# Passed test on Julia 1.1.1 (Giuseppe Forte edit 220619)

using Distributions, Random, LinearAlgebra, Optim, ForwardDiff, Statistics

Random.seed!(2)

N=1000
K=4
σ2index = K + 2

genX = MvNormal(Matrix{Float64}(I, K, K))
X = rand(genX,N)
X = X'
X_noconstant = X
constant = ones(N)
X = [constant X]

genEpsilon = Normal(0, 1)
epsilon = rand(genEpsilon,N)
trueParams = [0.01,0.05,0.05,0.07,-0.02]
Y = X*trueParams + epsilon

function loglike(rho, Y, X)
    beta = rho[1:5]
    sigma2 = exp(rho[σ2index])
    residual = Y-X*beta
    dist = Normal(0, sigma2)
    contributions = logpdf.(dist,residual)
    loglikelihood = sum(contributions)
    #println(-loglikelihood)
    return -loglikelihood
end


params0 = [.1,.2,.3,.4,.5,.6]
function wrapLoglike(rho)
    return loglike(rho,Y,X)
end
optimum = optimize(wrapLoglike,params0,ConjugateGradient())
MLE_CG = Optim.minimizer(optimum)
MLE_CG[σ2index] = exp(MLE_CG[σ2index])
println(MLE_CG)

vcovmat_CG = inv(ForwardDiff.hessian(wrapLoglike, MLE_CG))
# Asymptotically N(0,1) t-tests: test if coefficients are equal to zero and if σ2 is 1:
t_CG = (MLE_CG .- [0, 0, 0, 0, 0, 1]) ./ sqrt.(diag(vcovmat_CG))
p_CG = 2*(1 .- cdf.(Normal(), abs.(t_CG)))


# Bootstrap
B=1000
samples = zeros(B,σ2index)

for b=1:B
    theIndex = sample(1:N,N)
    x = X[theIndex,:]
    y = Y[theIndex,:]
    function wrapLoglike(rho)
        return loglike(rho,y,x)
    end
    # alternative: take the MLE estimates as starting values for optimizer
    samples[b,:] = optimize(wrapLoglike,params0,ConjugateGradient()).minimizer
end
samples[:,σ2index] = exp.(samples[:,σ2index])

# Center the empirical distributions around 0
nullDistribution = samples
for i=1:σ2index
    nullDistribution[:,i] = nullDistribution[:,i] .- mean(nullDistribution[:,i])
end

# Bootstrap SE from Hansen (2019, pg. 337)
bootstrapSE = sqrt.( 1/(B-1) .* sum(nullDistribution.^2, dims=1))
# or equivalently
bootstrapSE = std(nullDistribution, dims = 1)

pvalues = [mean(abs.(MLE_CG[i]) .< abs.(nullDistribution[:,i])) for i=1:σ2index]
