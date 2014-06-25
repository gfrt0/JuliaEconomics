# Bradley J. Setzler
# JuliaEconomics.com
# Tutorial 3: Bootstrapping and Non-parametric p-values in Julia

srand(2)

using Distributions
using Optim

N=1000
K=3

genX = MvNormal(eye(K))
X = rand(genX,N)
X = X'
X_noconstant = X
constant = ones(N)
X = [constant X]

genEpsilon = Normal(0, 1)
epsilon = rand(genEpsilon,N)
trueParams = [0.1,0.5,-0.3,0.]
Y = X*trueParams + epsilon

function loglike(rho,y,x)
    beta = rho[1:4]
    sigma2 = exp(rho[5])
    residual = y-x*beta
    dist = Normal(0, sigma2)
    contributions = logpdf(dist,residual)
    loglikelihood = sum(contributions)
    return -loglikelihood
end


params0 = [.1,.2,.3,.4,.5]
function wrapLoglike(rho)
    return loglike(rho,Y,X)
end
optimum = optimize(wrapLoglike,params0,method=:cg)
MLE = optimum.minimum
MLE[5] = exp(MLE[5])


M=convert(Int,N/2)
B=1000
samples = zeros(B,5)

for b=1:B
    theIndex = randperm(N)[1:M]
    x = X[theIndex,:]
    y = Y[theIndex,:]
    function wrapLoglike(rho)
        return loglike(rho,y,x)
    end
    samples[b,:] = optimize(wrapLoglike,params0,method=:cg).minimum
end
samples[:,5] = exp(samples[:,5])

bootstrapSE = std(samples,1)

nullDistribution = samples
pvalues = ones(5)
for i=1:5
    nullDistribution[:,i] = nullDistribution[:,i]-mean(nullDistribution[:,i])
end
nullDistribution[:,5] = 1 + nullDistribution[:,5]

pvalues = [mean(abs(MLE[i]).<abs(nullDistribution[:,i])) for i=1:5]
