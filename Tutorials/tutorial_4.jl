# Bradley J. Setzler
# JuliaEconomics.com
# Tutorial 4: Stepdown p-values for Multiple Hypothesis Testing in Julia
# Passed test on Julia 1.1.1 (Giuseppe Forte edit 220619)

using Distributions, Random, LinearAlgebra, Optim, ForwardDiff, Statistics

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
trueParams = [0.01,0.05,0.05,0.07]
Y = X*trueParams + epsilon


function Holm(p)
    K = length(p)
    sort_index = -ones(K)
    sorted_p = sort(p)
    sorted_p_adj = sorted_p.*[K+1-i for i=1:K]
    for j=1:K
        num_ties = length(sort_index[(p.==sorted_p[j]) .& (sort_index.<0)])
        sort_index[(p.==sorted_p[j]) .& (sort_index.<0)] = j:(j-1+num_ties)
    end
    sorted_holm_p = [minimum([maximum(sorted_p_adj[1:k]),1]) for k=1:K]
    holm_p = [sorted_holm_p[Int.(sort_index[k])] for k=1:K]
    return holm_p
end


function stepdown(MLE,bootSamples)
    K = length(MLE)
    tMLE = MLE
    bootstrapSE = std(bootSamples, dims=1)
    tNullDistribution = bootSamples
    p = -ones(K)
    for i=1:K
        tMLE[i] = abs(tMLE[i]/bootstrapSE[i])
        tNullDistribution[:,i] = abs.((tNullDistribution[:,i] .- mean(tNullDistribution[:,i]))/bootstrapSE[i])
        p[i] = mean(tMLE[i].<tNullDistribution[:,i])
    end
    sort_index = stepdown_p = -ones(K)
    sorted_p = sort(p)
    for j=1:K
        num_ties = length(sort_index[(p.==sorted_p[j]) .& (sort_index.<0)])
        sort_index[(p.==sorted_p[j]) .& (sort_index.<0)] = j:(j-1+num_ties)
        sort_index = Int.(sort_index)
    end
    for k=1:K
    	current_index = [sort_index.>=k]
        stepdown_p[sort_index[k]] = mean(maximum(tNullDistribution[:,sort_index.>=k], dims = 2).>tMLE[sort_index[k]])
    end
    return ["single_pvalues"=>p,"stepdown_pvalues"=>stepdown_p,"Holm_pvalues"=>Holm(p)]
end







function loglike(rho,y,x)
    beta = rho[1:(K+1)]
    sigma2 = exp(rho[K+2])
    residual = y-x*beta
    dist = Normal(0, sigma2)
    contributions = logpdf.(dist,residual)
    loglikelihood = sum(contributions)
    return -loglikelihood
end


params0 = [.1,.2,.3,.4,.5]
function wrapLoglike(rho)
    return loglike(rho,Y,X)
end
optimum = optimize(wrapLoglike,params0,ConjugateGradient())
MLE_CG = Optim.minimizer(optimum)
MLE_CG[K+2] = exp(MLE_CG[K+2])
println(MLE_CG)


B=1000
samples = zeros(B,(K+2))

for b=1:B
    theIndex = sample(1:N,N)
    xB = X[theIndex,:]
    yB = Y[theIndex,:]
    function wrapLoglike(rho)
        return loglike(rho,yB,xB)
    end
    samples[b,:] = optimize(wrapLoglike,params0,ConjugateGradient()).minimizer
end
samples[:,(K+2)] = exp.(samples[:,(K+2)])


stepdown(MLE_CG[1:(K+1)],samples[:,1:(K+1)])
