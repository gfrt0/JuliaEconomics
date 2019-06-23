# Bradley J. Setzler
# JuliaEconomics.com
# Tutorial 5: Parallel Processing in Julia: Bootstrapping the MLE
# Passed test on Julia 1.1.1 (Giuseppe Forte edit 230619)


using Pkg, Distributions, DataFrames, Random, LinearAlgebra, Optim, ForwardDiff, Statistics, CSV, Distributed

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

data = DataFrame(hcat(Y,X))
names!(data,[:Y,:one,:X1,:X2,:X3])
CSV.write("/Users/gforte/Downloads/data.csv",data)

if length(procs()) == 1
    addprocs(3)
end

@everywhere include("/Users/gforte/Dropbox/git/JuliaEconomics/Tutorials/tutorial_5_bootstrapFunction.jl")

B=Int.(1000)
b=Int.(B/4)
samples_pmap = pmap(bootstrapSamples,[b,b,b,b])
samples = vcat(samples_pmap[1],samples_pmap[2],samples_pmap[3],samples_pmap[4])
# @elapsed pmap(bootstrapSamples,[b,b,b,b])
# @elapsed bootstrapSamples(B)
