# Compute the maximum of the inverse (regularized) Christoffel function for the Fourier extension frame 
# and confirm the results by computing an approximation with linear oversampling.
# This script reproduces Figure 9 and 10.

using Pkg
Pkg.activate("env")
Pkg.instantiate()
using BasisFunctions, Plots, LinearAlgebra, LaTeXStrings, Random, Statistics

# For the computation of the exact Christoffel function, we need very high precision
setprecision(BigFloat,256) 

# compute gram matrix
function fe_gram_matrix(Phi,W)
    n = length(Phi)
    T = prectype(Phi)
    Tpi = T(π)
    G = zeros(Complex{T}, n, n)
    for i in 1:n
        for j in 1:n
            fi = native_index(Phi, i)
            fj = native_index(Phi, j)
            A = 2*Tpi*im*(fi-fj)
            if abs(A) > 0
                G[i,j] = (exp(A*W) - exp(-A*W)) / A
            else
                G[i,j] = 2W
            end
        end
    end
    G
end

# computes the maximum of the inverse (regularized) christoffel function
function maxchristoffel(n,W)
    @show(n)
    T = typeof(W)
    Tpi = T(π)
    Phi0 = rescale(Fourier{T}(n), -one(T)/2, one(T)/2)
    D = Diagonal([one(T) / Phi0[k](-one(T)/2) * exp(-Tpi*im*k) for k in ordering(Phi0)])
    Phi = D*Phi0
    G = fe_gram_matrix(Phi,W) 
    F = svd!(G)
    if (minimum(F.S) < 10*eps(typeof(W))) 
        throw("Precision does not suffice.")    # Check whether precision suffices for the computation of the exact Christoffel function
    end
    xx = -W:0.0001:W
    k = 0*xx
    A = BasisFunctions.leastsquares_matrix(Phi,xx)

    # k1: inverse regularized christoffel function related to single precision
    # k2: inverse regularized christoffel function related to double precision
    # k3: inverse christoffel function
    eps1 = 100*eps(Float32)
    eps2 = 100*eps(Float64)
    k1 = 0*xx 
    k2 = 0*xx
    k3 = 0*xx
    for l = 1:n
        k1 = k1 + (abs.(A*F.V[:,l])).^2/(F.S[l] + eps1^2)
        k2 = k2 + (abs.(A*F.V[:,l])).^2/(F.S[l] + eps2^2)
        k3 = k3 + (abs.(A*F.V[:,l])).^2/(F.S[l])
    end
    [maximum(k1), maximum(k2), maximum(k3)]
end

########################################################################################    
##                             calculations for W = 0.3                               ##
########################################################################################    
nlist = 3:4:71

K1_1 = []
K2_1 = []
K3_1 = []

println("Computing the Christoffel functions for W = 0.3.")
@time for n = nlist 
    k1, k2, k3 = maxchristoffel(n,big(3)/10)
    append!(K1_1,k1)
    append!(K2_1,k2)
    append!(K3_1,k3)
end

eps1 = 100*eps(Float32)
eps2 = 100*eps(Float64)
FS = 14
p1 = plot(nlist,K3_1,label="",xlabel=L"n",linewidth=2, color=:black, ylims=(0, Inf),xtickfontsize=FS,ytickfontsize=FS,xguidefontsize=20,yguidefontsize=FS,xlim=(0,75))
scatter!(nlist,K1_1,label="",color=:black,shape=:circle,markersize=5)
scatter!(nlist,K2_1,label="", color=:black,shape=:square)
plot!(nlist[end-12:end],5*log10(1/eps1)*nlist[end-12:end] .- 50,style=:dash, color=:black, linewidth=1.5, label="")
plot!(nlist[end-5:end],5*log10(1/eps2)*nlist[end-5:end] .- 650,style=:dash, color=:black, linewidth=1.5, label="")
savefig(p1,"uniformsampling1.pdf")

########################################################################################    
##                             calculations for W = 0.1                               ##
########################################################################################    
nlist = 3:2:41 

# This computation takes a couple of minutes on my laptop
setprecision(BigFloat,256)
K1_2 = [] 
K2_2 = []
K3_2 = []

println("Computing the Christoffel functions for W = 0.1.")
@time for n = nlist 
    k1, k2, k3 = maxchristoffel(n,big(1)/10)
    append!(K1_2,k1)
    append!(K2_2,k2)
    append!(K3_2,k3)
end

FS = 14
p2 = plot(nlist,K3_2,label="",xlabel=L"n",linewidth=2, color=:black, ylims=(0, Inf),xtickfontsize=FS,ytickfontsize=FS,xguidefontsize=20,yguidefontsize=FS,xlim=(0,45))
scatter!(nlist,K1_2,label="",color=:black,shape=:circle,markersize=5)
scatter!(nlist,K2_2,label="", color=:black,shape=:square)
plot!(nlist[5:end],5*log10(1/eps1)*nlist[5:end] .+ 250,style=:dash, color=:black, linewidth=1.5, label="")
plot!(nlist[10:end],5*log10(1/eps2)*nlist[10:end] .+ 600,style=:dash, color=:black, linewidth=1.5, label="")
savefig(p2,"uniformsampling2.pdf")

########################################################################################    
##                                  Approximations                                    ##
########################################################################################   
println("Computing the numerical approximations.")
nlist = (2:2:18).^2
epsilon = 100*eps(Float64)
oversamplingfactor1 = 3
oversamplingfactor2 = 10
reps = 10
quad_list = zeros(length(nlist),1)
lin_list1 = zeros(length(nlist),reps)
lin_list2 = zeros(length(nlist),reps)

W = .3
f = (x) -> 1/(1-3.2x)
errgrid = range(-W,W,5000)
F = f.(errgrid)

function tsvd(A,epsilon,f)
    U = nothing
    S = nothing
    V = nothing
    try
        U,S,V = svd(A)
    catch err
        if err isa LinearAlgebra.LAPACKException
            @warn "SVD DivideAndConquer failed, retrying with QRIteration" err
            U,S,V = svd(A; alg=LinearAlgebra.QRIteration())
        else
            rethrow(err)
        end
    end
    S .= (S .> epsilon) .* 1 ./ S
    c = V * (S .* (U' * f))
end

Random.seed!(1)
for i = 1:length(nlist)
    n = nlist[i]
    @show(n)
    T = typeof(W)
    Tpi = T(π)
    Phi0 = rescale(Fourier{T}(n), -one(T)/2, one(T)/2)
    D = Diagonal([one(T) / Phi0[k](-one(T)/2) * exp(-Tpi*im*k) for k in ordering(Phi0)])
    B = D*Phi0

    # quadratic oversampling
    samples = range(-W,W,maximum([2n^2,100]))
    A = BasisFunctions.leastsquares_matrix(B,samples)/sqrt(length(samples))
    c = tsvd(A,epsilon,f.(samples)/sqrt(length(samples)))
    approx = Expansion(B,c)
    quad_list[i] = norm(abs.(approx.(errgrid) - F))/sqrt(length(errgrid))

    for j = 1:reps
        # linear oversampling (factor of 3)
        samples = 2W*rand(oversamplingfactor1*n) .- W
        A = BasisFunctions.leastsquares_matrix(B,samples)/sqrt(length(samples))
        c = tsvd(A,epsilon,f.(samples)/sqrt(length(samples)))
        approx = Expansion(B,c)
        lin_list1[i,j] = norm(abs.(approx.(errgrid) - F))/sqrt(length(errgrid))

        # linear oversampling (factor of 15)
        samples = 2W*rand(oversamplingfactor2*n) .- W
        A = BasisFunctions.leastsquares_matrix(B,samples)/sqrt(length(samples))
        c = tsvd(A,epsilon,f.(samples)/sqrt(length(samples)))
        approx = Expansion(B,c)
        lin_list2[i,j] = norm(abs.(approx.(errgrid) - F))/sqrt(length(errgrid))
    end
end

# compute geometric mean and variance
mean_curve1      = 10 .^ mean(log10.(lin_list1), dims=2)
std_curve1       = std(log10.(lin_list1), dims=2)
curve_min1       = 10 .^ (log10.(mean_curve1) .- std_curve1)
curve_max1       = 10 .^ (log10.(mean_curve1) .+ std_curve1)
mean_curve2      = 10 .^ mean(log10.(lin_list2), dims=2)
std_curve2       = std(log10.(lin_list2), dims=2)
curve_min2       = 10 .^ (log10.(mean_curve2) .- std_curve1)
curve_max2       = 10 .^ (log10.(mean_curve2) .+ std_curve1)

FS = 12
p3 = plot(sqrt.(nlist),quad_list,yscale=:log10,xlabel=L"\sqrt{n}",ylabel=L"$L^2$ error",label="", ylims=(1e-13,1e1), xlims=(0,23), colour=:black, xtickfontsize=FS,ytickfontsize=FS,legendfontsize=FS,xguidefontsize=16, style=:dash, yguidefontsize=14, top_margin=5Plots.mm,yticks=[1e-10,1e-5,1e0],linewidth=1.2)
plot!(sqrt.(nlist),mean_curve1,shape=:circle,label="", colour=:black)
plot!(vcat(sqrt.(nlist), reverse(sqrt.(nlist))), vcat(curve_min1, reverse(curve_max1)), seriestype =:shape, color=:black, alpha=0.08, label="", linewidth=0)
plot!(sqrt.(nlist),mean_curve2,shape=:circle,label="", colour=:black)
plot!(vcat(sqrt.(nlist), reverse(sqrt.(nlist))), vcat(curve_min2, reverse(curve_max2)), seriestype =:shape, color=:black, alpha=0.08, label="", linewidth=0)
annotate!(20.5, 10^(-10.9),(L"m = 10n",14))
annotate!(20.3, 10^(-8.4),(L"m = 3n",14))
savefig(p3,"uniformsampling3.pdf")