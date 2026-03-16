# This script computes the effective dimension of a one-dimensional Fourier extension frame.
# This script reproduces Figure 8.

using Pkg
Pkg.activate("env")
Pkg.instantiate()
using BasisFunctions, Plots, LaTeXStrings, LinearAlgebra

# Compute Fourier extension gram matrix
function compute_effective_dimension(n, W, epsilon)
    T = prectype(W)
    Tpi = T(π)
    G = zeros(Complex{T}, n, n)
    f = Int((n-1)/2)
    freqlist = -f:1:f
    for i in 1:n
        fi = freqlist[i]
        for j in 1:n
            fj = freqlist[j]
            if abs(fi - fj) > 0
                G[i,j] = sin(2 * Tpi * (fi - fj) * W)/(Tpi * (fi - fj))
            else
                G[i,j] = 2*W
            end
        end
    end
    F = svd!(G)
    effdim = sum(F.S ./(F.S .+ epsilon^2))
end

setprecision(BigFloat,128) # the unit roundoff should be significantly smaller than the regularization parameter
Wlist = range(big(1)/20,big(9)/20,9)
n = 141
########################################################################################    
##                     calculations for double precision                              ##
########################################################################################   
u_dp = 100*eps(Float64)
doflist_dp = compute_effective_dimension.(n,Wlist,u_dp)

FS = 14
p1 = plot(Wlist,doflist_dp,label="",ylabel=L"n^\epsilon",shape=:circle,xlabel=L"W",seriescolor=:black, ylims=(0,200))
plot!(Wlist,ceil.(2*n*Wlist) .+ 2 .+ 2/pi^2*(log(8/u_dp^2) + 1)*log(4n),style=:dash,label="",seriescolor=:black,xtickfontsize=FS,ytickfontsize=FS,legendfontsize=FS,xguidefontsize=16, yguidefontsize=16)
plot!(Wlist,141*ones(length(Wlist)),linecolor=:black,label="",style=:dash)
annotate!(0.26,175,Plots.text("theoretical bound", rotation = 21, pointsize = 12))
annotate!(0.09,150,Plots.text(L"n = 141", pointsize = 14))
savefig(p1,"effectivedimension1.pdf")

########################################################################################    
##                     calculations for single precision                              ##
########################################################################################   
u_sp = 100*eps(Float32)
doflist_sp = compute_effective_dimension.(n,Wlist,u_sp)

FS = 14
p2 = plot(Wlist,doflist_sp,label="",ylabel=L"n^\epsilon",shape=:circle,xlabel=L"W",seriescolor=:black, ylims=(0,200))
plot!(Wlist,ceil.(2*n*Wlist) .+ 2 .+ 2/pi^2*(log(8/u_sp^2) + 1)*log(4n),style=:dash,label="",seriescolor=:black,xtickfontsize=FS,ytickfontsize=FS,legendfontsize=FS,xguidefontsize=16, yguidefontsize=16)
plot!(Wlist,141*ones(length(Wlist)),linecolor=:black,label="",style=:dash)
annotate!(0.12,84,Plots.text("theoretical bound", rotation = 21, pointsize = 12))
annotate!(0.09,150,Plots.text(L"n = 141", pointsize = 14))
savefig(p2,"effectivedimension2.pdf")