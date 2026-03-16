# Compute the maximum of the inverse (regularized) Christoffel function for the Fourier extension frame.
# This script reproduces Figure 6(a).
using Pkg
Pkg.activate("env")
Pkg.instantiate()
using BasisFunctions, Plots, LaTeXStrings, LinearAlgebra

function fe_gram_matrix(Phi)
    n = length(Phi)
    T = prectype(Phi)
    Tpi = T(π)
    G = zeros(Complex{T}, n, n)
    for i in 1:n
        fi = native_index(Phi, i)
        for j in 1:n
            fj = native_index(Phi, j)
            A = im * Tpi * (fi - fj) / 2
            if abs(fi - fj) > 0
                G[i,j] = (exp(A) - exp(-A)) / A
            else
                G[i,j] = 2
            end
        end
    end
    return G
end

setprecision(BigFloat,256)
T = typeof(big(0.))
n = 101
Phi0 = rescale(Fourier{T}(n), -2one(T), 2one(T))
ks = collect(ordering(Phi0))  
D = Diagonal(exp.(-im*T(pi) .* ks))
Phi = D * Phi0
G = fe_gram_matrix(Phi) 
F = svd!(G)
if (minimum(F.S) < 10*eps(T)) 
    throw("Precision does not suffice.")
end
λ1 = 10^(-14)

xx = -1:0.001:1
A = BasisFunctions.leastsquares_matrix(Phi,xx)
k_exact = zeros(Float64,length(xx),1)
k_num = zeros(Float64,length(xx),1)
for l = 1:n
    k_exact .+= abs.(A*F.V[:,l]).^2/(F.S[l])
    k_num .+= abs.(A*F.V[:,l]).^2/(F.S[l] + λ1^2)
end
p1 = plot(xx,k_exact,linewidth=2,label="",seriescolor=:blue,xtickfontsize=14,ytickfontsize=14, xguidefontsize=16,yscale=:log10)
plot!(xx,k_num,ylims=(1,Inf),linewidth=2,label="",seriescolor=:red)
xlabel!(L"x")
yticks!([1,10,100,1000])
annotate!(0, 10^(1.8),(L"k",18,:blue))
annotate!(0, 10^(1.2),(L"k^\epsilon",18,:red))
savefig(p1,"christoffel_fe.pdf")

xx = 10 .^(-6.5:0.01:0) .- 1
A = BasisFunctions.leastsquares_matrix(Phi,xx)
k_exact = zeros(Float64,length(xx),1)
k_num = zeros(Float64,length(xx),1)
for l = 1:n
    k_exact .+= abs.(A*F.V[:,l]).^2/(F.S[l])
    k_num .+= abs.(A*F.V[:,l]).^2/(F.S[l] + λ1^2)
end
p2 = plot(1 .+ xx,k_exact,linewidth=2,label="",seriescolor=:blue,xtickfontsize=14,ytickfontsize=14, xguidefontsize=12,xscale=:log10)
plot!(1 .+ xx,k_num,ylims=(1,Inf),linewidth=2,label="",seriescolor=:red)
xlabel!("distance to -1")
annotate!(10^(-3), 900,(L"k",18,:blue))
annotate!(10^(-3.3), 300,(L"k^\epsilon",18,:red))
savefig(p2,"christoffel_fe2.pdf")

numdim = sum([F.S[l]/(F.S[l] + λ1^2) for l = 1:n])