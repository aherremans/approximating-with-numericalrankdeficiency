# This script shows how regularization relaxes the discretization condition.
# Specifically, we use Legendre points and exponentially clustered points to approximate in the Legendre + weighted Legendre basis
# and compare gamma_reg to gamma_exact.
# This script reproduces Figure 5 and Figure 6(b).

using Pkg
Pkg.activate("env")
Pkg.instantiate()

using BasisFunctions, Plots, LaTeXStrings, FastTransforms, ColorSchemes, GenericLinearAlgebra, LinearAlgebra, Bessels
import BasisFunctions: ⊕

pyplot()    # use python plot backend

# Here, I set some parameters. 
# Note that we use higher precision to compute gamma_exact and gamma_reg, while we use standard Float64 to compute
# the numerical approximation. 
F = Float64
T = BigFloat
setprecision(T,512) 
m1list = 2:8:98                 # number of Legendre points
m2list = 2:4:42                 # number of exponentially clustered points
n = 80

########################################################################################    
##                               compute Gram matrix                                  ##
######################################################################################## 
# This piece of code computes the Gram matrix efficiently and in high precision using FastTransforms.jl.
# We will need this to compute gamma_exact gamma_reg

b = BasisFunctions.normalize(Legendre{T}(Int(n/2)))

println("Computing Gram matrix in high precision.")
G = zeros(T,n,n)
for i = 1:n 
    for j = i:n
        if (i <= Int(n/2) && j <= Int(n/2))
            # standard Legendre inner product
            G[i,j] = BasisFunctions.gramelement(b,i,j)
        elseif (i <= Int(n/2) && j > Int(n/2))
            # convert to Jacobi polynomials for efficient computation of inner products with weight function
            v1 = jac2jac(T.(I[1:Int(n/2),i]),0,0,0,1/2; norm1=true, norm2=true)
            v2 = jac2jac(T.(I[1:Int(n/2),j-Int(n/2)]),0,0,0,1/2; norm1=true, norm2=true)
            G[i,j] = v1'*v2/sqrt(T(2.))
        else 
            # convert to Jacobi polynomials for efficient computation of inner products with weight function
            v1 = jac2jac(T.(I[1:Int(n/2),i-Int(n/2)]),0,0,0,1; norm1=true, norm2=true)
            v2 = jac2jac(T.(I[1:Int(n/2),j-Int(n/2)]),0,0,0,1; norm1=true, norm2=true)
            G[i,j] = v1'*v2/T(2.)
        end
        if (j != i)
            G[j,i] = conj(G[i,j])
        end
    end
end

########################################################################################    
##                                 compute gamma's                                    ##
######################################################################################## 
# This computation takes a long time (approx. 18 min on my laptop).
# You can also use precomputed values via
# gamma_reg = readdlm("precomputed/gamma_reg.log")
# gamma_exact = readdlm("precomputed/gamma_exact.log")

gamma_reg = zeros(T,length(m1list),length(m2list))
gamma_exact = zeros(T,length(m1list),length(m2list))

w = x -> sqrt((T(1)+x)/T(2))
frame = BasisFunctions.normalize(Legendre{T}(Int(n/2))) ⊕ w*BasisFunctions.normalize(Legendre{T}(Int(n/2)))
epsilon = T(10)^(-14)  # regularization parameter

println("Computing gamma's in high precision. (This takes about twelve minutes.)")
@time for i = 1:length(m1list)
    Aleg = Matrix(evaluation(frame,LegendreNodes(m1list[i])))
    for j = 1:length(m2list)
        @show (i,j)
        Aexp = Matrix(evaluation(frame,-1 .+ 10 .^range(-10,0,m2list[j])))
        Gd = [Aleg; Aexp]'*[Aleg; Aexp]*T(2.)/(m1list[i]+m2list[j])

        # Compute gamma_reg and gamma_exact via generalized eigenvalue problem.
        # We use a custom branch of GenericLinearAlgebra.jl (dh/geneig) to support
        # generalized eigenvalue problems with BigFloat matrices.
        e = minimum(real.(eigvals(Gd + epsilon^2*I, G)))
        gamma_reg[i,j] = e > 0 ? sqrt(e) : NaN
        e2 = minimum(real.(eigvals(Gd, G)))
        gamma_exact[i,j] = e2 > 0 ? sqrt(e2) : NaN
    end
end

X = -2:125
E = eigvals(G)
nl = sum( E ./ (E .+ epsilon^2) )

# (pyplot is not compatible with BigFloat, so we convert the final result to Float64.)
p1 = heatmap(m1list,m2list,F.(log10.(1 ./gamma_exact')), xlabel="# Legendre points", ylabel="# exponentially clustered points", clim=(0,14), size=(570,410),c=cgrad(:matter), tickfontsize=10,cbar_title=L"1 / \gamma_{exact}",colorbar_ticks=([0, 5, 10, 14], [L"1",L"10^{5}",L"10^{10}",L"\geq 10^{14}"]),colorbar_tickfontsize=12,colorbar_titlefontsize=12,rightmargin=3Plots.mm,leftmargin=3Plots.mm,bottommargin=3Plots.mm,ylabelfontsize=14,xlabelfontsize=14)
plot!([(8*x[1]-6.65,4*x[2]-2) for x in Tuple.(findall(isnan,gamma_exact))], marker=:square, markercolor=:black, markersize=22, xlims=xlims(p1), ylims=ylims(p1), label="")
plot!(X,n .- X,colour=:white, linewidth=3, xlims=xlims(p1), ylims=ylims(p1),label="")
annotate!(33,40, text(L"\hat{n}", 20, :white))
plot!(X,nl .- X,colour=:white, linewidth=3, xlims=xlims(p1), ylims=ylims(p1),label="")
annotate!(11,34, text(L"\hat{n}^\epsilon", 20, :white))
savefig(p1,"weightedlegendre_gammaexact.pdf")

p2 = heatmap!(m1list,m2list,F.(log10.(1 ./gamma_reg')), xlabel="# Legendre points", ylabel="# exponentially clustered points", size=(570,410),clim=(0,14),colorbar_ticks=([0, 5, 10, 14], [L"1",L"10^{5}",L"10^{10}",L"\geq 10^{14}"]),c=cgrad(:matter),tickfontsize=10,colorbar_tickfontsize=12,cbar_title=L"1 / \gamma_{reg}",colorbar_titlefontsize=12,rightmargin=3Plots.mm,leftmargin=3Plots.mm,bottommargin=3Plots.mm,ylabelfontsize=14,xlabelfontsize=14)
plot!(X,n .- X,colour=:white, linewidth=3, xlims=xlims(p2), ylims=ylims(p2),label="")
annotate!(33,40, text(L"\hat{n}", 20, :white))
plot!(X,nl .- X,colour=:white, linewidth=3, xlims=xlims(p1), ylims=ylims(p1),label="")
annotate!(11,34, text(L"\hat{n}^\epsilon", 20, :white))
savefig(p2,"weightedlegendre_gammareg.pdf")

########################################################################################    
##                        compare with approximation error                            ##
######################################################################################## 
# We compare the constants gamma_reg and gamma_exact to error of a numerical approximation
# in standard floating-point precision (i.e., Float64).

approxerr = zeros(F,length(m1list),length(m2list))
epsilon = F(10)^(-14)  # regularization parameter

w = x -> sqrt((F(1)+x)/F(2))
frame = BasisFunctions.normalize(Legendre{F}(Int(n/2))) ⊕ w*BasisFunctions.normalize(Legendre{F}(Int(n/2)))
f = x -> besselj(1/2,x+1) + 1 ./(x^2+1)

# independent error grid
errgrid1 = LegendreNodes(1000)
errgrid2 = -1 .+ 10 .^range(-10,0,1000)
errgrid = [errgrid1; errgrid2]
Aerr = Matrix(evaluation(frame,errgrid))
ferr = f.(errgrid)

println("Computing numerical approximation in Float64.")
for i = 1:length(m1list)
    Aleg = Matrix(evaluation(frame,LegendreNodes(m1list[i])))
    for j = 1:length(m2list)
        @show (i,j)
        Aexp = Matrix(evaluation(frame,-1 .+ 10 .^range(-10,0,m2list[j])))
        samples = [LegendreNodes(m1list[i]); -1 .+ 10 .^range(-10,0,m2list[j])]
        A = [Aleg; Aexp] ./ sqrt(length(samples))
        b = f.(samples) ./ sqrt(length(samples))

        # TSVD approximation with truncation threshold = 10^(-14)
        U,S,V = svd(A)
        S .= (S .> epsilon) .* 1 ./S
        c = V * (diagm(S) * ( U'*b ))
        approxerr[i,j] = maximum(abs.(ferr - Aerr*c))
    end
end

X = -2:125
E = eigvals(G)
effdim = sum( E ./ (E .+ epsilon^2) )

p3 = heatmap(m1list,m2list,log10.(approxerr'), xlabel="# Legendre points", ylabel="# exponentially clustered points", c=:matter,tickfontsize=10, size=(570,410),rightmargin=3Plots.mm,leftmargin=3Plots.mm,bottommargin=3Plots.mm,colorbar_ticks=([-14,-10,-5,0], [L"\leq 10^{-14}",L"10^{-10}",L"10^{-5}",L"\geq 1"]),colorbar_tickfontsize=12,cbar_title="uniform approximation error",colorbar_titlefontsize=12,clim=(-14,0),ylabelfontsize=14,xlabelfontsize=14)
plot!(X,n .- X,colour=:white, linewidth=3, xlims=xlims(p3), ylims=ylims(p3),label="")
annotate!(33,40, text(L"\hat{n}", 20, :white))
plot!(X,effdim .- X,colour=:white, linewidth=3, xlims=xlims(p3), ylims=ylims(p3),label="")
annotate!(11,34, text(L"\hat{n}^\epsilon", 20, :white))
savefig(p3,"weightedlegendre_approxerr.pdf")


########################################################################################    
##                             plot Christoffel functions                             ##
######################################################################################## 
# We compute the Christoffel function to the regularized variant. Both are computed in high precision.
println("Computing Christoffel functions in high precision.")

gr() # use standard plotting backend

epsilon = T(10)^(-14)  # regularization parameter
w = x -> sqrt((T(1)+x)/T(2))
frame = BasisFunctions.normalize(Legendre{T}(Int(n/2))) ⊕ w*BasisFunctions.normalize(Legendre{T}(Int(n/2)))
SVD = svd(G)

xx = -1:0.001:1
k_exact = zeros(T,length(xx))
k_reg = zeros(T,length(xx))
A = BasisFunctions.leastsquares_matrix(frame,xx)
for l = 1:n
    k_exact .+= (abs.(A*SVD.V[:,l])).^2/(SVD.S[l])
    k_reg .+= (abs.(A*SVD.V[:,l])).^2/(SVD.S[l] + epsilon^2)
end
p4 = plot(xx,k_exact,yscale=:log10,linewidth=2,label="",seriescolor=:blue,xtickfontsize=14,ytickfontsize=14, xguidefontsize=16)
plot!(xx,k_reg,ylims=(1,Inf),linewidth=2,label="",seriescolor=:red)
yticks!([1,10^2,10^4,10^6])
annotate!(0, 10^(1.7),(L"k",18,:blue))
annotate!(0, 10^(0.8),(L"k^\epsilon",18,:red))
xlabel!(L"x")
savefig(p4,"weightedlegendre_christoffel1.pdf")

xx = 10 .^(-9:0.01:0) .- 1
k_exact = zeros(T,length(xx))
k_reg = zeros(T,length(xx))
A = BasisFunctions.leastsquares_matrix(frame,xx)
for l = 1:n
    k_exact .+= (abs.(A*SVD.V[:,l])).^2/(SVD.S[l])
    k_reg .+= (abs.(A*SVD.V[:,l])).^2/(SVD.S[l] + epsilon^2)
end
p5 = plot(xx .+ 1,k_exact,yscale=:log10,xscale=:log10,linewidth=2,label="",seriescolor=:blue,xtickfontsize=14,ytickfontsize=14,xguidefontsize=12)
plot!(xx .+ 1,k_reg,ylims=(1,Inf),linewidth=2,label="",seriescolor=:red)
yticks!([1,10^2,10^4,10^6])
xticks!([10^(-9),10^(-6),10^(-3),1])
annotate!(10^(-1), 10^(2.4),(L"k",18,:blue))
annotate!(10^(-1), 10^(1),(L"k^\epsilon",18,:red))
xlabel!("distance to -1")
savefig(p5,"weightedlegendre_christoffel2.pdf")