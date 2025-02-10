

include("graph.jl")
using LinearAlgebra,StatsBase,Arpack,RandomizedLinAlg
using CSV, DataFrames,DelimitedFiles
using MatrixMarket, SparseArrays

function approxK(k::Vector{Float64},graph::Graph, sample_num::Int)
    n = graph.n
    d = degree_vector(graph)

    in_forests = falses(n)
    next_node = zeros(Int, n)
    root = zeros(Int, n)
    nbr,_ = neighbor_weight(graph)

    row=Int[]
    col=Int[]
    val=Float64[]
    for ll in 1:sample_num
        in_forests .= false
        next_node .= -1
        root .= -1
        
        for src in 1:n
            u = src
            while !in_forests[u]
                if rand() * (d[u] + k[u]) < k[u]
                    in_forests[u] = true
                    root[u] = u
                    push!(row, u)
                    push!(col, u)
                    push!(val, 1 / sample_num/k[u])
                    break
                end
                next_node[u] = rand(nbr[u])
                u = next_node[u]
            end

            r = root[u]
            u = src
            while !in_forests[u]
                in_forests[u] = true
                root[u] = r
                push!(row, u)
                push!(col, r)
                push!(val, 1 / sample_num/k[r])
                u = next_node[u]
            end
        end

    end
    ans=sparse(row, col, val, n, n)
    return ans
end

file="wiki.txt"
g=read_data(file,true)
@show g.n,g.m
n=g.n
m=g.m


# L=laplacian_matrix(g)
# @time exactQ=inv(Matrix(I+L))

ll=1000
#ARL-F
k=ones(n)
# ARL-P
# k=5*degree_vector(g)
# ARL-K
# dmax=maximum(degree_vector(g))
# k=(dmax+1)*ones(n)-degree_vector(g)
@time Q=approxK(k,g,ll)
@show norm(exactQ-Q)/norm(exactQ)

for (i, j) in zip(findnz(Q)...)
    Q[i, j] = log10(Q[i, j] * 2 * ll * maximum(k))
    if Q[i, j] <= 0
        Q[i, j] = 0  
    end
end

d=128
println("begin svd")

@time ur,sr,vr=svds(Q,nsv=d)[1]
zs=ur*sqrt.(diagm(sr))
zt=vr*sqrt.(diagm(sr))

norm1=norm(sr)
norm2=norm(Q)
println("svd_error:", norm1/norm2)

file_zs = string("data/embs/", file[1:end-4], "-ds.csv")
writedlm(file_zs, round.(zs, digits=6), ',')
file_zt=string("data/embs/", file[1:end-4], "-dt.csv")
writedlm(file_zt, round.(zt, digits=6), ',')