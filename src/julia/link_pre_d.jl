using StatsBase, SparseArrays
include("graph.jl")



function generate_sample(graph, test_ratio=0.3)
    edges = [(graph.u[i], graph.v[i]) for i in 1:length(graph.u)]

    
    num_test_edges = Int(floor(length(edges) * test_ratio))
    test_positive_edges = StatsBase.sample(edges, num_test_edges, replace=false)


    training_edges = setdiff(edges, test_positive_edges)


    test_negative_edges=[]
    existing_edges = Set(edges)
    while(length(test_negative_edges)<num_test_edges)
        u=rand(1:graph.n)
        v=rand(1:graph.n)
        if u!=v && !in((u,v),existing_edges)
            push!(test_negative_edges,(u,v))
            push!(existing_edges,(u,v))
        end
    end


    test_edges = vcat([(edge, 1) for edge in test_positive_edges], [(edge, 0) for edge in test_negative_edges])

    return training_edges, test_edges
end

function evaluate(test_edges, Q)
    scores = [((u, v), Q[u, v], label) for ((u, v), label) in test_edges]
    sort!(scores, by=x -> x[2], rev=true)

    top_k_count = Int(floor(length(scores) * 0.5))
    top_k_edges = scores[1:top_k_count]

    correct_positive = count(x -> x[3] == 1, top_k_edges)
    total_samples = length(top_k_edges)
    precision = correct_positive / total_samples

    println("correct positive: ", correct_positive)
    println("wrong positive: ", total_samples - correct_positive)
    println("total samples: ", total_samples)

    return precision
end

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
        # println("ll = ", ll)
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

file="wiki-u.txt"
g=read_data(file,false)
@show g.n,g.m
n=g.n
m=g.m
train,test=generate_sample(g, 0.3)

g1=add_edge!(n,train)
for k in 1:1
    global Q,Q2,pre,ll
    ll=1000
    Q=approxK(k*ones(n),g1,ll)
    for (i, j) in zip(findnz(Q)...)
        Q[i, j] = log10(Q[i, j] * 2 * ll * maximum(k))
        if Q[i, j] <= 0
            Q[i, j] = 0  
        end
    end
    pre=evaluate(test,Q)
    println("k = ", k, ", precision = ", pre)
end
