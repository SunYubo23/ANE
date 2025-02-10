using LinearAlgebra
using SparseArrays
using Laplacians

struct Graph
    n :: Int
    m :: Int
    u :: Vector{Int}
    v :: Vector{Int}
    w :: Vector{Float32}
    directed :: Bool
end

function add_edge!(n::Int,edges::Vector{Tuple{Int,Int}})
	u = Int[]
	v = Int[]  
	w = Float32[] 
	for edge in edges
		uu = edge[1]
		vv = edge[2]
		ww = 1.0
		push!(u, uu)
		push!(v, vv)
		push!(w, ww)
	end
	m = length(u)  # 边的数量
	return Graph(n, m, u, v, w, false)
end

function read_data(filename::AbstractString, directed::Bool = false)
    u = Int[] 
	v = Int[] 
	w = Float32[]  

	filename = "data/$filename" 
	lines = readlines(filename)
	
	n = parse(Int, strip(lines[1]))
	# @show n
	lines = lines[2:end] 

	for line in lines
		line = strip(line)
		if startswith(line, '#') || startswith(line, '%')
			continue
		end
		
		split_line = split(line)
		
		uu = parse(Int, split_line[1])
		vv = parse(Int, split_line[2])
		ww = length(split_line) < 3 ? 1.0 : parse(Float32, split_line[3])
		
		push!(u, uu+1)
		push!(v, vv+1)
		push!(w, ww)
	end

	m = length(u) 
    return Graph(n, m, u, v, w, directed)
end


function degree_matrix(G)
	u=zeros(G.n);
	d=zeros(G.n);
	for i=1:G.n
		u[i]=i
	end
	for i=1:G.m
		if G.w[i]>0
			if !G.directed
				d[G.v[i]]+=G.w[i];
			end
			d[G.u[i]]+=G.w[i];			
		elseif G.w[i]<0
			if !G.directed
				d[G.v[i]]-=G.w[i];
			end
			d[G.u[i]]-=G.w[i];
		end
	end
	return sparse(u,u,d)
end

function degree_vector(G)
	d=zeros(G.n);
	for i=1:G.m
		if G.w[i]>0
			if !G.directed
				d[G.v[i]]+=G.w[i];
			end
			d[G.u[i]]+=G.w[i];
		elseif G.w[i]<0
			if !G.directed
				d[G.v[i]]-=G.w[i];
			end
			d[G.u[i]]-=G.w[i];
		end
	end
	return d
end


function adjacency_matrix(G)
	n=G.n
	matrix = spzeros(Int, n, n)
	if G.directed
		for i in 1:G.m
			matrix[G.u[i], G.v[i]] = G.w[i]
		end
	else
		for i in 1:G.m
			matrix[G.u[i], G.v[i]] = G.w[i]
			matrix[G.v[i], G.u[i]] = G.w[i]
		end
	end
	return matrix
end

function laplacian_matrix(G :: Graph)
	d=zeros(G.n);
	for i=1:G.m
		if G.w[i]>0
			d[G.u[i]]+=G.w[i];
			d[G.v[i]]+=G.w[i];
		elseif G.w[i]<0
			d[G.u[i]]-=G.w[i];
			d[G.v[i]]-=G.w[i];
		end
	end
	uu=zeros(2*G.m+G.n);
	vv=zeros(2*G.m+G.n);
	ww=zeros(2*G.m+G.n);
	a=zeros(G.n);
	for i=1:G.n
		a[i]=i;
	end
	uu[1:G.m]=G.u;
	uu[G.m+1:2*G.m]=G.v;
	uu[2*G.m+1:2*G.m+G.n]=a;
	vv[1:G.m]=G.v;
	vv[G.m+1:2*G.m]=G.u;
	vv[2*G.m+1:2*G.m+G.n]=a;
	ww[1:G.m].=-G.w;
	ww[G.m+1:2*G.m].=-G.w;
	ww[2*G.m+1:2*G.m+G.n]=d;
    return sparse(uu,vv,ww)
end

function twice_incidence(G)
	u=zeros(4*G.m)
	v=zeros(4*G.m)
	w=zeros(4*G.m)
	for i=1:G.m
		if G.w[i]>0
			u[i]=i
			v[i]=G.u[i]
			w[i]=1
			u[G.m+i]=i
			v[G.m+i]=G.v[i]
			w[G.m+i]=-1
			u[2*G.m+i]=i+G.m
			v[2*G.m+i]=G.n+G.u[i]
			w[2*G.m+i]=-1
			u[3*G.m+i]=i+G.m
			v[3*G.m+i]=G.n+G.v[i]
			w[3*G.m+i]=1
		elseif G.w[i]<0
			u[i]=i
			v[i]=G.u[i]+G.n
			w[i]=1
			u[G.m+i]=i
			v[G.m+i]=G.v[i]
			w[G.m+i]=-1
			u[2*G.m+i]=i+G.m
			v[2*G.m+i]=G.u[i]
			w[2*G.m+i]=1
			u[3*G.m+i]=i+G.m
			v[3*G.m+i]=G.v[i]+G.n
			w[3*G.m+i]=-1
		end
	end
	return sparse(u,v,w)
end


function gremban_expansion(G::Graph)
	n = G.n
	m = G.m
	u = zeros(Int, 2*m)
	v = zeros(Int, 2*m)
	w = zeros(Float64, 2*m)
	for i in 1:G.m
		if G.w[i] > 0
			u[i] = G.u[i]
			v[i] = G.v[i]
			w[i] = G.w[i]
			u[i+m] = G.u[i]+n
			v[i+m] = G.v[i]+n
			w[i+m] = G.w[i]
		elseif G.w[i] < 0
			u[i] = G.u[i]
			v[i] = G.v[i]+n
			w[i] = -G.w[i]
			u[i+m] = G.u[i]+n
			v[i+m] = G.v[i]
			w[i+m] = -G.w[i]
		end
	end
	return Graph(2*n, 2*m, u, v, w,G.directed)
end


function neighbor_weight(G::Graph)
	n = G.n
	m = G.m
	nbr = Array{Array{Int, 1}}(undef, n)
	weight = Array{Array{Float64, 1}}(undef, n)
	for i in 1:n
		nbr[i] = Int[]
		weight[i] = Float64[]
	end
	for i in 1:m
		push!(nbr[G.u[i]], G.v[i])
		push!(nbr[G.v[i]], G.u[i])
		push!(weight[G.u[i]], abs(G.w[i]))
		push!(weight[G.v[i]], abs(G.w[i]))
	end
	return nbr, weight
end

function transition_matrix(G::Graph)
	n=G.n
	m=G.m
	d=zeros(G.n);
	for i=1:G.m
		if G.directed
			d[G.u[i]]+=G.w[i];
		else
			d[G.u[i]]+=abs(G.w[i]);
			d[G.v[i]]+=abs(G.w[i]);	
		end	
	end
	if G.directed
		u=zeros(G.m)
		v=zeros(G.m)
		w=zeros(G.m)
		for i=1:G.m
			u[i]=G.u[i]
			v[i]=G.v[i]
			w[i]=G.w[i]/d[G.u[i]]
		end
		return sparse(u,v,w)
	else
		u=zeros(2*G.m);
		v=zeros(2*G.m);
		w=zeros(2*G.m);
		absw=zeros(2*G.m);
		for i=1:G.m
			u[i]=G.u[i];
			v[i]=G.v[i];
			w[i]=G.w[i]/d[G.u[i]];
			absw[i]=abs(w[i]);
			u[G.m+i]=G.v[i];
			v[G.m+i]=G.u[i];
			w[G.m+i]=G.w[i]/d[G.v[i]];
			absw[G.m+i]=abs(w[G.m+i]);
		end
		return sparse(u,v,w)
	end
end

function transpose(G::Graph)
	return Graph(G.n, G.m, G.v, G.u, G.w, G.directed)
end

