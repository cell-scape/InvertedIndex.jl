"""
    build_document_vector(postings::DataFrame)::Matrix{Float64}

Build a document vector of terms vs doc_ids using tfidf as the score.

# Arguments
- `postings::DataFrame`: Postings table from inverted index

# Returns
- `::Matrix{Float64}`: A Matrix of tfidf weights

# Examples
```julia-repl
julia> A = build_document_vector(dictionary, postings)
366462x22843 Matrix{Float64}
[...]
```
"""
function build_document_vector(postings)
    dvec = zeros(X(unique(postings.term)), Y(unique(postings.doc_id)))
    for row in eachrow(postings)
        dvec[X(At(row.term)), Y(At(row.doc_id))] = row.tfidf
    end
    return dvec
end


"""
    cosine_similarity(A::Vector{Float64}, B::Vector{Float64})

Calculate the cosine similarity between two vectors

# Arguments
- `A::Vector{Float64}`: A vector
- `B::Vector{Float64}`: A vector

# Returns
- `::Float64`
"""
cosine_similarity(A, B) = (A ⋅ B) / (norm(A) * norm(B))


"""
    query(keywords::Vector{String}, dvec::DimArray{Float64, 2})::String

Look up state of the union addresses by relevant words

# Arguments
- `keywords::Vector{String}`: A sequence of words to search for in the document matrix

# Returns
- `::Tuple{String, Float64}`: The doc_id of the highest weighted text and its cosine similarity to the query

# Examples
```julia-repl
julia> query(["freedom", "speech"])

```
"""
function query(keywords, dvec)
    isempty(keywords) && return nothing

    keywords = sanitize_string(keywords) |> sanitize_text
    terms, docs = dvec.dims
    keywords = filter(∈(terms), keywords)

    isempty(keywords) && return nothing

    kws = ones(Float64, length(keywords))
    dvec = dvec[X(At(keywords)), Y()]

    sim = 0.0
    idx = nothing
    for i in 1:length(docs)
        s = cosine_similarity(kws, dvec[:, i])
        if s > sim
            sim = s
            idx = i
        end
    end
    return docs[idx], sim
end


