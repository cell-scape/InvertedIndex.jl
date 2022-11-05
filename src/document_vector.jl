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
- `::`
"""
cosine_similarity(A, B) = (A ⋅ B) / (norm(A) * norm(B))


"""
    query(keywords::Vector{String}, dvec::DimArray{Float64, 2})::String

Look up state of the union addresses by relevant words

# Arguments
- `keywords::Vector{String}`: A sequence of words to search for in the document matrix

# Returns
- `::String`: The doc_id of the highest weighted text

# Examples
```julia-repl
julia> query(["freedom", "speech"])

```
"""
function query(keywords, dvec)::String
    isempty(keywords) && return ""

    terms, docs = dvec.dims
    term_freq = Dict(term => 0 for term in terms)
    for keyword in keywords
        if haskey(terms, keyword)
            term_freq[keyword] += 1
        end
    end

    best_sim = 0.0
    best_match = ""
    for doc in docs
        sim = cosine_similarity(terms, dvec[X(), Y(At(doc))])
        if sim > best_match
            best_sim = sim
            best_match = doc
        end
    end
    return best_match, best_sim
end


