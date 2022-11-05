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
    terms = unique(postings.term)
    documents = unique(postings.doc_id)
    A = zeros((length(terms), length(documents)))
end