"""
    build_document_vector(dictionary::DataFrame, postings::DataFrame)::Matrix{Float64}

Build a document vector of terms vs doc_ids using tfidf as the score.

# Arguments
- `dictionary::DataFrame`: Dictionary table from inverted index
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
function build_document_vector(dictionary, postings)
    terms = unique(dictionary.term)
    documents = unique(postings.doc_id)
    A = Matrix{Float64}((length(terms), length(documents)), undef)
end