"""
    build_inverted_index(df)

Builds a new dataframe as an inverted index using TF-IDF techniques.

# Arguments
- `df::DataFrame`: DataFrame retrieved from Database

# Returns
- `::DataFrame`: Inverted index table

# Examples
```julia-repl
julia> ii = build_inverted_index(df)
NxM DataFrame [...]
```
"""
function build_inverted_index(df::DataFrame, id_col1::Symbol, id_col2::Symbol, text_col::Symbol)::NTuple{2,DataFrame}
    isempty(df) && return df
    doc_ids = string.(df[!, id_col1], "_", df[!, id_col2])
    documents = replace.(ch -> ispunct(first(ch)) || iscntrl(first(ch)) ? " " : ch, split.(lowercase.(df[!, text_col]), "")) .|> join
    coll_freq = join(documents, ' ') |> split |> counter
    terms = collect(keys(coll_freq))
    dictionary_table = build_dictionary_table(coll_freq, terms, documents)
    postings_table = build_postings_table(doc_ids, terms, documents)
    return (dictionary_table, postings_table)
end

"""
    build_postings_table(doc_ids, terms, documents)::DataFrame

Build postings table for inverted index.

# Arguments
- `doc_ids::Vector{String}`: Document IDs
- `terms::Vector{String}`: Terms
- `documents::Vector{String}`: Documents

# Returns
- `::DataFrame`: Postings dataframe

# Examples
```julia-repl
julia> pt = build_postings_table(doc_ids, terms, documents)
NxM DataFrame
[...]
```
"""
function build_postings_table(doc_ids::Vector{String}, terms::Vector{String}, documents::Vector{String})
    postings = Dict(:term => String[], :doc_id => String[], termfreq => Float64[])
    for term in terms
        for (doc_id, document) in zip(doc_ids, documents)
            push!(postings[:term], term)
            push!(postings[:doc_id], doc_id)
            push!(postings[:termfreq], term_frequency(document))
        end
    end
    return DataFrame(postings)
end


"""
    build_dictionary_table(df::DataFrame, text_col::Symbol)::DataFrame

Builds the dictionary table for an inverted index.

# Arguments:
- `df::DataFrame`: A dataframe
- `text_col::Symbol`: The column in the dataframe with the text data to index

# Returns
- `::DataFrame`: the dictionary table as a dataframe

# Examples
```julia-repl
julia> dt = build_dictionary_table(df, :speech)
NxM DataFrame
[...]
```
"""
function build_dictionary_table(coll_freq, terms::Vector{String}, documents::Vector{String})::DataFrame
    doc_freq = document_frequency(terms, Set.(split.(documents)))
    DataFrame(Dict(
        :term => terms,
        :docfreq => [doc_freq[term] for term in terms],
        :collectionfreq => [coll_freq[term] for term in terms],
    ))
end


"""
    tf(term::String, document::String)::Float64

Augmented term frequency to prevent bias towards longer documents. Raw freq divided by 
raw freq of most common term.

# Arguments
- `term::String`: Term
- `document::String`: Document

# Returns
- `::Float64`: tf score

# Examples
```julia-repl
julia> tf(term, document)
0.3
```
"""
function tf(term::String, document::String)
    term_freq = split(document) |> counter
    if !haskey(term_freq, term)
        return 0.0
    end
    return 0.5 + 0.5 * term_freq[term] / maximum(values(term_freq))
end


"""
    idf(terms::Vector{String}, documents::Set{String})::Dict{String, Int}

Gets inverted document frequency for all unique terms in text.

# Arguments
- `terms::Vector{String}`: Set of unique terms
- `documents::Set{String}`: Set of documents

# Returns
- `::Dict{String, Float64}`: Document frequency table

# Examples
```julia-repl
julia> idf(terms, documents)
Dict{String, Int} with 24601 entries:
  "baleful"     => 2
[...]
```
"""
function idf(terms, documents)
    doc_freq = Dict{String,Int}()
    for term in terms
        if !haskey(doc_freq, term)
            push!(doc_freq, term => 0)
        end
        for doc in documents
            if term ∈ doc
                doc_freq[term] += 1
            end
        end
    end
    Dict(term => log10(length(documents) / (1 + doc_freq[term])) for term in terms)
end