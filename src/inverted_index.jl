"""
    build_inverted_index(df; id_col1::Symbol, id_col2::Symbol, text_col::Symbol, tf_method::Function, idf_method::Function)::NTuple{2, DataFrame}

Builds 2 new dataframes as an inverted index using TF-IDF weighting

# Arguments
- `df::DataFrame`: DataFrame retrieved from Database

# Keywords
- `id_col1::Symbol`: First of two columns used for a unique key in the index tables (default: :president)
- `id_col2::Symbol`: Second column used for unique key in index tables (default: :date)
- `text_col::Symbol`: The column with the text data to build the index (default: speech)
- `tf_method::Function`: tf method (default: relative_freq)
- `idf_method::Function`: idf method (default: inv_doc_freq_smooth)

# Returns
- `::NTuple{2, DataFrame}`: Inverted index tables (dictionary and posting)

# Examples
```julia-repl
julia> dictionary, posting = build_inverted_index(df)
(NxM DataFrame [...], NxM DataFrame [...])
```
"""
function build_inverted_index(df; id_col1=:president, id_col2=:date, text_col=:speech, tf_method=relative_freq, idf_method=inv_doc_freq_smooth)::NTuple{2,DataFrame}
    isempty(df) && return df
    doc_ids = string.(df[!, id_col1], "_", df[!, id_col2])
    documents = replace.(ch -> ispunct(first(ch)) || iscntrl(first(ch)) ? " " : ch, split.(lowercase.(df[!, text_col]), "")) .|> join
    coll_freq = join(documents, ' ') |> split |> counter
    terms = collect(keys(coll_freq))
    dictionary_table = build_dictionary_table(coll_freq, terms, documents; idf_method=idf_method)
    postings_table = build_postings_table(doc_ids, terms, documents; tf_method=tf_method)
    return (dictionary_table, postings_table)
end

"""
    build_postings_table(doc_ids, terms, documents; tf_method::Function)::DataFrame

Build postings table for inverted index.

# Arguments
- `doc_ids::Vector{String}`: Document IDs
- `terms::Vector{String}`: Terms
- `documents::Vector{String}`: Documents

# Keywords
- `tf_method::Function`: Method to use with tf calculation (default: `relative_freq()`)

# Returns
- `::DataFrame`: Postings dataframe

# Examples
```julia-repl
julia> pt = build_postings_table(doc_ids, terms, documents)
NxM DataFrame
[...]
```
"""
function build_postings_table(doc_ids, terms, documents; tf_method=relative_freq)::DataFrame
    postings = Dict(:term => String[], :doc_id => String[], termfreq => Float64[])
    for term in terms
        for (doc_id, document) in zip(doc_ids, documents)
            push!(postings[:term], term)
            push!(postings[:doc_id], doc_id)
            push!(postings[:termfreq], tf(term, document; fn=tf_method))
        end
    end
    return DataFrame(postings)
end


"""
    build_dictionary_table(coll_freq, terms::Vector{String}, documents::Vector{String}; idf_method::Function)::DataFrame

Builds the dictionary table for an inverted index.

# Arguments:
- `coll_freq::Accumulator`: Collection Frequency
- `terms::Vector{String}`: Set of terms in collection
- `documents::Vector{String}`: All the documents in the collection

# Keywords
- `idf_method::Function`: Method to use with idf calculation (default: `inv_doc_freq_smooth()`)

# Returns
- `::DataFrame`: the dictionary table as a dataframe

# Examples
```julia-repl
julia> dt = build_dictionary_table(coll_freq, terms, documents)
NxM DataFrame
[...]
```
"""
function build_dictionary_table(coll_freq, terms, documents; idf_method=inv_doc_freq_smooth)::DataFrame
    doc_freq = document_frequency(terms, Set.(split.(documents)))
    DataFrame(Dict(
        :term => terms,
        :docfreq => [doc_freq[term] for term in terms],
        :idf => [idf(term, doc_freq, length(documents), fn=idf_method) for term in terms],
        :collectionfreq => [coll_freq[term] for term in terms],
    ))
end


"""
    document_frequency(terms::Vector{String}, documents::Set{String})::Dict{String, Int}

Gets document frequency for all unique terms in collection

# Arguments
- `terms::Vector{String}`: Set of unique terms
- `documents::Set{String}`: Set of documents

# Returns
- `::Dict{String, Int}`: Document frequency table

# Examples
```julia-repl
julia> document_frequency(terms, documents)
Dict{String, Int} with 24601 entries:
  "baleful"     => 2
[...]
```
"""
function document_frequency(terms, documents)::Dict{String,Int}
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
    return doc_freq
end



"""
    tf(term::String, document::String; fn=Function)::Float64

Augmented term frequency to prevent bias towards longer documents. Raw freq divided by 
raw freq of most common term.

# Arguments
- `term::String`: Term
- `document::String`: Document

# Keywords
- `fn::Function`: tf method to apply (default: `relative_freq()`)

# Returns
- `::Float64`: tf score

# Examples
```julia-repl
julia> tf(term, document)
0.3
```
"""
function tf(term::String, document::String; fn=relative_freq)::Float64
    term_freq = split(document) |> counter
    if !haskey(term_freq, term)
        return 0.0
    end
    return fn(term, term_freq)
end

#= Different tf methods =#

augmented(term, term_freq)::Float64 = 0.5 + 0.5 * term_freq[term] / maximum(values(term_freq))
log_scaled(term, term_freq)::Float64 = log10(1.0 + term_freq[term])
boolean_freq(term, term_freq)::Float64 = iszero(term_freq[term]) ? 0.0 : 1.0
raw_count(term, term_freq)::Float64 = term_freq[term]
relative_freq(term, term_freq)::Float64 = term_freq[term] / (sum(values(term_freq)) - term_freq[term])

const TF_METHODS = Dict{String,Function}(
    "augmented" => augmented,
    "log_scaled" => log_scaled,
    "boolean_freq" => boolean_freq,
    "raw_count" => raw_count,
    "relative_freq" => relative_freq,
)

"""
    idf(terms::Vector{String}, doc_freq::Dict{String, Int}, ndocs::Int; fn::Function)::Float64

Get inverse document frequency. Pass a function to fn to appy different idf methods.

# Arguments
- `term::Vector{String}`: A term
- `doc_freq::Dict{String, Int}`: Raw document frequency dictionary
- `ndocs::Int`: Number of documents in collection

# Keywords
- `fn::Function`: A function to apply for different idf methods (default: `inv_doc_freq_smooth()`)

# Returns
- `::Float64`: idf score for term

# Examples
```julia-repl
julia> idf(term, doc_freq, ndocs)
0.3
```
"""
idf(term, doc_freq, ndocs; fn=inv_doc_freq_smooth)::Float64 = fn(term, doc_freq, ndocs)

#= Different idf methods =#

unary(term, doc_freq, ndocs)::Float64 = iszero(doc_freq[term]) ? 0.0 : 1.0
inv_doc_freq(term, doc_freq, ndocs)::Float64 = log10(ndocs / doc_freq[term])
inv_doc_freq_smooth(term, doc_freq, ndocs)::Float64 = log10(ndocs / (1.0 + doc_freq[term])) + 1.0
inv_doc_freq_max(term, doc_freq, ndocs)::Float64 = log10(maximum(values(doc_freq)) / (1.0 + doc_freq[term]))
probabilistic_inv_doc_freq(term, doc_freq, ndocs)::Float64 = log10((ndocs - doc_freq[term]) / doc_freq[term])

const IDF_METHODS = Dict{String,Function}(
    "unary" => unary,
    "inv_doc_freq" => inv_doc_freq,
    "inv_doc_freq_smooth" => inv_doc_freq_smooth,
    "inv_doc_freq_max" => inv_doc_freq_max,
    "probabilistic_inv_doc_freq" => probabilistic_inv_doc_freq,
)