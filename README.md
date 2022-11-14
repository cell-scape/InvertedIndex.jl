# InvertedIndex


Fall 2022, CIS 612
Lab 4
Bradley Dowling, 2657649


# Setup

This project is implemented in Julia and Python using PostgreSQL 15 as the database.
Tested on Ubuntu 22.04 x86_64 with Julia 1.8.2, and Python >= 3.7 with NLTK installed.
The code is available at https://github.com/hairshirt/InvertedIndex.jl. 
It has a command line interface that can be used with `julia -e "InvertedIndex.jl; julia_main()"`
and can be compiled to a standalone binary using PackageCompiler.jl.

# Database Connection

The State of the Union speeches are loaded into the data base from in memory dataframe objects and retrieved using LibPQ as the connection library.

```julia
const CONN = Ref{LibPQ.Connection}()

connect(user, pass, host, port, dbname) = LibPQ.Connection("dbname=$dbname user=$user password=$pass port=$port host=$host")

function get_table(conn, table; columns=["*"])
    columns = length(columns) > 1 ? join(columns, ",") : first(columns)
    q = """
        SELECT $(columns) from $(table);
    """
    LibPQ.execute(conn, q) |> DataFrame
end

function load_table(conn, df, table; column_defs=nothing)
    if isnothing(column_defs)
        column_defs = [string(name, " TEXT") for name in names]
    end
    _ = create_table(conn, table, column_defs)
    dropmissing!(df) # just in case
    row_strings = [string(join(collect(row), ','), "\n") for row in eachrow(df)]
    copyin = LibPQ.CopyIn("COPY $table FROM STDIN (FORMAT CSV);", row_strings)
    LibPQ.execute(conn, copyin)
end

function create_table(conn, table, columns)
    q = """
        DROP TABLE IF EXISTS $(table);
        CREATE TABLE IF NOT EXISTS $(table)(
            $(join(columns, ",\n"))
        );
    """
    LibPQ.execute(conn, q)
end
```

![db connection](images/Initial%20DB%20Connection.png)

Here is how they appear in PgAdmin4.

![sou db](images/SOU%20DB.png)

And pulling them back into a dataframe from the database.

![pull db](images/Pull%20from%20DB.png)

# Build Dictionary and Postings Tables

The dictionary and postings tables are built by sanitizing and stemming all of the input text and collecting all unique terms into an accumulator.
The term and document frequency is tracked, and then the tables are loaded into the database. Tf-idf is used as the weight, and several options are
available for methods of calculating tf and idf.

```julia
const TERM_FREQUENCIES = Ref(Dict{AbstractString,Accumulator}())

function build_inverted_index(df; id_col1=:president, id_col2=:date, text_col=:speech, tf_method=relative_freq, idf_method=inv_doc_freq_smooth)::NTuple{2,DataFrame}
    @info "initial df size" size(df)
    dropmissing!(df, [id_col1, id_col2, text_col])
    @info "Dropped missings: " size(df)

    isempty(df) && return df

    doc_ids = string.(df[!, id_col1], "_", df[!, id_col2])
    @info "doc_ids" length(doc_ids)

    documents = replace.(ch -> (isascii(first(ch)) && isletter(first(ch))) ? ch : " ", split.(lowercase.(df[!, text_col]), "")) .|> join
    @info "sanitize documents" length(documents)

    coll_freq = join(documents, ' ') |> sanitize_text |> counter
    @info "Collection Frequency" length(coll_freq)

    terms = collect(keys(coll_freq))
    @info "Unique terms" length(terms)

    dictionary_table = build_dictionary_table(coll_freq, terms, documents; idf_method=idf_method)
    sort!(unique!(dictionary_table, :term), :term)
    @info "dictionary table" size(dictionary_table)

    postings_table = build_postings_table(doc_ids, documents; tf_method=tf_method)
    sort!(unique!(postings_table, [:doc_id, :term]), [:doc_id, :term])
    dd = Dict(row.term => row.idf for row in eachrow(dictionary_table))
    postings_table[!, :tfidf] = [dd[row.term] * row.tf for row in eachrow(postings_table)]
    @info "postings table" size(postings_table)

    dictionary_table, postings_table
end

function build_postings_table(doc_ids, documents; tf_method=relative_freq)::DataFrame
    postings = Dict(:term => String[], :doc_id => String[], :termfreq => Int64[], :tf => Float64[])
    for (doc_id, document) in zip(doc_ids, documents)
        if !haskey(TERM_FREQUENCIES[], doc_id)
            TERM_FREQUENCIES[][doc_id] = sanitize_text(document) |> counter
        end
        term_freq = TERM_FREQUENCIES[][doc_id]
        for (term, freq) in term_freq
            push!(postings[:term], term)
            push!(postings[:doc_id], doc_id)
            push!(postings[:termfreq], freq)
            push!(postings[:tf], tf(term, term_freq; fn=tf_method))
        end
    end
    return DataFrame(postings)
end


function build_dictionary_table(coll_freq, terms, documents; idf_method=inv_doc_freq_smooth)::DataFrame
    doc_freq = document_frequency(terms, Set.(split.(documents)))
    DataFrame(Dict(
        :term => terms,
        :docfreq => [doc_freq[term] for term in terms],
        :idf => [idf(term, doc_freq, length(documents), fn=idf_method) for term in terms],
        :collectionfreq => [coll_freq[term] for term in terms],
    ))
end

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


tf(term, term_freq; fn=relative_freq)::Float64 = fn(term, term_freq)


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


idf(term, doc_freq, ndocs; fn=inv_doc_freq_smooth)::Float64 = fn(term, doc_freq, ndocs)

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

function __init__()
    py"""
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize


    def sanitize_text(text: str) -> list[str]:
        '''
        Removes english stop words and suffixes from a string using nltk.
        May require nltk.download('stopwords') and nltk.download('punkt').

        Parameters:
        text (str): A string

        Returns:
        list[str]: A list of word stems with stop words removed
        '''
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))
        words = list(filter(lambda word: word not in stop_words, word_tokenize(text)))
        return list(map(lambda word: stemmer.stem(word), words))
    """
end

sanitize_text(text) = py"sanitize_text"(text)

sanitize_string(s) = replace(ch -> (isascii(first(ch)) && isletter(first(ch))) ? ch : " ", split(lowercase(s), "")) |> join |> strip
```

![build inverted index](images/Process%20Initial%20Table.png)

![load dictionary](images/Add%20Dictionary%20Table%20to%20DB.png)

![load postings](images/Add%20Postings%20%20table%20to%20DB.png)

# Build Document Vector

Building a matrix of weights for terms and the documents in which they appear.

```julia
function build_document_vector(postings)
    dvec = zeros(X(unique(postings.term)), Y(unique(postings.doc_id)))
    for row in eachrow(postings)
        dvec[X(At(row.term)), Y(At(row.doc_id))] = row.tfidf
    end
    return dvec
end

```

![Build dvec](images/Build%20Doc%20Vector.png)

# Query Results

Query results are computed by cosine similarity on common terms between the document and the query

```julia
cosine_similarity(A, B) = (A ⋅ B) / (norm(A) * norm(B))

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
```

![query results](images/Query%20Results.png)

# Full procedure

![full procedure](images/full%20procedure.png)