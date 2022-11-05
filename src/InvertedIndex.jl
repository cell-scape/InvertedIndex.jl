module InvertedIndex

using ArgParse
using DataFrames
using DataFramesMeta
using DataStructures: counter, Accumulator
using Dates
using DimensionalData
using LibPQ
using LinearAlgebra
using PyCall

include("db.jl")
include("inverted_index.jl")
include("document_vector.jl")

export build_inverted_index, build_dictionary_table, build_postings_table, connect, get_table, julia_main, CONN
export build_document_vector, cosine_similarity, remove_stopwords

#= PyCall =#

function __init__()
    py"""
    import string
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize


    def sanitize_text(text: str) -> str:
        '''
        Removes english stop words, numbers, punctuation, etc. from a string using nltk.
        May require nltk.download('stopwords') and nltk.download('punkt').

        Parameters:
        text (str): A string

        Returns:
        str: A string with stop words, numbers, punctuation, etc. removed, with all words stemmed
        '''
        text = " ".join(filter(lambda ch: ch in string.ascii_letters, text)).strip().lower()
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))
        words = list(filter(lambda word: word not in stop_words, word_tokenize(text)))
        return list(map(lambda word: stemmer.stem(word), words))
    """
end

#= CLI =#

"""
    argparser()::ArgParseSettings

Returns argument parser settings.

# Arguments
- `::Nothing`: none

# Returns
- `::ArgParseSettings`: Initialized argparser

# Examples
```julia-repl
julia> argparser()
[...]
```
"""
function argparser()
    s = ArgParse.ArgParseSettings(prog="Inverted Index", description="Creates an inverted index table in Database", epilog="---", autofix_names=true)
    @add_arg_table! s begin
        "--load-idx-from-db", "-L"
        help = "Load inverted index from database"
        action = :store_true
        "--upload-to-db", "-U"
        help = "Upload inverted index tables to database"
        action = :store_true
        "--table", "-t"
        help = "Database table name"
        default = "stateofunion"
        "--dictionary", "-D"
        help = "Dictionary table name"
        default = "sou_dictionary"
        "--postings"
        help = "Postings table name"
        default = "sou_postings"
        "--user", "-u"
        help = "Database username"
        default = "postgres"
        "--pass", "-p"
        help = "Database password"
        default = "postgres"
        "--host", "-H"
        help = "Database hostname"
        default = "localhost"
        "--port", "-P"
        help = "Port number"
        arg_type = Int
        default = 5432
        "--db", "-d"
        help = "Database name"
        default = "postgres"
        "--idf", "-I"
        help = "idf function"
        default = "inv_doc_freq_smooth"
        "--tf", "-T"
        help = "tf function"
        default = "relative_freq"
        "--columns", "-c"
        help = "Columns to select from database (comma separated list)"
        default = "*"
        "--idcol1"
        help = "ID column 1 for unique record key in inverted index (need both)"
        arg_type = Symbol
        default = :president
        "--idcol2"
        help = "ID column 2 for unique record key in inverted index (need both)"
        arg_type = Symbol
        default = :date
        "--text_col", "-C"
        help = "Text column in table"
        arg_type = Symbol
        default = :speech
        "--search-string", "-S"
        help = "A search string for use with document vector"
        default = "freedom"
    end
    return s
end


"""
    julia_main()::Cint

Binary entrypoint. Parses Base.ARGS from commandline with argparser.

# Arguments
- `::Nothing`: none

# Returns
- `::Cint`: C-ABI compatible return code

# Examples
```julia-repl
julia> julia_main()
0
```
"""
function julia_main()::Cint
    @info "Entrypoint: julia_main()"
    ap = argparser()
    args = parse_args(ARGS, ap, as_symbols=true)
    @info "CLI args: " args
    try
        @info "Connecting to database"
        CONN[] = connect(args[:user], args[:pass], args[:host], args[:port], args[:db])
        @info CONN[]

        dictionary, postings = if args[:load_idx_from_db]
            @info "Loading dictionary, postings from DB"
            get_table(CONN[], args[:dictionary]), get_table(CONN[], args[:postings])
        else
            df = get_table(CONN[], args[:table]; columns=split(args[:columns], ','))
            @info "Retrieved SOU table from database" size(df)

            @info "Building inverted index"
            build_inverted_index(
                df;
                id_col1=args[:idcol1],
                id_col2=args[:idcol2],
                text_col=args[:text_col],
                tf_method=TF_METHODS[args[:tf]],
                idf_method=IDF_METHODS[args[:idf]]
            )
        end

        if args[:upload_to_db]
            @info "Loading inverted index into database"

            @info "Loading dictionary table:" args[:dictionary]
            load_table(CONN[], dictionary, args[:dictionary], column_defs=[
                "collectionfreq INT NOT NULL",
                "docfreq INT NOT NULL",
                "idf DOUBLE PRECISION NOT NULL",
                "term TEXT PRIMARY KEY NOT NULL"
            ])
            @info "Successfully loaded dictionary table"

            @info "Loading postings table:" args[:postings]
            load_table(CONN[], postings, args[:postings], column_defs=[
                "doc_id TEXT NOT NULL",
                "term TEXT NOT NULL REFERENCES $(args[:dictionary]) ON DELETE CASCADE",
                "termfreq INTEGER NOT NULL",
                "tf DOUBLE PRECISION NOT NULL",
                "tfidf DOUBLE PRECISION NOT NULL",
                "PRIMARY KEY (doc_id, term)"
            ])
            @info "Successfully loaded postings table"
        end

        @info "Building document vector"
        dvec = build_document_vector(postings)

        if !isempty(args[:search_string])

        end
    catch e
        ex = stacktrace(catch_backtrace())
        @error "Exception:" e, ex
        return -1
    finally
        @info "Closing database connection"
        close(CONN[])
        @info CONN[]
    end
    @info "Success"
    return 0
end

end
