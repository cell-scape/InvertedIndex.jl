module InvertedIndex

using ArgParse
using DataFrames
using DataFramesMeta
using DataStructures: counter
using Dates
using LibPQ

include("db.jl")
include("inverted_index.jl")

export build_inverted_index, build_dictionary_table, build_postings_table, connect, get_table

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
        "--table", "-t"
        help = "Database table name"
        default = "stateofunion"
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
    ap = argparser()
    args = parse_args(ARGS, ap, as_symbols=true)
    try
        @info "Connecting to database"
        conn = connect(args[:user], args[:pass], args[:host], args[:port], args[:db])
        @info conn

        @info "Retrieving table from database"
        df = get_table(conn, args[:table]; columns=split(args[:columns], ','))
        @info size(df)

        @info "Building inverted index"
        dictionary, postings = build_inverted_index(
            df;
            id_col1=args[:idcol1],
            id_col2=args[:idcol2],
            text_col=args[:text_col],
            tf_method=TF_METHODS[args[:tf]],
            idf_method=IDF_METHODS[args[:idf]]
        )
        @info "Dictionary" size(dictionary)
        @info "Posting" size(postings)

    catch e
        ex = stacktrace(catch_backtrace())
        @error "Exception:" e, ex
        return -1
    end
    return 0
end

end
