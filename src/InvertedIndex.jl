module InvertedIndex

using ArgParse
using DataFrames
using DataFramesMeta
using DataStructures: counter
using Dates
using LibPQ

include("db.jl")
include("inverted_index.jl")

export build_inverted_index

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
        arg_type = String
        default = "stateofunion"
        "--user", "-u"
        help = "Database username"
        arg_type = String
        default = "postgres"
        "--pass", "-p"
        help = "Database password"
        arg_type = String
        default = "postgres"
        "--host", "-h"
        help = "Database hostname"
        arg_type = String
        default = "localhost"
        "--port", "-P"
        help = "Port number"
        arg_type = Int
        default = 5432
        "--db", "-d"
        help = "Database name"
        arg_type = String
        default = "postgres"
        "--idf", "-I"
        help = "idf function"
        arg_type = String
        default = "inv_doc_freq_smooth"
        "--tf", "-T"
        help = "tf function"
        arg_type = String
        default = "relative_freq"
    end
    return s
end

"""
    julia_main()::Cint

Binary entrypoint.

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

    catch e
        ex = stacktrace(catch_backtrace())
        @error "Exception:" e, ex
        return -1
    end
    return 0
end

end
