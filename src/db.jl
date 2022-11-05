"""
    connect(user::String, pass::String, host::String, port::Int, dbname::String)

Get a database connection

# Arguments
- `user::String`: Postgres Database user
- `pass::String`: User's password
- `host::String`: Hostname
- `port::Int`: port number
- `dbname::String`: Database name

# Returns
- `::LibPQ.Connection`: Postgres Connection

# Examples
```julia
julia> conn = connect("postgres", "postgres", "localhost", 5432, "postgres")

PostgreSQL connection (CONNECTION_OK) with parameters:
  user = postgres
[...]
```
"""
connect(user, pass, host, port, dbname) = LibPQ.Connection("dbname=$dbname user=$user password=$pass port=$port host=$host")

"""
    get_table(conn::LibPQ.Connection, table::String; columns::Vector{String}=["*"])::DataFrame

Retrieve an entire table from database and return as a DataFrame.

# Arguments
- `conn::LibPQ.Connection`: Database connection handle
- `table::String`: Table name

# Keywords
- `columns::Vector{String}`: A sequence of column names (default: ["*"])

# Returns
- `::DataFrame`: Database table as a DataFrame object

# Examples
```julia-repl
julia> df = get_table(conn, "stateofunion", columns=["president", "date", "speech"])
NxM DataFrame [...]
```
"""
function get_table(conn, table; columns=["*"])
    columns = length(columns) > 1 ? join(columns, ",") : first(columns)
    q = """
        SELECT $columns from $table;
    """
    LibPQ.execute(conn, q) |> DataFrame
end