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
  password = ********************
  channel_binding = prefer
  dbname = postgres
  host = /var/run/postgresql
  port = 5432
  client_encoding = UTF8
  options = -c DateStyle=ISO,YMD -c IntervalStyle=iso_8601 -c TimeZone=UTC
  application_name = LibPQ.jl
  sslmode = prefer
  sslcompression = 0
  sslsni = 1
  ssl_min_protocol_version = TLSv1.2
  gssencmode = prefer
  krbsrvname = postgres
  target_session_attrs = any
```
"""
connect(user::String, pass::String, host::String, port::Int, dbname::String) = LibPQ.Connection("dbname=$dbname user=$user password=$pass port=$port host=$host")

"""
    get_table(conn::LibPQ.Connection, table::String, columns=["*"])::DataFrame

Retrieve an entire table from database and return as a DataFrame.

# Arguments
- `conn::LibPQ.Connection`: Database connection handle
- `table::String`: Table name
- `columns::Vector{String}`: A sequence of column names

# Returns
- `::DataFrame`: Database table as a DataFrame object

# Examples
```julia-repl
julia> df = get_table(conn, "stateofunion", ["president", "date", "speech"])
NxM DataFrame [...]
```
"""
function get_table(conn, table::String, columns::Vector{String}=["*"])
    columns = length(columns) > 1 ? join(columns, ",") : first(columns)
    q = """
        SELECT $columns from $table;
    """
    LibPQ.execute(conn, q) |> DataFrame
end