const CONN = Ref{LibPQ.Connection}()

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
        SELECT $(columns) from $(table);
    """
    LibPQ.execute(conn, q) |> DataFrame
end


"""
    load_table(conn::LibPQ.Connection, df::DataFrame, table::String; column_defs::Vector{String})

Load a table into the database.

# Arguments
- `conn::LibPQ.Connection`: Database connection handle
- `df::DataFrame`: Dataframe to load into DB
- `table::String`: Table name

# Keywords
- `column_defs::Union{Nothing, Vector{String}}`: Column definitions (e.g. `ID INT PRIMARY KEY NOT NULL`)

# Returns
- `::Nothing`: nothing

# Examples
```julia-repl
julia> load_table(conn, df, table, columns)
[...]
```
"""
function load_table(conn, df, table; column_defs=nothing)
    if isnothing(column_defs)
        column_defs = [string(name, " TEXT") for name in names]
    end
    _ = create_table(conn, table, column_defs)
    dropmissing!(df) # just in case
    row_strings = IOBuffer(join([string(join(collect(row), ','), "\n") for row in eachrow(df)]))
    copyin = LibPQ.CopyIn("COPY $table FROM STDIN (FORMAT CSV);", row_strings.data)
    LibPQ.execute(conn, copyin)
end


"""
    create_table(conn::LibPQ.Connection, table::String, columns::Vector{String})

Create a database table.

# Arguments
- `conn::LibPQ.Connection`: Database connection handle
- `table::String`: Table name
- `columns::Vector{String}`: Sequence of column definitions to add (e.g.  `id int primary key not null`, not just name)

# Returns
- `::Nothing`: nothing

 # Examples
 ```julia-repl
julia> create_table(conn, table, columns)
[...]
 ```
"""
function create_table(conn, table, columns)
    q = """
        DROP TABLE IF EXISTS $(table);
        CREATE TABLE IF NOT EXISTS $(table)(
            $(join(columns, ",\n"))
        );
    """
    LibPQ.execute(conn, q)
end