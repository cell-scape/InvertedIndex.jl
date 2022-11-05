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
        SELECT $columns from $table;
    """
    LibPQ.execute(conn, q) |> DataFrame
end


"""
    load_table(conn::LibPQ.Connection, df::DataFrame)

Load a table into the database.

# Arguments
- `conn::LibPQ.Connection`: Database connection handle
- `table::String`: Table name
- `df::DataFrame`: Dataframe to load into DB

# Returns
- `::Nothing`: nothing

# Examples
```julia-repl
julia> load_table(conn, df)
[...]
```
"""
function load_table(conn, table, df)
    _ = create_table(conn, table, names(df))
    row_strings = map(eachrow(df)) do row
        rowstring = String[]
        for field in row
            if ismissing(field)
                push!(rowstring, ",")
            else
                push!(rowstring, string(field))
            end
        end
        join(rowstring, ',')
        push!(rowstring, "\n")
        join(rowstring)
    end
    copyin = LibPQ.CopyIn("COPY $table FROM STDIN (FORMAT CSV);", row_strings)
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
        CREATE TABLE IF NOT EXISTS $table (
            $(join(columns, ','))
        );
    """
    LibPQ.execute(conn, q)
end