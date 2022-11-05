module InvertedIndex

using DataFrames
using DataFramesMeta
using DataStructures: counter
using Dates
using LibPQ

include("db.jl")
include("inverted_index.jl")

export build_inverted_index

end
