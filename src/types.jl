#=

# Various boolean types

# Idea (from <Gaunard-simd.pdf>): Use Mask{N,T} instead of booleans
# with different sizes

abstract Boolean <: Integer

for sz in (8, 16, 32, 64, 128)
    Intsz = Symbol(:Int, sz)
    UIntsz = Symbol(:UInt, sz)
    Boolsz = Symbol(:Bool, sz)
    @eval begin
        immutable $Boolsz <: Boolean
            int::$UIntsz
            $Boolsz(b::Bool) =
                new(ifelse(b, typemax($UIntsz), typemin($UIntsz)))
        end
        booltype(::Type{Val{$sz}}) = $Boolsz
        inttype(::Type{Val{$sz}}) = $Intsz
        uinttype(::Type{Val{$sz}}) = $UIntsz

        Base.convert(::Type{Bool}, b::$Boolsz) = b.int != 0

        Base. ~(b::$Boolsz) = $Boolsz(~b.int)
        Base. !(b::$Boolsz) = ~b
        Base. &(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int & b2.int)
        Base. |(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int | b2.int)
        Base.$(:$)(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int $ b2.int)

        Base. ==(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int == b2.int)
        Base. !=(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int != b2.int)
        Base. <(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int < b2.int)
        Base. <=(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int <= b2.int)
        Base. >(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int > b2.int)
        Base. >=(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int >= b2.int)
    end
end
Base.convert(::Type{Bool}, b::Boolean) = error("impossible")
Base.convert{I<:Integer}(::Type{I}, b::Boolean) = I(Bool(b))
Base.convert{B<:Boolean}(::Type{B}, b::Boolean) = B(Bool(b))
Base.convert{B<:Boolean}(::Type{B}, i::Integer) = B(i!=0)

booltype{T}(::Type{T}) = booltype(Val{8*sizeof(T)})
inttype{T}(::Type{T}) = inttype(Val{8*sizeof(T)})
uinttype{T}(::Type{T}) = uinttype(Val{8*sizeof(T)})

=#

const BoolTypes = Union{Bool}
const IntTypes = Union{Int8, Int16, Int32, Int64, Int128}
const UIntTypes = Union{UInt8, UInt16, UInt32, UInt64, UInt128}
const IntegerTypes = Union{BoolTypes, IntTypes, UIntTypes}
const FloatingTypes = Union{Float16, Float32, Float64}
const ScalarTypes = Union{IntegerTypes, FloatingTypes}

const VE = Base.VecElement

# The Julia SIMD vector type
export Vec
struct Vec{N,T<:ScalarTypes} <: DenseArray{T,1}   # <: Number
    elts::NTuple{N,VE{T}}
    @inline (::Type{Vec{N,T}}){N,T}(elts::NTuple{N, VE{T}}) = new{N,T}(elts)
end

function Base.show(io::IO, v::Vec{N,T}) where {N,T}
    print(io, T, "⟨")
    for i in 1:N
        @static if VERSION < v"0.6-"
            i>1 && print(io, ",")
        else
            i>1 && print(io, ", ")
        end
        print(io, v.elts[i].value)
    end
    print(io, "⟩")
end

# Base.print_matrix wants to access a second dimension that doesn't exist for
# Vec. (In Julia, every array can be accessed as N-dimensional array, for
# arbitrary N.) Instead of implementing this, output our Vec the usual way.
function Base.print_matrix(io::IO, X::Vec,
        pre::AbstractString = " ",  # pre-matrix string
        sep::AbstractString = "  ", # separator between elements
        post::AbstractString = "",  # post-matrix string
        hdots::AbstractString = "  \u2026  ",
        vdots::AbstractString = "\u22ee",
        ddots::AbstractString = "  \u22f1  ",
        hmod::Integer = 5, vmod::Integer = 5)
    print(io, X)
end

# Type properties

# eltype and ndims are provided by DenseArray
# Base.eltype{N,T}(::Type{Vec{N,T}}) = T
# Base.ndims{N,T}(::Type{Vec{N,T}}) = 1
Base.length(::Type{Vec{N,T}}) where {N,T} = N
Base.size(::Type{Vec{N,T}}) where {N,T} = (N,)
Base.size(::Type{Vec{N,T}}, n::Integer) where {N,T} = (N,)[n]
# Base.eltype{N,T}(::Vec{N,T}) = T
# Base.ndims{N,T}(::Vec{N,T}) = 1
Base.length(::Vec{N,T}) where {N,T} = N
Base.size(::Vec{N,T}) where {N,T} = (N,)
Base.size(::Vec{N,T}, n::Integer) where {N,T} = (N,)[n]

# Type conversion

# Create vectors from scalars or tuples
@generated function (::Type{Vec{N,T}}){N,T,S<:ScalarTypes}(x::S)
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}(tuple($([:(VE{T}(T(x))) for i in 1:N]...)))
    end
end
(::Type{Vec{N,T}}){N,T<:ScalarTypes}(xs::Tuple{}) = error("illegal argument")
@generated function (::Type{Vec{N,T}}){N,T,S<:ScalarTypes}(xs::NTuple{N,S})
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}(tuple($([:(VE{T}(T(xs[$i]))) for i in 1:N]...)))
    end
end
(::Type{Vec}){N,T<:ScalarTypes}(xs::NTuple{N,T}) = Vec{N,T}(xs)

# Convert between vectors
@inline Base.convert(::Type{Vec{N,T}}, v::Vec{N,T}) where {N,T} = v
@inline Base.convert(::Type{Vec{N,R}}, v::Vec{N,T}) where {N,R,T} = Vec{N,R}(Tuple(v))
@generated function Base. %(v::Vec{N,T}, ::Type{Vec{N,R}}) where {N,R,T}
    quote
        $(Expr(:meta, :inline))
        Vec{N,R}(tuple($([:(v.elts[$i].value % R) for i in 1:N]...)))
    end
end

# Convert vectors to tuples
@generated function Base.convert(::Type{NTuple{N,R}}, v::Vec{N,T}) where {N,R,T}
    quote
        $(Expr(:meta, :inline))
        tuple($([:(R(v.elts[$i].value)) for i in 1:N]...))
    end
end
@inline Base.convert(::Type{Tuple}, v::Vec{N,T}) where {N,T} =
    Base.convert(NTuple{N,T}, v)

# Promotion rules

# Note: Type promotion only works for subtypes of Number
# Base.promote_rule{N,T<:ScalarTypes}(::Type{Vec{N,T}}, ::Type{T}) = Vec{N,T}

Base.zero(::Type{Vec{N,T}}) where {N,T} = Vec{N,T}(zero(T))
Base.one(::Type{Vec{N,T}}) where {N,T} = Vec{N,T}(one(T))
