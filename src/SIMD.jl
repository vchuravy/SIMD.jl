__precompile__()

module SIMD
using LLVM

include("types.jl")

# Floating point formats

uint_type(::Type{Float16}) = UInt16
uint_type(::Type{Float32}) = UInt32
uint_type(::Type{Float64}) = UInt64

import Base: significand_mask, exponent_mask, sign_mask

# Convert Julia types to LLVM types
llvmtype(::Type{T}) where T = repr(convert(LLVMType, T))

# Type-dependent optimization flags
# fastflags{T<:IntTypes}(::Type{T}) = "nsw"
# fastflags{T<:UIntTypes}(::Type{T}) = "nuw"
# fastflags{T<:FloatingTypes}(::Type{T}) = "fast"

suffix{T<:IntegerTypes}(N::Integer, ::Type{T}) = "v$(N)i$(8*sizeof(T))"
suffix{T<:FloatingTypes}(N::Integer, ::Type{T}) = "v$(N)f$(8*sizeof(T))"

# Type-dependent LLVM constants
function llvmconst{T}(::Type{T}, val)
    T(val) === T(0) && return "zeroinitializer"
    typ = llvmtype(T)
    "$typ $val"
end
function llvmconst(::Type{Bool}, val)
    Bool(val) === false && return "zeroinitializer"
    typ = "i1"
    "$typ $(Int(val))"
end
function llvmconst{T}(N::Integer, ::Type{T}, val)
    T(val) === T(0) && return "zeroinitializer"
    typ = llvmtype(T)
    "<" * join(["$typ $val" for i in 1:N], ", ") * ">"
end
function llvmconst(N::Integer, ::Type{Bool}, val)
    Bool(val) === false && return "zeroinitializer"
    typ = "i1"
    "<" * join(["$typ $(Int(val))" for i in 1:N], ", ") * ">"
end
function llvmtypedconst{T}(::Type{T}, val)
    typ = llvmtype(T)
    T(val) === T(0) && return "$typ zeroinitializer"
    "$typ $val"
end
function llvmtypedconst(::Type{Bool}, val)
    typ = "i1"
    Bool(val) === false && return "$typ zeroinitializer"
    "$typ $(Int(val))"
end

# Type-dependent LLVM intrinsics
llvmins{T<:IntegerTypes}(::Type{Val{:+}}, N, ::Type{T}) = "add"
llvmins{T<:IntegerTypes}(::Type{Val{:-}}, N, ::Type{T}) = "sub"
llvmins{T<:IntegerTypes}(::Type{Val{:*}}, N, ::Type{T}) = "mul"
llvmins{T<:IntTypes}(::Type{Val{:div}}, N, ::Type{T}) = "sdiv"
llvmins{T<:IntTypes}(::Type{Val{:rem}}, N, ::Type{T}) = "srem"
llvmins{T<:UIntTypes}(::Type{Val{:div}}, N, ::Type{T}) = "udiv"
llvmins{T<:UIntTypes}(::Type{Val{:rem}}, N, ::Type{T}) = "urem"

llvmins{T<:IntegerTypes}(::Type{Val{:~}}, N, ::Type{T}) = "xor"
llvmins{T<:IntegerTypes}(::Type{Val{:&}}, N, ::Type{T}) = "and"
llvmins{T<:IntegerTypes}(::Type{Val{:|}}, N, ::Type{T}) = "or"
llvmins{T<:IntegerTypes}(::Type{Val{:⊻}}, N, ::Type{T}) = "xor"

llvmins{T<:IntegerTypes}(::Type{Val{:<<}}, N, ::Type{T}) = "shl"
llvmins{T<:IntegerTypes}(::Type{Val{:>>>}}, N, ::Type{T}) = "lshr"
llvmins{T<:UIntTypes}(::Type{Val{:>>}}, N, ::Type{T}) = "lshr"
llvmins{T<:IntTypes}(::Type{Val{:>>}}, N, ::Type{T}) = "ashr"

llvmins{T<:IntegerTypes}(::Type{Val{:(==)}}, N, ::Type{T}) = "icmp eq"
llvmins{T<:IntegerTypes}(::Type{Val{:(!=)}}, N, ::Type{T}) = "icmp ne"
llvmins{T<:IntTypes}(::Type{Val{:(>)}}, N, ::Type{T}) = "icmp sgt"
llvmins{T<:IntTypes}(::Type{Val{:(>=)}}, N, ::Type{T}) = "icmp sge"
llvmins{T<:IntTypes}(::Type{Val{:(<)}}, N, ::Type{T}) = "icmp slt"
llvmins{T<:IntTypes}(::Type{Val{:(<=)}}, N, ::Type{T}) = "icmp sle"
llvmins{T<:UIntTypes}(::Type{Val{:(>)}}, N, ::Type{T}) = "icmp ugt"
llvmins{T<:UIntTypes}(::Type{Val{:(>=)}}, N, ::Type{T}) = "icmp uge"
llvmins{T<:UIntTypes}(::Type{Val{:(<)}}, N, ::Type{T}) = "icmp ult"
llvmins{T<:UIntTypes}(::Type{Val{:(<=)}}, N, ::Type{T}) = "icmp ule"

llvmins{T}(::Type{Val{:ifelse}}, N, ::Type{T}) = "select"

llvmins{T<:FloatingTypes}(::Type{Val{:+}}, N, ::Type{T}) = "fadd"
llvmins{T<:FloatingTypes}(::Type{Val{:-}}, N, ::Type{T}) = "fsub"
llvmins{T<:FloatingTypes}(::Type{Val{:*}}, N, ::Type{T}) = "fmul"
llvmins{T<:FloatingTypes}(::Type{Val{:/}}, N, ::Type{T}) = "fdiv"
llvmins{T<:FloatingTypes}(::Type{Val{:inv}}, N, ::Type{T}) = "fdiv"
llvmins{T<:FloatingTypes}(::Type{Val{:rem}}, N, ::Type{T}) = "frem"

llvmins{T<:FloatingTypes}(::Type{Val{:(==)}}, N, ::Type{T}) = "fcmp oeq"
llvmins{T<:FloatingTypes}(::Type{Val{:(!=)}}, N, ::Type{T}) = "fcmp une"
llvmins{T<:FloatingTypes}(::Type{Val{:(>)}}, N, ::Type{T}) = "fcmp ogt"
llvmins{T<:FloatingTypes}(::Type{Val{:(>=)}}, N, ::Type{T}) = "fcmp oge"
llvmins{T<:FloatingTypes}(::Type{Val{:(<)}}, N, ::Type{T}) = "fcmp olt"
llvmins{T<:FloatingTypes}(::Type{Val{:(<=)}}, N, ::Type{T}) = "fcmp ole"

llvmins{T<:FloatingTypes}(::Type{Val{:^}}, N, ::Type{T}) =
    "@llvm.pow.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:abs}}, N, ::Type{T}) =
    "@llvm.fabs.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:ceil}}, N, ::Type{T}) =
    "@llvm.ceil.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:copysign}}, N, ::Type{T}) =
    "@llvm.copysign.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:cos}}, N, ::Type{T}) =
    "@llvm.cos.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:exp}}, N, ::Type{T}) =
    "@llvm.exp.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:exp2}}, N, ::Type{T}) =
    "@llvm.exp2.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:floor}}, N, ::Type{T}) =
    "@llvm.floor.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:fma}}, N, ::Type{T}) =
    "@llvm.fma.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:log}}, N, ::Type{T}) =
    "@llvm.log.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:log10}}, N, ::Type{T}) =
    "@llvm.log10.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:log2}}, N, ::Type{T}) =
    "@llvm.log2.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:max}}, N, ::Type{T}) =
    "@llvm.maxnum.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:min}}, N, ::Type{T}) =
    "@llvm.minnum.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:muladd}}, N, ::Type{T}) =
    "@llvm.fmuladd.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:powi}}, N, ::Type{T}) =
    "@llvm.powi.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:round}}, N, ::Type{T}) =
    "@llvm.rint.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:sin}}, N, ::Type{T}) =
    "@llvm.sin.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:sqrt}}, N, ::Type{T}) =
    "@llvm.sqrt.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:trunc}}, N, ::Type{T}) =
    "@llvm.trunc.$(suffix(N,T))"

# Convert between LLVM scalars, vectors, and arrays

function scalar2vector(vec, siz, typ, sca)
    instrs = []
    accum(nam, i) = i<0 ? "undef" : i==siz-1 ? nam : "$(nam)_iter$i"
    for i in 0:siz-1
        push!(instrs,
            "$(accum(vec,i)) = " *
                "insertelement <$siz x $typ> $(accum(vec,i-1)), " *
                "$typ $sca, i32 $i")
    end
    instrs
end

function array2vector(vec, siz, typ, arr, tmp="$(arr)_av")
    instrs = []
    accum(nam, i) = i<0 ? "undef" : i==siz-1 ? nam : "$(nam)_iter$i"
    for i in 0:siz-1
        push!(instrs, "$(tmp)_elem$i = extractvalue [$siz x $typ] $arr, $i")
        push!(instrs,
            "$(accum(vec,i)) = " *
                "insertelement <$siz x $typ> $(accum(vec,i-1)), " *
                "$typ $(tmp)_elem$i, i32 $i")
    end
    instrs
end

function vector2array(arr, siz, typ, vec, tmp="$(vec)_va")
    instrs = []
    accum(nam, i) = i<0 ? "undef" : i==siz-1 ? nam : "$(nam)_iter$i"
    for i in 0:siz-1
        push!(instrs,
            "$(tmp)_elem$i = extractelement <$siz x $typ> $vec, i32 $i")
        push!(instrs,
            "$(accum(arr,i)) = "*
                "insertvalue [$siz x $typ] $(accum(arr,i-1)), " *
                "$typ $(tmp)_elem$i, $i")
    end
    instrs
end

# TODO: change argument order
function subvector(vec, siz, typ, rvec, rsiz, roff, tmp="$(rvec)_sv")
    instrs = []
    accum(nam, i) = i<0 ? "undef" : i==rsiz-1 ? nam : "$(nam)_iter$i"
    @assert 0 <= roff
    @assert roff + rsiz <= siz
    for i in 0:rsiz-1
        push!(instrs,
            "$(tmp)_elem$i = extractelement <$siz x $typ> $vec, i32 $(roff+i)")
        push!(instrs,
            "$(accum(rvec,i)) = " *
                "insertelement <$rsiz x $typ> $(accum(rvec,i-1)), " *
                "$typ $(tmp)_elem$i, i32 $i")
    end
    instrs
end

function extendvector(vec, siz, typ, voff, vsiz, val, rvec, tmp="$(rvec)_ev")
    instrs = []
    accum(nam, i) = i<0 ? "undef" : i==siz+vsiz-1 ? nam : "$(nam)_iter$i"
    rsiz = siz + vsiz
    for i in 0:siz-1
        push!(instrs,
            "$(tmp)_elem$i = extractelement <$siz x $typ> $vec, i32 $i")
        push!(instrs,
            "$(accum(rvec,i)) = " *
                "insertelement <$rsiz x $typ> $(accum(rvec,i-1)), " *
                "$typ $(tmp)_elem$i, i32 $i")
    end
    for i in siz:siz+vsiz-1
        push!(instrs,
            "$(accum(rvec,i)) = " *
                "insertelement <$rsiz x $typ> $(accum(rvec,i-1)), $val, i32 $i")
    end
    instrs
end

# Element-wise access

export setindex
@generated function setindex{N,T,I}(v::Vec{N,T}, x::Number, ::Type{Val{I}})
    @assert isa(I, Integer)
    1 <= I <= N || throw(BoundsError())
    typ = llvmtype(T)
    ityp = llvmtype(Int)
    vtyp = "<$N x $typ>"
    decls = []
    instrs = []
    push!(instrs, "%res = insertelement $vtyp %0, $typ %1, $ityp $(I-1)")
    push!(instrs, "ret $vtyp %res")
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{T}}, Tuple{NTuple{N,VE{T}}, T}, v.elts, T(x)))
    end
end

@generated function setindex{N,T}(v::Vec{N,T}, x::Number, i::Int)
    typ = llvmtype(T)
    ityp = llvmtype(Int)
    vtyp = "<$N x $typ>"
    decls = []
    instrs = []
    push!(instrs, "%res = insertelement $vtyp %0, $typ %2, $ityp %1")
    push!(instrs, "ret $vtyp %res")
    quote
        $(Expr(:meta, :inline))
        @boundscheck 1 <= i <= N || throw(BoundsError())
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{T}}, Tuple{NTuple{N,VE{T}}, Int, T},
            v.elts, i-1, T(x)))
    end
end
setindex{N,T}(v::Vec{N,T}, x::Number, i) = setindex(v, Int(i), x)

Base.getindex{N,T,I}(v::Vec{N,T}, ::Type{Val{I}}) = v.elts[I].value
Base.getindex{N,T}(v::Vec{N,T}, i) = v.elts[i].value

# Type conversion

@generated function Base.reinterpret{N,R,N1,T1}(::Type{Vec{N,R}},
        v1::Vec{N1,T1})
    @assert N*sizeof(R) == N1*sizeof(T1)
    typ1 = llvmtype(T1)
    vtyp1 = "<$N1 x $typ1>"
    typr = llvmtype(R)
    vtypr = "<$N x $typr>"
    decls = []
    instrs = []
    push!(instrs, "%res = bitcast $vtyp1 %0 to $vtypr")
    push!(instrs, "ret $vtypr %res")
    quote
        $(Expr(:meta, :inline))
        Vec{N,R}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{R}}, Tuple{NTuple{N1,VE{T1}}}, v1.elts))
    end
end

# Generic function wrappers

# Functions taking one argument
@generated function llvmwrap{Op,N,T1,R}(::Type{Val{Op}}, v1::Vec{N,T1},
        ::Type{R} = T1)
    @assert isa(Op, Symbol)
    typ1 = llvmtype(T1)
    vtyp1 = "<$N x $typ1>"
    typr = llvmtype(R)
    vtypr = "<$N x $typr>"
    ins = llvmins(Val{Op}, N, T1)
    decls = []
    instrs = []
    if ins[1] == '@'
        push!(decls, "declare $vtypr $ins($vtyp1)")
        push!(instrs, "%res = call $vtypr $ins($vtyp1 %0)")
    else
        if Op === :~
            @assert T1 <: IntegerTypes
            otherval = -1
        elseif Op === :inv
            @assert T1 <: FloatingTypes
            otherval = 1.0
        else
            otherval = 0
        end
        otherarg = llvmconst(N, T1, otherval)
        push!(instrs, "%res = $ins $vtyp1 $otherarg, %0")
    end
    push!(instrs, "ret $vtypr %res")
    quote
        $(Expr(:meta, :inline))
        Vec{N,R}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{R}}, Tuple{NTuple{N,VE{T1}}}, v1.elts))
    end
end

# Functions taking one Bool argument
@generated function llvmwrap{Op,N}(::Type{Val{Op}}, v1::Vec{N,Bool},
        ::Type{Bool} = Bool)
    @assert isa(Op, Symbol)
    btyp = llvmtype(Bool)
    vbtyp = "<$N x $btyp>"
    ins = llvmins(Val{Op}, N, Bool)
    decls = []
    instrs = []
    push!(instrs, "%arg1 = trunc $vbtyp %0 to <$N x i1>")
    otherarg = llvmconst(N, Bool, true)
    push!(instrs, "%res = $ins <$N x i1> $otherarg, %arg1")
    push!(instrs, "%resb = zext <$N x i1> %res to $vbtyp")
    push!(instrs, "ret $vbtyp %resb")
    quote
        $(Expr(:meta, :inline))
        Vec{N,Bool}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{Bool}}, Tuple{NTuple{N,VE{Bool}}}, v1.elts))
    end
end

# Functions taking two arguments
@generated function llvmwrap{Op,N,T1,T2,R}(::Type{Val{Op}}, v1::Vec{N,T1},
        v2::Vec{N,T2}, ::Type{R} = T1)
    @assert isa(Op, Symbol)
    typ1 = llvmtype(T1)
    vtyp1 = "<$N x $typ1>"
    typ2 = llvmtype(T2)
    vtyp2 = "<$N x $typ2>"
    typr = llvmtype(R)
    vtypr = "<$N x $typr>"
    ins = llvmins(Val{Op}, N, T1)
    decls = []
    instrs = []
    if ins[1] == '@'
        push!(decls, "declare $vtypr $ins($vtyp1, $vtyp2)")
        push!(instrs, "%res = call $vtypr $ins($vtyp1 %0, $vtyp2 %1)")
    else
        push!(instrs, "%res = $ins $vtyp1 %0, %1")
    end
    push!(instrs, "ret $vtypr %res")
    quote
        $(Expr(:meta, :inline))
        Vec{N,R}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{R}}, Tuple{NTuple{N,VE{T1}}, NTuple{N,VE{T2}}},
            v1.elts, v2.elts))
    end
end

# Functions taking two arguments, returning Bool
@generated function llvmwrap{Op,N,T1,T2}(::Type{Val{Op}}, v1::Vec{N,T1},
        v2::Vec{N,T2}, ::Type{Bool})
    @assert isa(Op, Symbol)
    btyp = llvmtype(Bool)
    vbtyp = "<$N x $btyp>"
    abtyp = "[$N x $btyp]"
    typ1 = llvmtype(T1)
    vtyp1 = "<$N x $typ1>"
    atyp1 = "[$N x $typ1]"
    typ2 = llvmtype(T2)
    vtyp2 = "<$N x $typ2>"
    atyp2 = "[$N x $typ2]"
    ins = llvmins(Val{Op}, N, T1)
    decls = []
    instrs = []
    if false && N == 1
        append!(instrs, array2vector("%arg1", N, typ1, "%0", "%arg1arr"))
        append!(instrs, array2vector("%arg2", N, typ2, "%1", "%arg2arr"))
        push!(instrs, "%cond = $ins $vtyp1 %arg1, %arg2")
        push!(instrs, "%res = zext <$N x i1> %cond to $vbtyp")
        append!(instrs, vector2array("%resarr", N, btyp, "%res"))
        push!(instrs, "ret $abtyp %resarr")
    else
        push!(instrs, "%res = $ins $vtyp1 %0, %1")
        push!(instrs, "%resb = zext <$N x i1> %res to $vbtyp")
        push!(instrs, "ret $vbtyp %resb")
    end
    quote
        $(Expr(:meta, :inline))
        Vec{N,Bool}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{Bool}}, Tuple{NTuple{N,VE{T1}}, NTuple{N,VE{T2}}},
            v1.elts, v2.elts))
    end
end

# Functions taking a vector and a scalar argument
# @generated function llvmwrap{Op,N,T1,T2,R}(::Type{Val{Op}}, v1::Vec{N,T1},
#         x2::T2, ::Type{R} = T1)
#     @assert isa(Op, Symbol)
#     typ1 = llvmtype(T1)
#     atyp1 = "[$N x $typ1]"
#     vtyp1 = "<$N x $typ1>"
#     typ2 = llvmtype(T2)
#     typr = llvmtype(R)
#     atypr = "[$N x $typr]"
#     vtypr = "<$N x $typr>"
#     ins = llvmins(Val{Op}, N, T1)
#     decls = []
#     instrs = []
#     append!(instrs, array2vector("%arg1", N, typ1, "%0", "%arg1arr"))
#     if ins[1] == '@'
#         push!(decls, "declare $vtypr $ins($vtyp1, $typ2)")
#         push!(instrs, "%res = call $vtypr $ins($vtyp1 %arg1, $typ2 %1)")
#     else
#         push!(instrs, "%res = $ins $vtyp1 %arg1, %1")
#     end
#     append!(instrs, vector2array("%resarr", N, typr, "%res"))
#     push!(instrs, "ret $atypr %resarr")
#     quote
#         $(Expr(:meta, :inline))
#         Vec{N,R}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
#             NTuple{N,R}, Tuple{NTuple{N,T1}, T2}, v1.elts, x2))
#     end
# end

# Functions taking two Bool arguments, returning Bool
@generated function llvmwrap{Op,N}(::Type{Val{Op}}, v1::Vec{N,Bool},
        v2::Vec{N,Bool}, ::Type{Bool} = Bool)
    @assert isa(Op, Symbol)
    btyp = llvmtype(Bool)
    vbtyp = "<$N x $btyp>"
    ins = llvmins(Val{Op}, N, Bool)
    decls = []
    instrs = []
    push!(instrs, "%arg1 = trunc $vbtyp %0 to <$N x i1>")
    push!(instrs, "%arg2 = trunc $vbtyp %1 to <$N x i1>")
    push!(instrs, "%res = $ins <$N x i1> %arg1, %arg2")
    push!(instrs, "%resb = zext <$N x i1> %res to $vbtyp")
    push!(instrs, "ret $vbtyp %resb")
    quote
        $(Expr(:meta, :inline))
        Vec{N,Bool}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{Bool}}, Tuple{NTuple{N,VE{Bool}}, NTuple{N,VE{Bool}}},
            v1.elts, v2.elts))
    end
end

# Functions taking three arguments
@generated function llvmwrap{Op,N,T1,T2,T3,R}(::Type{Val{Op}}, v1::Vec{N,T1},
        v2::Vec{N,T2}, v3::Vec{N,T3}, ::Type{R} = T1)
    @assert isa(Op, Symbol)
    typ1 = llvmtype(T1)
    vtyp1 = "<$N x $typ1>"
    typ2 = llvmtype(T2)
    vtyp2 = "<$N x $typ2>"
    typ3 = llvmtype(T3)
    vtyp3 = "<$N x $typ3>"
    typr = llvmtype(R)
    vtypr = "<$N x $typr>"
    ins = llvmins(Val{Op}, N, T1)
    decls = []
    instrs = []
    if ins[1] == '@'
        push!(decls, "declare $vtypr $ins($vtyp1, $vtyp2, $vtyp3)")
        push!(instrs,
            "%res = call $vtypr $ins($vtyp1 %0, $vtyp2 %1, $vtyp3 %2)")
    else
        push!(instrs, "%res = $ins $vtyp1 %0, %1, %2")
    end
    push!(instrs, "ret $vtypr %res")
    quote
        $(Expr(:meta, :inline))
        Vec{N,R}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{R}},
            Tuple{NTuple{N,VE{T1}}, NTuple{N,VE{T2}}, NTuple{N,VE{T3}}},
            v1.elts, v2.elts, v3.elts))
    end
end

@generated function llvmwrapshift{Op,N,T,I}(::Type{Val{Op}}, v1::Vec{N,T},
                                            ::Type{Val{I}})
    @assert isa(Op, Symbol)
    if I >= 0
        op = Op
        i = I
    else
        if Op === :>> || Op === :>>>
            op = :<<
        else
            @assert Op === :<<
            if T <: Unsigned
                op = :>>>
            else
                op = :>>
            end
        end
        i = -I
    end
    @assert op in (:<<, :>>, :>>>)
    @assert i >= 0
    typ = llvmtype(T)
    vtyp = "<$N x $typ>"
    ins = llvmins(Val{op}, N, T)
    decls = []
    instrs = []
    nbits = 8*sizeof(T)
    if (op === :>> && T <: IntTypes) || i < nbits
        count = llvmconst(N, T, min(nbits-1, i))
        push!(instrs, "%res = $ins $vtyp %0, $count")
        push!(instrs, "ret $vtyp %res")
    else
        zero = llvmconst(N, T, 0)
        push!(instrs, "return $vtyp $zero")
    end
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{T}}, Tuple{NTuple{N,VE{T}}}, v1.elts))
    end
end

@generated function llvmwrapshift{Op,N,T}(::Type{Val{Op}}, v1::Vec{N,T},
                                          x2::Unsigned)
    @assert isa(Op, Symbol)
    typ = llvmtype(T)
    vtyp = "<$N x $typ>"
    ins = llvmins(Val{Op}, N, T)
    decls = []
    instrs = []
    append!(instrs, scalar2vector("%count", N, typ, "%1"))
    nbits = 8*sizeof(T)
    push!(instrs, "%tmp = $ins $vtyp %0, %count")
    push!(instrs, "%inbounds = icmp ult $typ %1, $nbits")
    if Op === :>> && T <: IntTypes
        nbits1 = llvmconst(N, T, 8*sizeof(T)-1)
        push!(instrs, "%limit = $ins $vtyp %0, $nbits1")
        push!(instrs, "%res = select i1 %inbounds, $vtyp %tmp, $vtyp %limit")
    else
        zero = llvmconst(N, T, 0)
        push!(instrs, "%res = select i1 %inbounds, $vtyp %tmp, $vtyp $zero")
    end
    push!(instrs, "ret $vtyp %res")
    quote
        $(Expr(:meta, :inline))
        # Note that this function might be called with out-of-bounds
        # values for x2, assuming that the results are then ignored
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{T}}, Tuple{NTuple{N,VE{T}}, T}, v1.elts, x2 % T))
    end
end

@generated function llvmwrapshift{Op,N,T}(::Type{Val{Op}}, v1::Vec{N,T},
                                          x2::Integer)
    if Op === :>> || Op === :>>>
        NegOp = :<<
    else
        @assert Op === :<<
        if T <: Unsigned
            NegOp = :>>>
        else
            NegOp = :>>
        end
    end
    ValOp = Val{Op}
    ValNegOp = Val{NegOp}
    quote
        $(Expr(:meta, :inline))
        ifelse(x2 >= 0,
               llvmwrapshift($ValOp, v1, unsigned(x2)),
               llvmwrapshift($ValNegOp, v1, unsigned(-x2)))
    end
end

@generated function llvmwrapshift{Op,N,T,U<:UIntTypes}(::Type{Val{Op}},
                                                       v1::Vec{N,T},
                                                       v2::Vec{N,U})
    @assert isa(Op, Symbol)
    typ = llvmtype(T)
    vtyp = "<$N x $typ>"
    ins = llvmins(Val{Op}, N, T)
    decls = []
    instrs = []
    push!(instrs, "%tmp = $ins $vtyp %0, %1")
    nbits = llvmconst(N, T, 8*sizeof(T))
    push!(instrs, "%inbounds = icmp ult $vtyp %1, $nbits")
    if Op === :>> && T <: IntTypes
        nbits1 = llvmconst(N, T, 8*sizeof(T)-1)
        push!(instrs, "%limit = $ins $vtyp %0, $nbits1")
        push!(instrs,
            "%res = select <$N x i1> %inbounds, $vtyp %tmp, $vtyp %limit")
    else
        zero = llvmconst(N, T, 0)
        push!(instrs,
            "%res = select <$N x i1> %inbounds, $vtyp %tmp, $vtyp $zero")
    end
    push!(instrs, "ret $vtyp %res")
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{T}}, Tuple{NTuple{N,VE{T}}, NTuple{N,VE{T}}},
            v1.elts, (v2 % Vec{N,T}).elts))
    end
end

@generated function llvmwrapshift{Op,N,T,U<:IntegerTypes}(::Type{Val{Op}},
                                                          v1::Vec{N,T},
                                                          v2::Vec{N,U})
    if Op === :>> || Op === :>>>
        NegOp = :<<
    else
        @assert Op === :<<
        if T <: Unsigned
            NegOp = :>>>
        else
            NegOp = :>>
        end
    end
    ValOp = Val{Op}
    ValNegOp = Val{NegOp}
    quote
        $(Expr(:meta, :inline))
        ifelse(v2 >= 0,
               llvmwrapshift($ValOp, v1, v2 % Vec{N,unsigned(U)}),
               llvmwrapshift($ValNegOp, v1, -v2 % Vec{N,unsigned(U)}))
    end
end

# Conditionals

for op in (:(==), :(!=), :(<), :(<=), :(>), :(>=))
    @eval begin
        @inline Base.$op{N,T}(v1::Vec{N,T}, v2::Vec{N,T}) =
            llvmwrap(Val{$(QuoteNode(op))}, v1, v2, Bool)
    end
end
@inline function Base.isfinite{N,T<:FloatingTypes}(v1::Vec{N,T})
    U = uint_type(T)
    em = Vec{N,U}(exponent_mask(T))
    iv = reinterpret(Vec{N,U}, v1)
    iv & em != em
end
@inline Base.isinf{N,T<:FloatingTypes}(v1::Vec{N,T}) = abs(v1) == Vec{N,T}(Inf)
@inline Base.isnan{N,T<:FloatingTypes}(v1::Vec{N,T}) = v1 != v1
@inline function Base.issubnormal{N,T<:FloatingTypes}(v1::Vec{N,T})
    U = uint_type(T)
    em = Vec{N,U}(exponent_mask(T))
    sm = Vec{N,U}(significand_mask(T))
    iv = reinterpret(Vec{N,U}, v1)
    (iv & em == Vec{N,U}(0)) & (iv & sm != Vec{N,U}(0))
end
@inline function Base.signbit{N,T<:FloatingTypes}(v1::Vec{N,T})
    U = uint_type(T)
    sm = Vec{N,U}(sign_mask(T))
    iv = reinterpret(Vec{N,U}, v1)
    iv & sm != Vec{N,U}(0)
end

@generated function Base.ifelse{N,T}(v1::Vec{N,Bool}, v2::Vec{N,T},
        v3::Vec{N,T})
    btyp = llvmtype(Bool)
    vbtyp = "<$N x $btyp>"
    abtyp = "[$N x $btyp]"
    typ = llvmtype(T)
    vtyp = "<$N x $typ>"
    atyp = "[$N x $typ]"
    decls = []
    instrs = []
    if false && N == 1
        append!(instrs, array2vector("%arg1", N, btyp, "%0", "%arg1arr"))
        append!(instrs, array2vector("%arg2", N, typ, "%1", "%arg2arr"))
        append!(instrs, array2vector("%arg3", N, typ, "%2", "%arg3arr"))
        push!(instrs, "%cond = trunc $vbtyp %arg1 to <$N x i1>")
        push!(instrs, "%res = select <$N x i1> %cond, $vtyp %arg2, $vtyp %arg3")
        append!(instrs, vector2array("%resarr", N, typ, "%res"))
        push!(instrs, "ret $atyp %resarr")
    else
        push!(instrs, "%cond = trunc $vbtyp %0 to <$N x i1>")
        push!(instrs, "%res = select <$N x i1> %cond, $vtyp %1, $vtyp %2")
        push!(instrs, "ret $vtyp %res")
    end
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{T}},
            Tuple{NTuple{N,VE{Bool}}, NTuple{N,VE{T}}, NTuple{N,VE{T}}},
            v1.elts, v2.elts, v3.elts))
    end
end

# Integer arithmetic functions

for op in (:~, :+, :-)
    @eval begin
        @inline Base.$op{N,T<:IntegerTypes}(v1::Vec{N,T}) =
            llvmwrap(Val{$(QuoteNode(op))}, v1)
    end
end
@inline Base. !{N}(v1::Vec{N,Bool}) = ~v1
@inline function Base.abs{N,T<:IntTypes}(v1::Vec{N,T})
    # s = -Vec{N,T}(signbit(v1))
    s = v1 >> Val{8*sizeof(T)}
    # Note: -v1 == ~v1 + 1
    (s ⊻ v1) - s
end
@inline Base.abs{N,T<:UIntTypes}(v1::Vec{N,T}) = v1
# TODO: Try T(v1>0) - T(v1<0)
#       use a shift for v1<0
#       evaluate v1>0 as -v1<0 ?
@inline Base.sign{N,T<:IntTypes}(v1::Vec{N,T}) =
    ifelse(v1 == Vec{N,T}(0), Vec{N,T}(0),
        ifelse(v1 < Vec{N,T}(0), Vec{N,T}(-1), Vec{N,T}(1)))
@inline Base.sign{N,T<:UIntTypes}(v1::Vec{N,T}) =
    ifelse(v1 == Vec{N,T}(0), Vec{N,T}(0), Vec{N,T}(1))
@inline Base.signbit{N,T<:IntTypes}(v1::Vec{N,T}) = v1 < Vec{N,T}(0)
@inline Base.signbit{N,T<:UIntTypes}(v1::Vec{N,T}) = Vec{N,Bool}(false)

for op in (:&, :|, :⊻, :+, :-, :*, :div, :rem)
    @eval begin
        @inline Base.$op{N,T<:IntegerTypes}(v1::Vec{N,T}, v2::Vec{N,T}) =
            llvmwrap(Val{$(QuoteNode(op))}, v1, v2)
    end
end
@inline Base.copysign{N,T<:IntTypes}(v1::Vec{N,T}, v2::Vec{N,T}) =
    ifelse(signbit(v2), -abs(v1), abs(v1))
@inline Base.copysign{N,T<:UIntTypes}(v1::Vec{N,T}, v2::Vec{N,T}) = v1
@inline Base.flipsign{N,T<:IntTypes}(v1::Vec{N,T}, v2::Vec{N,T}) =
    ifelse(signbit(v2), -v1, v1)
@inline Base.flipsign{N,T<:UIntTypes}(v1::Vec{N,T}, v2::Vec{N,T}) = v1
@inline Base.max{N,T<:IntegerTypes}(v1::Vec{N,T}, v2::Vec{N,T}) =
    ifelse(v1>=v2, v1, v2)
@inline Base.min{N,T<:IntegerTypes}(v1::Vec{N,T}, v2::Vec{N,T}) =
    ifelse(v1>=v2, v2, v1)

@inline function Base.muladd{N,T<:IntegerTypes}(v1::Vec{N,T}, v2::Vec{N,T},
        v3::Vec{N,T})
    v1*v2+v3
end

# TODO: Handle negative shift counts
#       use ifelse
#       ensure ifelse is efficient
for op in (:<<, :>>, :>>>)
    @eval begin
        @inline Base.$op{N,T<:IntegerTypes,I}(v1::Vec{N,T}, ::Type{Val{I}}) =
            llvmwrapshift(Val{$(QuoteNode(op))}, v1, Val{I})
        @inline Base.$op{N,T<:IntegerTypes}(v1::Vec{N,T}, x2::Unsigned) =
            llvmwrapshift(Val{$(QuoteNode(op))}, v1, x2)
        @inline Base.$op{N,T<:IntegerTypes}(v1::Vec{N,T}, x2::Int) =
            llvmwrapshift(Val{$(QuoteNode(op))}, v1, x2)
        @inline Base.$op{N,T<:IntegerTypes}(v1::Vec{N,T}, x2::Integer) =
            llvmwrapshift(Val{$(QuoteNode(op))}, v1, x2)
        @inline Base.$op{N,T<:IntegerTypes,U<:UIntTypes}(v1::Vec{N,T},
                                                         v2::Vec{N,U}) =
            llvmwrapshift(Val{$(QuoteNode(op))}, v1, v2)
        @inline Base.$op{N,T<:IntegerTypes,U<:IntegerTypes}(v1::Vec{N,T},
                                                            v2::Vec{N,U}) =
            llvmwrapshift(Val{$(QuoteNode(op))}, v1, v2)
        @inline Base.$op{N,T<:IntegerTypes}(x1::T, v2::Vec{N,T}) =
            $op(Vec{N,T}(x1), v2)
    end
end

# Floating point arithmetic functions

for op in (
        :+, :-,
        :abs, :ceil, :cos, :exp, :exp2, :floor, :inv, :log, :log10, :log2,
        :round, :sin, :sqrt, :trunc)
    @eval begin
        @inline Base.$op{N,T<:FloatingTypes}(v1::Vec{N,T}) =
            llvmwrap(Val{$(QuoteNode(op))}, v1)
    end
end
@inline Base.exp10{N,T<:FloatingTypes}(v1::Vec{N,T}) = Vec{N,T}(10)^v1
@inline Base.sign{N,T<:FloatingTypes}(v1::Vec{N,T}) =
    ifelse(v1 == Vec{N,T}(0.0), Vec{N,T}(0.0), copysign(Vec{N,T}(1.0), v1))

for op in (:+, :-, :*, :/, :^, :copysign, :max, :min, :rem)
    @eval begin
        @inline Base.$op{N,T<:FloatingTypes}(v1::Vec{N,T}, v2::Vec{N,T}) =
            llvmwrap(Val{$(QuoteNode(op))}, v1, v2)
    end
end
@inline Base. ^{N,T<:FloatingTypes}(v1::Vec{N,T}, x2::Integer) =
    llvmwrap(Val{:powi}, v1, Int(x2))
@inline Base.flipsign{N,T<:FloatingTypes}(v1::Vec{N,T}, v2::Vec{N,T}) =
    ifelse(signbit(v2), -v1, v1)

for op in (:fma, :muladd)
    @eval begin
        @inline function Base.$op{N,T<:FloatingTypes}(v1::Vec{N,T},
                v2::Vec{N,T}, v3::Vec{N,T})
            llvmwrap(Val{$(QuoteNode(op))}, v1, v2, v3)
        end
    end
end

# Type promotion

# Promote scalars of all IntegerTypes to vectors of IntegerTypes, leaving the
# vector type unchanged

for op in (
        :(==), :(!=), :(<), :(<=), :(>), :(>=),
        :&, :|, :⊻, :+, :-, :*, :copysign, :div, :flipsign, :max, :min, :rem)
    @eval begin
        @inline Base.$op{N}(s1::Bool, v2::Vec{N,Bool}) =
            $op(Vec{N,Bool}(s1), v2)
        @inline Base.$op{N,T<:IntegerTypes}(s1::IntegerTypes, v2::Vec{N,T}) =
            $op(Vec{N,T}(s1), v2)
        @inline Base.$op{N,T<:IntegerTypes}(v1::Vec{N,T}, s2::IntegerTypes) =
            $op(v1, Vec{N,T}(s2))
    end
end
@inline Base.ifelse{N,T<:IntegerTypes}(c::Vec{N,Bool}, s1::IntegerTypes,
        v2::Vec{N,T}) =
    ifelse(c, Vec{N,T}(s1), v2)
@inline Base.ifelse{N,T<:IntegerTypes}(c::Vec{N,Bool}, v1::Vec{N,T},
        s2::IntegerTypes) =
    ifelse(c, v1, Vec{N,T}(s2))

for op in (:muladd,)
    @eval begin
        @inline Base.$op{N,T<:IntegerTypes}(s1::IntegerTypes, v2::Vec{N,T},
                v3::Vec{N,T}) =
            $op(Vec{N,T}(s1), v2, v3)
        @inline Base.$op{N,T<:IntegerTypes}(v1::Vec{N,T}, s2::IntegerTypes,
                v3::Vec{N,T}) =
            $op(v1, Vec{N,T}(s2), v3)
        @inline Base.$op{N,T<:IntegerTypes}(s1::IntegerTypes, s2::IntegerTypes,
                v3::Vec{N,T}) =
            $op(Vec{N,T}(s1), Vec{N,T}(s2), v3)
        @inline Base.$op{N,T<:IntegerTypes}(v1::Vec{N,T}, v2::Vec{N,T},
                s3::IntegerTypes) =
            $op(v1, v2, Vec{N,T}(s3))
        @inline Base.$op{N,T<:IntegerTypes}(s1::IntegerTypes, v2::Vec{N,T},
                s3::IntegerTypes) =
            $op(Vec{N,T}(s1), v2, Vec{N,T}(s3))
        @inline Base.$op{N,T<:IntegerTypes}(v1::Vec{N,T}, s2::IntegerTypes,
                s3::IntegerTypes) =
            $op(v1, Vec{N,T}(s2), Vec{N,T}(s3))
    end
end

# Promote scalars of all ScalarTypes to vectors of FloatingTypes, leaving the
# vector type unchanged

for op in (
        :(==), :(!=), :(<), :(<=), :(>), :(>=),
        :+, :-, :*, :/, :^, :copysign, :flipsign, :max, :min, :rem)
    @eval begin
        @inline Base.$op{N,T<:FloatingTypes}(s1::ScalarTypes, v2::Vec{N,T}) =
            $op(Vec{N,T}(s1), v2)
        @inline Base.$op{N,T<:FloatingTypes}(v1::Vec{N,T}, s2::ScalarTypes) =
            $op(v1, Vec{N,T}(s2))
    end
end
@inline Base.ifelse{N,T<:FloatingTypes}(c::Vec{N,Bool}, s1::ScalarTypes,
        v2::Vec{N,T}) =
    ifelse(c, Vec{N,T}(s1), v2)
@inline Base.ifelse{N,T<:FloatingTypes}(c::Vec{N,Bool}, v1::Vec{N,T},
        s2::ScalarTypes) =
    ifelse(c, v1, Vec{N,T}(s2))

for op in (:fma, :muladd)
    @eval begin
        @inline Base.$op{N,T<:FloatingTypes}(s1::ScalarTypes, v2::Vec{N,T},
                v3::Vec{N,T}) =
            $op(Vec{N,T}(s1), v2, v3)
        @inline Base.$op{N,T<:FloatingTypes}(v1::Vec{N,T}, s2::ScalarTypes,
                v3::Vec{N,T}) =
            $op(v1, Vec{N,T}(s2), v3)
        @inline Base.$op{N,T<:FloatingTypes}(s1::ScalarTypes, s2::ScalarTypes,
                v3::Vec{N,T}) =
            $op(Vec{N,T}(s1), Vec{N,T}(s2), v3)
        @inline Base.$op{N,T<:FloatingTypes}(v1::Vec{N,T}, v2::Vec{N,T},
                s3::ScalarTypes) =
            $op(v1, v2, Vec{N,T}(s3))
        @inline Base.$op{N,T<:FloatingTypes}(s1::ScalarTypes, v2::Vec{N,T},
                s3::ScalarTypes) =
            $op(Vec{N,T}(s1), v2, Vec{N,T}(s3))
        @inline Base.$op{N,T<:FloatingTypes}(v1::Vec{N,T}, s2::ScalarTypes,
                s3::ScalarTypes) =
            $op(v1, Vec{N,T}(s2), Vec{N,T}(s3))
    end
end

# Reduction operations

# TODO: map, mapreduce

function getneutral{T}(op::Symbol, ::Type{T})
    zs = Dict{Symbol,T}()
    if T <: IntegerTypes
        zs[:&] = ~T(0)
        zs[:|] = T(0)
    end
    zs[:max] = typemin(T)
    zs[:min] = typemax(T)
    zs[:+] = T(0)
    zs[:*] = T(1)
    zs[op]
end

# We cannot pass in the neutral element via Val{}; if we try, Julia refuses to
# inline this function, which is then disastrous for performance
@generated function llvmwrapreduce{Op,N,T}(::Type{Val{Op}}, v::Vec{N,T})
    @assert isa(Op, Symbol)
    z = getneutral(Op, T)
    typ = llvmtype(T)
    decls = []
    instrs = []
    n = N
    nam = "%0"
    nold,n = n,nextpow2(n)
    if n > nold
        namold,nam = nam,"%vec_$n"
        append!(instrs,
            extendvector(namold, nold, typ, n, n-nold,
                llvmtypedconst(T,z), nam))
    end
    while n > 1
        nold,n = n, div(n, 2)
        namold,nam = nam,"%vec_$n"
        vtyp = "<$n x $typ>"
        ins = llvmins(Val{Op}, n, T)
        append!(instrs, subvector(namold, nold, typ, "$(nam)_1", n, 0))
        append!(instrs, subvector(namold, nold, typ, "$(nam)_2", n, n))
        if ins[1] == '@'
            push!(decls, "declare $vtyp $ins($vtyp, $vtyp)")
            push!(instrs,
                "$nam = call $vtyp $ins($vtyp $(nam)_1, $vtyp $(nam)_2)")
        else
            push!(instrs, "$nam = $ins $vtyp $(nam)_1, $(nam)_2")
        end
    end
    push!(instrs, "%res = extractelement <$n x $typ> $nam, i32 0")
    push!(instrs, "ret $typ %res")
    quote
        $(Expr(:meta, :inline))
        Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            T, Tuple{NTuple{N,VE{T}}}, v.elts)
    end
end

@inline Base.all{N,T<:IntegerTypes}(v::Vec{N,T}) = llvmwrapreduce(Val{:&}, v)
@inline Base.any{N,T<:IntegerTypes}(v::Vec{N,T}) = llvmwrapreduce(Val{:|}, v)
@inline Base.maximum{N,T<:FloatingTypes}(v::Vec{N,T}) =
    llvmwrapreduce(Val{:max}, v)
@inline Base.minimum{N,T<:FloatingTypes}(v::Vec{N,T}) =
    llvmwrapreduce(Val{:min}, v)
@inline Base.prod{N,T}(v::Vec{N,T}) = llvmwrapreduce(Val{:*}, v)
@inline Base.sum{N,T}(v::Vec{N,T}) = llvmwrapreduce(Val{:+}, v)

@generated function Base.reduce{Op,N,T}(::Type{Val{Op}}, v::Vec{N,T})
    @assert isa(Op, Symbol)
    z = getneutral(Op, T)
    stmts = []
    n = N
    push!(stmts, :($(Symbol(:v,n)) = v))
    nold,n = n,nextpow2(n)
    if n > nold
        push!(stmts,
            :($(Symbol(:v,n)) = Vec{$n,T}($(Expr(:tuple,
                [:($(Symbol(:v,nold)).elts[$i]) for i in 1:nold]...,
                [z for i in nold+1:n]...)))))
    end
    while n > 1
        nold,n = n, div(n, 2)
        push!(stmts,
            :($(Symbol(:v,n,"lo")) = Vec{$n,T}($(Expr(:tuple,
                [:($(Symbol(:v,nold)).elts[$i]) for i in 1:n]...)))))
        push!(stmts,
            :($(Symbol(:v,n,"hi")) = Vec{$n,T}($(Expr(:tuple,
                [:($(Symbol(:v,nold)).elts[$i]) for i in n+1:nold]...)))))
        push!(stmts,
            :($(Symbol(:v,n)) =
                $Op($(Symbol(:v,n,"lo")), $(Symbol(:v,n,"hi")))))
    end
    push!(stmts, :(v1[1]))
    Expr(:block, Expr(:meta, :inline), stmts...)
end

@inline Base.maximum{N,T<:IntegerTypes}(v::Vec{N,T}) = reduce(Val{:max}, v)
@inline Base.minimum{N,T<:IntegerTypes}(v::Vec{N,T}) = reduce(Val{:min}, v)

# Load and store functions

export valloc
function valloc{T}(::Type{T}, N::Int, sz::Int)
    @assert N > 0
    @assert sz >= 0
    padding = N-1
    mem = Vector{T}(sz + padding)
    addr = Int(pointer(mem))
    off = mod(-addr, N * sizeof(T))
    @assert mod(off, sizeof(T)) == 0
    off = fld(off, sizeof(T))
    @assert 0 <= off <= padding
    res = view(mem, off+1 : off+sz)
    addr2 = Int(pointer(res))
    @assert mod(addr2, N * sizeof(T)) == 0
    res
end
function valloc{T}(f, ::Type{T}, N::Int, sz::Int)
    mem = valloc(T, N, sz)
    @inbounds for i in 1:sz
        mem[i] = f(i)
    end
    mem
end

export vload, vloada
@generated function vload{N,T,Aligned}(::Type{Vec{N,T}}, ptr::Ptr{T},
                                       ::Type{Val{Aligned}} = Val{false})
    @assert isa(Aligned, Bool)
    ptyp = llvmtype(Int)
    typ = llvmtype(T)
    vtyp = "<$N x $typ>"
    decls = []
    instrs = []
    if Aligned
        align = N * sizeof(T)
    else
        align = sizeof(T)   # This is overly optimistic
    end
    flags = [""]
    if align > 0
        push!(flags, "align $align")
    end
    if VERSION < v"v0.7.0-DEV"
        push!(instrs, "%ptr = bitcast $typ* %0 to $vtyp*")
    else
        push!(instrs, "%ptr = inttoptr $ptyp %0 to $vtyp*")
    end
    push!(instrs, "%res = load $vtyp, $vtyp* %ptr" * join(flags, ", "))
    push!(instrs, "ret $vtyp %res")
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{T}}, Tuple{Ptr{T}}, ptr))
    end
end

@inline vloada{N,T}(::Type{Vec{N,T}}, ptr::Ptr{T}) =
    vload(Vec{N,T}, ptr, Val{true})

@inline function vload{N,T,Aligned}(::Type{Vec{N,T}},
                                    arr::Union{Array{T,1},SubArray{T,1}},
                                    i::Integer,
                                    ::Type{Val{Aligned}} = Val{false})
    #TODO @boundscheck 1 <= i <= length(arr) - (N-1) || throw(BoundsError())
    vload(Vec{N,T}, pointer(arr, i), Val{Aligned})
end
@inline function vloada{N,T}(::Type{Vec{N,T}},
                             arr::Union{Array{T,1},SubArray{T,1}},
                             i::Integer)
    vload(Vec{N,T}, arr, i, Val{true})
end

@generated function vload{N,T,Aligned}(::Type{Vec{N,T}}, ptr::Ptr{T},
                                       mask::Vec{N,Bool},
                                       ::Type{Val{Aligned}} = Val{false})
    @assert isa(Aligned, Bool)
    ptyp = llvmtype(Int)
    typ = llvmtype(T)
    vtyp = "<$N x $typ>"
    btyp = llvmtype(Bool)
    vbtyp = "<$N x $btyp>"
    decls = []
    instrs = []
    if Aligned
        align = N * sizeof(T)
    else
        align = sizeof(T)   # This is overly optimistic
    end

    if VERSION < v"v0.7.0-DEV"
        push!(instrs, "%ptr = bitcast $typ* %0 to $vtyp*")
    else
        push!(instrs, "%ptr = inttoptr $ptyp %0 to $vtyp*")
    end
    push!(instrs, "%mask = trunc $vbtyp %1 to <$N x i1>")
    push!(decls,
        "declare $vtyp @llvm.masked.load.$(suffix(N,T))($vtyp*, i32, " *
            "<$N x i1>, $vtyp)")
    push!(instrs,
        "%res = call $vtyp @llvm.masked.load.$(suffix(N,T))($vtyp* %ptr, " *
            "i32 $align, <$N x i1> %mask, $vtyp $(llvmconst(N, T, 0)))")
    push!(instrs, "ret $vtyp %res")
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{T}}, Tuple{Ptr{T}, NTuple{N,VE{Bool}}}, ptr, mask.elts))
    end
end

@inline vloada{N,T}(::Type{Vec{N,T}}, ptr::Ptr{T}, mask::Vec{N,Bool}) =
    vload(Vec{N,T}, ptr, mask, Val{true})

@inline function vload{N,T,Aligned}(::Type{Vec{N,T}},
                                    arr::Union{Array{T,1},SubArray{T,1}},
                                    i::Integer, mask::Vec{N,Bool},
                                    ::Type{Val{Aligned}} = Val{false})
    #TODO @boundscheck 1 <= i <= length(arr) - (N-1) || throw(BoundsError())
    vload(Vec{N,T}, pointer(arr, i), mask, Val{Aligned})
end
@inline function vloada{N,T}(::Type{Vec{N,T}},
                             arr::Union{Array{T,1},SubArray{T,1}}, i::Integer,
                             mask::Vec{N,Bool})
    vload(Vec{N,T}, arr, i, mask, Val{true})
end

export vstore, vstorea
@generated function vstore{N,T,Aligned}(v::Vec{N,T}, ptr::Ptr{T},
                                        ::Type{Val{Aligned}} = Val{false})
    @assert isa(Aligned, Bool)
    ptyp = llvmtype(Int)
    typ = llvmtype(T)
    vtyp = "<$N x $typ>"
    decls = []
    instrs = []
    if Aligned
        align = N * sizeof(T)
    else
        align = sizeof(T)   # This is overly optimistic
    end
    flags = [""]
    if align > 0
        push!(flags, "align $align")
    end
    if VERSION < v"v0.7.0-DEV"
        push!(instrs, "%ptr = bitcast $typ* %1 to $vtyp*")
    else
        push!(instrs, "%ptr = inttoptr $ptyp %1 to $vtyp*")
    end
    push!(instrs, "store $vtyp %0, $vtyp* %ptr" * join(flags, ", "))
    push!(instrs, "ret void")
    quote
        $(Expr(:meta, :inline))
        Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            Void, Tuple{NTuple{N,VE{T}}, Ptr{T}}, v.elts, ptr)
    end
end

@inline vstorea{N,T}(v::Vec{N,T}, ptr::Ptr{T}) = vstore(v, ptr, Val{true})

@inline function vstore{N,T,Aligned}(v::Vec{N,T},
                                     arr::Union{Array{T,1},SubArray{T,1}},
                                     i::Integer,
                                     ::Type{Val{Aligned}} = Val{false})
    @boundscheck 1 <= i <= length(arr) - (N-1) || throw(BoundsError())
    vstore(v, pointer(arr, i), Val{Aligned})
end
@inline function vstorea{N,T}(v::Vec{N,T}, arr::Union{Array{T,1},SubArray{T,1}},
                              i::Integer)
    vstore(v, arr, i, Val{true})
end

@generated function vstore{N,T,Aligned}(v::Vec{N,T}, ptr::Ptr{T},
                                        mask::Vec{N,Bool},
                                        ::Type{Val{Aligned}} = Val{false})
    @assert isa(Aligned, Bool)
    ptyp = llvmtype(Int)
    typ = llvmtype(T)
    vtyp = "<$N x $typ>"
    btyp = llvmtype(Bool)
    vbtyp = "<$N x $btyp>"
    decls = []
    instrs = []
    if Aligned
        align = N * sizeof(T)
    else
        align = sizeof(T)   # This is overly optimistic
    end
    if VERSION < v"v0.7.0-DEV"
        push!(instrs, "%ptr = bitcast $typ* %1 to $vtyp*")
    else
        push!(instrs, "%ptr = inttoptr $ptyp %1 to $vtyp*")
    end
    push!(instrs, "%mask = trunc $vbtyp %2 to <$N x i1>")
    push!(decls,
        "declare void @llvm.masked.store.$(suffix(N,T))($vtyp, $vtyp*, i32, " *
            "<$N x i1>)")
    push!(instrs,
        "call void @llvm.masked.store.$(suffix(N,T))($vtyp %0, $vtyp* %ptr, " *
            "i32 $align, <$N x i1> %mask)")
    push!(instrs, "ret void")
    quote
        $(Expr(:meta, :inline))
        Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            Void, Tuple{NTuple{N,VE{T}}, Ptr{T}, NTuple{N,VE{Bool}}},
            v.elts, ptr, mask.elts)
    end
end

@inline vstorea{N,T}(v::Vec{N,T}, ptr::Ptr{T}, mask::Vec{N,Bool}) =
    vstore(v, ptr, mask, Val{true})

@inline function vstore{N,T,Aligned}(v::Vec{N,T},
                                     arr::Union{Array{T,1},SubArray{T,1}},
                                     i::Integer,
                                     mask::Vec{N,Bool},
                                     ::Type{Val{Aligned}} = Val{false})
    #TODO @boundscheck 1 <= i <= length(arr) - (N-1) || throw(BoundsError())
    vstore(v, pointer(arr, i), mask, Val{Aligned})
end
@inline function vstorea{N,T}(v::Vec{N,T},
                              arr::Union{Array{T,1},SubArray{T,1}},
                              i::Integer, mask::Vec{N,Bool})
    vstore(v, arr, i, mask, Val{true})
end

# Vector shuffles

function shufflevector_instrs(N, T, I, two_operands)
    typ = llvmtype(T)
    vtyp2 = vtyp1 = "<$N x $typ>"
    M = length(I)
    vtyp3 = "<$M x i32>"
    vtypr = "<$M x $typ>"
    mask = "<" * join(map(x->string("i32 ", x), I), ", ") * ">"
    instrs = []
    v2 = two_operands ? "%1" : "undef"
    push!(instrs, "%res = shufflevector $vtyp1 %0, $vtyp2 $v2, $vtyp3 $mask")
    push!(instrs, "ret $vtypr %res")
    return M, [], instrs
end

export shufflevector
@generated function shufflevector{N,T,I}(v1::Vec{N,T}, v2::Vec{N,T},
                                         ::Type{Val{I}})
    M, decls, instrs = shufflevector_instrs(N, T, I, true)
    quote
        $(Expr(:meta, :inline))
        Vec{$M,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{$M,VE{T}},
            Tuple{NTuple{N,VE{T}}, NTuple{N,VE{T}}},
            v1.elts, v2.elts))
    end
end

@generated function shufflevector{N,T,I}(v1::Vec{N,T}, ::Type{Val{I}})
    M, decls, instrs = shufflevector_instrs(N, T, I, false)
    quote
        $(Expr(:meta, :inline))
        Vec{$M,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{$M,VE{T}},
            Tuple{NTuple{N,VE{T}}},
            v1.elts))
    end
end

end
