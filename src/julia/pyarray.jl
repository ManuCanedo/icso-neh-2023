struct PythonLikeArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end

Base.size(A::PythonLikeArray) = size(A.data)

function _handle_negative_index(i::Int, dim_size::Int)
    return i < 1 ? dim_size + i : i
end

function Base.getindex(A::PythonLikeArray{T,N}, I::Vararg{Int,N}) where {T,N}
    corrected_indices = ntuple(i -> _handle_negative_index(I[i], size(A, i)), N)
    return getindex(A.data, corrected_indices...)
end

function Base.setindex!(A::PythonLikeArray{T,N}, v, I::Vararg{Int,N}) where {T,N}
    corrected_indices = ntuple(i -> _handle_negative_index(I[i], size(A, i)), N)
    return setindex!(A.data, v, corrected_indices...)
end

Base.IndexStyle(::Type{<:PythonLikeArray}) = IndexLinear()

function toPythonLikeArray(arr::Array{T,N}) where {T,N}
    return PythonLikeArray(arr)
end
