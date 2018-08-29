import FunctionalCollections: push

##########################
# fast persistent vector #
##########################

struct LinkedList{T}
    value::Union{T,Nothing}
    parent::Union{LinkedList{T},Nothing}
end

LinkedList{T}() where {T} = LinkedList{T}(nothing, nothing)

function LinkedList{T}(arr::Vector{T}) where {T}
    if length(arr) == 0
        return LinkedList{T}()
    end
    node::LinkedList{T} = LinkedList(arr[1], nothing)
    for value in arr[2:end]
        node = LinkedList{T}(value, node)
    end
    node
end

function push(vec::LinkedList{T}, value::T) where {T}
    if vec.value === nothing
        LinkedList{T}(value, nothing)
    else
        LinkedList{T}(value, vec)
    end
end

#function Base.collect(node::LinkedList{T}) where {T}
    #arr = T[node.vlaue]
    #while node.parent !== nothing
        #node = node.parent
        #push!(arr, node.value)
    #end
    #reverse(arr)
#end
#
## incredibly slow random access
#function Base.get(vec::LinkedList, i::Int)
    #collect(vec)[i]
#end


#########
# trace #
#########

struct FastVectorTrace{T,U}
    subtraces::LinkedList{U}
    call::CallRecord{LinkedList{T}}
    is_empty::Bool
end

get_call_record(trace::FastVectorTrace) = trace.call
has_choices(trace::FastVectorTrace) = !trace.is_empty







