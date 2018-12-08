##################
# argdiff values #
##################

# these are data types that generative functions may accept. the built-in
# combinators accept these types for their argdiff values. user generative
# functions may or may not accept these.

struct NoArgDiff end
const noargdiff = NoArgDiff()
export NoArgDiff, noargdiff

struct UnknownArgDiff end
const unknownargdiff = UnknownArgDiff()
export UnknownArgDiff, unknownargdiff


############
# retdiffs #
############

"""
A default retdiff value
"""
struct DefaultRetDiff end
isnodiff(::DefaultRetDiff) = false

export DefaultRetDiff


"""
Every retdiff value must implement this function.
"""
function isnodiff end

isnodiff(retdiff::Bool) = retdiff

export isnodiff


############
# calldiff #
############

function isnew end
function isnodiff end
function isunknowndiff end

export isnew
export isnodiff
export isunknowndiff

# values that are returned to the gen function incremental computation path

"""
    NewCallDiff()

Singleton indicating that there was previously no call at this address.
"""
struct NewCallDiff end
isnew(::NewCallDiff) = true
isnodiff(::NewCallDiff) = false
isunknowndiff(::NewCallDiff) = false
export NewCallDiff

"""
    NoCallDiff()

Singleton indicating that the return value of the call has not changed.
"""
struct NoCallDiff end 
isnew(::NoCallDiff) = false
isnodiff(::NoCallDiff) = true
isunknowndiff(::NoCallDiff) = false
export NoCallDiff

"""
    UnknownCallDiff()

Singleton indicating that there was a previous call at this address, but that no information is known about the change to the return value.
"""
struct UnknownCallDiff end
isnew(::UnknownCallDiff) = true
isnodiff(::UnknownCallDiff) = false
isunknowndiff(::UnknownCallDiff) = false
export UnknownCallDiff

"""
    CustomCallDiff(retdiff)

Wrapper around a retdiff value, indicating that there was a previous call at this address, and that `isnodiff(retdiff) = false` (otherwise `NoCallDiff()` would have been returned).
"""
struct CustomCallDiff{T}
    retdiff::T
end
isnew(::CustomCallDiff) = false
isnodiff(::CustomCallDiff) = false
isunknowndiff(::CustomCallDiff) = false
export CustomCallDiff


get_call_diff(state, key) = get_leaf_node(state.calldiffs, key)


##############
# choicediff #
##############



"""
    NewChoiceDiff()

Singleton indicating that there was previously no random choice at this address.
"""
struct NewChoiceDiff end
isnew(::NewChoiceDiff) = true
isnodiff(::NewChoiceDiff) = false
export NewChoiceDiff

"""
    NoChoiceDiff()

Singleton indicating that the value of the random choice did not change.
"""
struct NoChoiceDiff end 
isnew(::NoChoiceDiff) = false
isnodiff(::NoChoiceDiff) = true
export NoChoiceDiff 

"""
    PrevChoiceDiff(prev)

Wrapper around the previous value of the random choice indicating that it may have changed.
"""
struct PrevChoiceDiff{T}
    prev::T
end
isnew(::PrevChoiceDiff) = false
isnodiff(::PrevChoiceDiff) = false
prev(diff::PrevChoiceDiff) = diff.prev
export PrevChoiceDiff
export prev


function set_choice_diff_no_prev!(state, key)
    choicediff = NewChoiceDiff()
    set_leaf_node!(state.choicediffs, key, choicediff)
end

function set_choice_diff!(state, key, value_changed::Bool,
                          prev_retval::T) where {T}
    if value_changed
        choicediff = PrevChoiceDiff(prev_retval)
    else
        choicediff = NoChoiceDiff()
    end
    set_leaf_node!(state.choicediffs, key, choicediff)
end

get_choice_diff(state, key) = get_leaf_node(state.choicediffs, key)
