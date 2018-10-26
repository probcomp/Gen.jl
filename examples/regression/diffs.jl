##################
# argdiff values #
##################

# depends completely on the generative function.
# the generative function may provide custom data types.
# there are no common data types implemented at the moment.


############
# @retdiff #
############

# @retdiff(true) means that there may have been a change (this is the default)
# > any caller will receive an UnknownCallDiff value from @calldiff

# @retdiff(false) means we are asserting there was no change
# > any caller will receive a NoCallDiff value from @calldiff

# @retdiff(custom::T)
# > any caller will receive a CustomCallDiff{T} value from @calldiff

# NOTE: a function should be able to declare its retdiff type


#############
# @calldiff #
#############

"""
There was no previous call at this address.
"""
struct NewCallDiff end
isnew(::NewCallDiff) = true
isnodiff(::NewCallDiff) = false
isunknowndiff(::NewCallDiff) = false

"""
The two return values are equal.
"""
struct NoCallDiff end 
isnew(::NoCallDiff) = false
isnodiff(::NoCallDiff) = true
isunknowndiff(::NoCallDiff) = false

"""
No information about the difference in the return values was provided.
"""
struct UnknownCallDiff end
isnew(::UnknownCallDiff) = true
isnodiff(::UnknownCallDiff) = false
isunknowndiff(::UnknownCallDiff) = false

"""
Custom information about the difference in the return values.
"""
struct CustomCallDiff{T}
    value::T
end
isnew(::CustomCallDiff) = false
isnodiff(::CustomCallDiff) = false
isunknowndiff(::CustomCallDiff) = false


###############
# @choicediff #
###############

"""
There was no previous random choice at this address.
"""
struct NewChoiceDiff end
isnew(::NewChoiceDiff) = true
isnodiff(::NewChoiceDiff) = false

"""
The value of the random choice has not changed.
"""
struct NoChoiceDiff end 
isnew(::NoChoiceDiff) = false
isnodiff(::NoChoiceDiff) = true

"""
The value of the choice may have changed, and this is the previous value.
"""
struct PrevChoiceDiff{T}
    prev::T
end
isnew(::PrevChoiceDiff) = false
isnodiff(::PrevChoiceDiff) = false
