##################
# argdiff values #
##################

# depends completely on the generative function.
# the generative function may provide custom data types.

# generative functions may optionally expect these arg diff types.

struct NoArgDiff end
const noargdiff = NoArgDiff()

struct UnknownArgDiff end
const unknownargdiff = UnknownArgDiff()

export noargdiff, unknownargdiff

# NOTE: argdiff values NEED NOT implement any is-no-difference function.

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

"""
Every retdiff value must implement this function?
"""
function isnoretdiff end

isnoretdiff(retdiff::Bool) = retdiff

export isnoretdiff
export isnew, isnodiff, isunknowndiff


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

# TODO should we check isnodiff on the sub-value?
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


###############################
# plate higher order function #
###############################

# valid argdiff values:


# possible retdiff values:


# the nested function can have arbitrary argdiff and retdiff values.

# the retdiff values returned from the nested function will be tested with
# isnoretdiff().

# if the number of applications did not change, and each returned a value with
# isnoretdiff() = true, then PlateNoRetDiff is returned as the retdiff from
# the Plate.

# if the number of applications changed or the retdiff for any retained
# application has isnoretdiff() = false, then a PlateCustomRetDiff will be
# returned, that contains the retdiff values for any retained applications for
# which isnoretdiff() = false.


###############################
## tree higher order function #
###############################
#
#"""
#The return value did not change.
#"""
#struct TreeNoRetDiff end
#isnoretdiff(::PlateNoRetDiff) = true
#
## NOTE: we might also return a custom DW retdiff type (which could itself have isnoretdiff = true)
#
#
### production function ##
#
## the production function accepts argdiff values of type DU.
#
## the retdiff values returned from the production function must have type:
#
#"""
#The v in the return value has vdiff (which may have isnoretdiff = true or
#false).  The number of children may have changed. For children that were
#retained, and for which their u value may have changed, we provide a custom
#udiff value, for which isnoretdiff() = false (otherwise we would not include
#it; we can assert this).
#"""
#struct TreeProductionRetDiff{DV,DU}
    #vdiff::DV
    #udiffs::Dict{Int,DU}
#end
#
## the type DV must implement isnoretdiff().
#
## the tree will test vdiff for isnoretdiff, and take control flow actions
## accordingly
#
#
### aggregation function ##
#
## the aggregation function must accept argdiff values with type:
#
#"""
#The v has vdiff (which may have isnoretdiff = true or false).
#The number of children may have changed.
#For retained children for which isnoretdiff(dw) = false, include the dw values.
#"""
#struct TreeAggregationArgDiff{DV,DW}
    #vdiff::DV
    #wdiffs::Dict{Int,DW}
#end
#
## the DV used in TreeProductionRetDiff and TreeAggregationArgDiff must be the same.
#
## the aggregation function must must return retdiff values of type DW. this
## type must implement isnoretdiff.
#
## when computing the argdiff for an aggregation node, the tree will only
## include DWs for those retained children for which isnoretdiff(dw) = false.
#
#
#
##############################
## lightweight gen functions #
##############################
#
#struct GenFunctionDefaultRetDiff end
#isnoretdiff(::GenFunctionDefaultRetDiff) = false
