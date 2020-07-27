abstract type AddressSchema end
abstract type StaticSchema <: AddressSchema end

struct StaticAddressSchema <: StaticSchema
    keys::Set{Symbol}
end
Base.keys(schema::StaticAddressSchema) = schema.keys

struct StaticInverseAddressSchema <: StaticSchema
    inv::StaticAddressSchema
end
struct InvertedKeys
    keys
end
Base.in(key, ik::InvertedKeys) = !(Base.in(key, ik.keys))
Base.keys(schema::StaticInverseAddressSchema) = InvertedKeys(keys(schema.inv))

struct EmptyAddressSchema <: StaticSchema end
struct AllAddressSchema <: StaticSchema end

struct VectorAddressSchema <: AddressSchema end 
struct SingleDynamicKeyAddressSchema <: AddressSchema end 
struct DynamicAddressSchema <: AddressSchema end 

export AddressSchema
export StaticAddressSchema # hierarchical
export VectorAddressSchema # hierarchical
export SingleDynamicKeyAddressSchema # hierarchical
export DynamicAddressSchema # hierarchical
export EmptyAddressSchema
export AllAddressSchema
export StaticInverseAddressSchema