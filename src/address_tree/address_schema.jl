abstract type AddressSchema end

struct StaticAddressSchema <: AddressSchema
    keys::Set{Symbol}
end

Base.keys(schema::StaticAddressSchema) = schema.keys

struct VectorAddressSchema <: AddressSchema end 
struct SingleDynamicKeyAddressSchema <: AddressSchema end 
struct DynamicAddressSchema <: AddressSchema end 
struct EmptyAddressSchema <: AddressSchema end
struct AllAddressSchema <: AddressSchema end

export AddressSchema
export StaticAddressSchema # hierarchical
export VectorAddressSchema # hierarchical
export SingleDynamicKeyAddressSchema # hierarchical
export DynamicAddressSchema # hierarchical
export EmptyAddressSchema
export AllAddressSchema