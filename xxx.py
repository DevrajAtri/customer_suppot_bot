from pydantic import BaseModel, Field , ValidationError
class Address(BaseModel):
    street: str
    city: str
    zip_code: int
class User(BaseModel):
    username : str = Field(alias="userName",description =" this is the name of the user ")
    id : int = Field(gt=0 , description="this is the unique id of the user ")
    is_active : bool = Field(default = True , description="this defines the statues of the user")
    address:Address
import json
#print(json.dumps(User.model_json_schema(), indent=2))
# 1. We define the input data (simulating a JSON payload)
input_data = {
    "userName": "jdoe",      # Matches the alias we set!
    "id": 123,
    "is_active": True,
    "address": {             # Just a standard dictionary
        "street": "123 Main St",
        "city": "New York",
        "zip_code": 10001
    }
}

# 2. We unpack the dictionary into the User model
user = User(**input_data)

# 3. Print it to see the magic
#print(user)
#print(type(user.model_dump_json()))

# The AI gives you this string:
json_response = '{"userName": "alice", "id": 555, "address": {"street": "1st Ave", "city": "Boston", "zip_code": 20000}}'

# How do we turn this string into a Pydantic User object?
# Hint: We used model_dump_json() to go OUT. We use model_validate_json() to come IN.
user_from_ai = User.model_validate_json(json_response) # What goes inside?
print(user_from_ai)