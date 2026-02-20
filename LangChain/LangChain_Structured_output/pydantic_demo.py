from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):

    # name: str = 'aniket'
    
    # Default Value

    name: str = 'aniket'

    # Optional Value
    age: Optional[int] = None

    # Type Validation
    email: EmailStr

    # Field Function
    cgpa: float = Field(gt=0, lt=10)

# Basic Examples

# new_student = {'name':'aniket'}  
# new_student = {'name': 32}  

# new_student ={'age': 32}

# Type Coercing
new_student = {'age': '32', 'email': 'abc123@gmail.com', 'cgpa':5}

student = Student(**new_student)

# print(type(student))

student_dict =dict(student)

print(student_dict['age'])

student_json = student.model_dump_json()
