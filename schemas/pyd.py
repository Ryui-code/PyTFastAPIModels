from pydantic import BaseModel

class IMDBSchema(BaseModel):
    text: str

class AGNewsSchema(BaseModel):
    text: str