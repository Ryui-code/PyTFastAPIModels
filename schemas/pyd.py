from pydantic import BaseModel

class IMDBSchema(BaseModel):
    text: str