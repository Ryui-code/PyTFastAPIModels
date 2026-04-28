from pydantic import BaseModel

class IMDBSchema(BaseModel):
    text: str

class AGNewsSchema(BaseModel):
    text: str

class GoEmotionsSchema(BaseModel):
    text: str

class YelpReviewSchema(BaseModel):
    text: str

class SentenceTypeSchema(BaseModel):
    text: str