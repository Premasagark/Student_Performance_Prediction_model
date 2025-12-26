from pydantic import BaseModel

class ModelInput(BaseModel):
    study_hours: int
    attendance: int
    internal_marks: int
    practice_hours: int



class ModelAccuracy(BaseModel):
    score: int