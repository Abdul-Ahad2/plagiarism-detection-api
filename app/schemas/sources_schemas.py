from pydantic import BaseModel


class SourceData(BaseModel):
    id: str          # MongoDB ObjectId as string
    title: str
    text: str
    source_url: str
    type: str 