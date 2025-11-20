from pydantic import BaseModel


class SearchEntry(BaseModel):
    url: str
    title: str
    publish_date: str
    length: int


class SearchResult(BaseModel):
    count: int
    results: list[SearchEntry]
