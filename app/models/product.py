from pydantic import BaseModel, Field

class Product(BaseModel):
	id: str
	name: str
	category: str
	brand: str

	price: float = Field(gt=0)
	currency: str = "CNY"

	rating: float = Field(ge=0, le=5)

	tags: list[str] = []
	description: str
