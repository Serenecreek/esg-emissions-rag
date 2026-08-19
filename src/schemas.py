from typing import Optional

from pydantic import BaseModel, Field


class ScopeEmission(BaseModel):

    value: Optional[float] = Field(
        default=None,
        description=(
            "Emission quantity. "
            "Return null if the value is not available."
        )
    )

    unit: Optional[str] = Field(
        default=None,
        description=(
            "Emission unit such as tCO2e. "
            "Return null if unavailable."
        )
    )


class ESGEmissionData(BaseModel):

    reporting_year: Optional[int] = Field(
        default=None,
        description="The reporting year."
    )

    scope_1: ScopeEmission = Field(
        description=(
            "Scope 1 direct greenhouse gas emissions."
        )
    )

    scope_2: ScopeEmission = Field(
        description=(
            "Scope 2 indirect emissions "
            "from purchased energy."
        )
    )

    scope_3: ScopeEmission = Field(
        description=(
            "Scope 3 value-chain emissions."
        )
    )