from __future__ import annotations

from typing import Annotated, ClassVar, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class StrictModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


MetricCount = Annotated[int, Field(ge=0)]
JsonObject: TypeAlias = dict[str, JsonValue]
