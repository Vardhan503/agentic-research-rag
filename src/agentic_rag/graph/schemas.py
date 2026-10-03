from typing import Literal

from pydantic import BaseModel, Field, model_validator


class RetrievalDecision(BaseModel):
    retrieve: bool
    reason: str = Field(min_length=3, max_length=400)


class DocumentGrade(BaseModel):
    source_id: str = Field(min_length=1)
    grade: Literal["correct", "ambiguous", "incorrect"]
    reason: str = Field(min_length=3, max_length=400)


class DocumentGradeBatch(BaseModel):
    grades: list[DocumentGrade] = Field(min_length=1)


class ContextAssessment(BaseModel):
    status: Literal["sufficient", "incomplete", "irrelevant"]
    reason: str = Field(min_length=3, max_length=600)
    missing_information: str = Field(default="", max_length=800)

    @model_validator(mode="after")
    def validate_missing_information(self) -> "ContextAssessment":
        if self.status == "sufficient":
            self.missing_information = ""

        return self


class RewrittenQuery(BaseModel):
    rewritten_query: str = Field(min_length=3, max_length=500)
    reason: str = Field(min_length=3, max_length=400)


class GeneratedAnswer(BaseModel):
    answer: str = Field(min_length=1)
    # Required with at least one entry so the structured-output
    # grammar forces the model to declare its sources.
    source_ids: list[str] = Field(min_length=1)


class HallucinationResult(BaseModel):
    grounded: bool
    reason: str = Field(min_length=3, max_length=600)
    unsupported_claims: list[str] = Field(default_factory=list)


class AnswerCritique(BaseModel):
    useful: bool
    needs_more_context: bool
    reason: str = Field(min_length=3, max_length=600)
    improvement_feedback: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def validate_useful_answer(self) -> "AnswerCritique":
        if self.useful:
            self.needs_more_context = False
            self.improvement_feedback = ""

        return self
