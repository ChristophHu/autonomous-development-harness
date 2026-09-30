"""Safe parsing boundary for model-generated structured responses."""

from __future__ import annotations

import logging

from pydantic import BaseModel, ValidationError


class StructuredOutputError(ValueError):
    """A model response failed its declared output contract (without echoing it)."""


def parse_model_output(schema: type[BaseModel], raw: str, *, agent: str):
    try:
        return schema.model_validate_json(raw)
    except (ValidationError, ValueError, TypeError) as error:
        logging.getLogger("harness").warning(
            "model.structured_output_invalid agent=%s schema=%s error=%s",
            agent,
            schema.__name__,
            type(error).__name__,
        )
        raise StructuredOutputError(
            f"{agent} returned output that does not match {schema.__name__}"
        ) from None
