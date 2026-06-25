"""Shared Pydantic base model with enhanced extra-field error messages."""

from typing import Any

from pydantic import BaseModel, model_validator


class JesterBaseModel(BaseModel):
    """Pydantic base model with actionable extra-field validation errors.

    When an unrecognized field is passed, instead of Pydantic's generic
    "Extra inputs are not permitted" message, raises a :class:`ValueError`
    that lists:

    - the unrecognized field name(s),
    - every valid field supported by the config class together with its
      default value (or ``<required>`` / ``<auto>`` where applicable).

    This fires in ``mode='before'``, so it pre-empts Pydantic's own
    ``extra='forbid'`` error and gives users something actionable.
    """

    @model_validator(mode="before")
    @classmethod
    def _report_extra_fields(cls, data: Any) -> Any:
        """Intercept extra fields and emit a helpful error before Pydantic does."""
        if not isinstance(data, dict):
            return data

        known = set(cls.model_fields)
        extra = sorted(set(data) - known)
        if not extra:
            return data

        rows: list[str] = []
        for name in sorted(cls.model_fields):
            field = cls.model_fields[name]
            if field.is_required():
                default_str = "<required>"
            elif field.default_factory is not None:  # type: ignore[misc]
                try:
                    default_str = repr(field.default_factory())  # type: ignore[misc]
                except Exception:
                    default_str = "<auto>"
            else:
                default_str = repr(field.default)
            rows.append(f"  {name:<35} {default_str}")

        extra_str = ", ".join(f"'{k}'" for k in extra)
        valid_str = "\n".join(rows)
        raise ValueError(
            f"Unrecognized field(s) in {cls.__name__}: {extra_str}\n"
            f"Supported fields and their defaults:\n{valid_str}"
        )
