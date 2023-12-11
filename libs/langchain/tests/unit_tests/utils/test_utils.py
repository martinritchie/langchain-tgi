import warnings

import pytest
from pydantic import BaseModel, Field

from langchain.utils.utils import build_extra_kwargs, get_pydantic_field_names


@pytest.fixture
def extra_kwargs():
    return {"model_kwarg_1": "foo", "model_kwarg_2": "bar"}


@pytest.fixture
def values():
    return {"model_kwarg_1": "foo"}


@pytest.fixture
def model_values():
    return {"model_kwarg_3": "baz"}


@pytest.fixture
def all_required_field_names():
    return {"pydantic_field_1", "pydantic_field_2"}


def test_get_pydantic_field_names():
    class PydanticCls(BaseModel):
        a: str
        b: int = Field(..., alias="B")

    all_required_field_names = get_pydantic_field_names(PydanticCls)
    assert all_required_field_names == {"a", "b", "B"}


def test_build_extra_kwargs(extra_kwargs, model_values, all_required_field_names):
    processed_kwargs = build_extra_kwargs(
        extra_kwargs, model_values, all_required_field_names
    )
    assert processed_kwargs == {
        "model_kwarg_1": "foo",
        "model_kwarg_2": "bar",
        "model_kwarg_3": "baz",
    }


def test_build_extra_kwargs_warning(
    extra_kwargs, model_values, all_required_field_names
):
    extra_kwargs = {"model_kwarg_1": "foo", "model_kwarg_2": "bar"}
    # Test non-default model kwarg raises warning
    with warnings.catch_warnings(record=True) as w:
        _ = build_extra_kwargs(extra_kwargs, model_values, all_required_field_names)

    assert len(w) == 1
    assert issubclass(w[-1].category, UserWarning)
    assert "model_kwarg_3" in str(w[-1].message)


def test_build_extra_kwargs_duplicate_model_kwargs(
    extra_kwargs, values, all_required_field_names
):
    # Test duplicated model kwarg raises ValueError
    with pytest.raises(ValueError, match="Found model_kwarg_1 supplied twice."):
        _ = build_extra_kwargs(extra_kwargs, values, all_required_field_names)


def test_build_extra_kwargs_invalid_model_kwargs(values, all_required_field_names):
    # test invalid model kwarg raises ValueError
    extra_kwargs = {"pydantic_field_1": "foo", "model_kwarg_2": "bar"}
    with pytest.raises(
        ValueError,
        match="Parameters {'pydantic_field_1'} should be specified explicitly. "
        "Instead they were passed in as part of `model_kwargs` parameter.",
    ):
        _ = build_extra_kwargs(extra_kwargs, values, all_required_field_names)
