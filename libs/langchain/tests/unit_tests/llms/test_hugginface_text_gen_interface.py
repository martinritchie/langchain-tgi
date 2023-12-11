from unittest.mock import MagicMock, patch

import pytest

from langchain.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference


@pytest.fixture
def huggingface_text_gen_interface():
    with patch.dict("sys.modules", text_generation=MagicMock()):
        return HuggingFaceTextGenInference()


def test_validate_environment(huggingface_text_gen_interface):
    values = {
        "inference_server_url": "http://localhost:8000",
        "timeout": 10,
        "server_kwargs": {},
    }

    values_with_client = huggingface_text_gen_interface.validate_environment(values)
    assert "client" in values_with_client
    assert "async_client" in values_with_client
    assert len(values_with_client) == 5

    with pytest.raises(ImportError):
        HuggingFaceTextGenInference.validate_environment(values)


def test__default_params(huggingface_text_gen_interface):
    # First check that the default params are correctly propagated to the property.
    assert (
        huggingface_text_gen_interface._default_params.items()
        <= huggingface_text_gen_interface.__dict__.items()
    )

    # Now check that the model_kwargs are correctly unpacked into the property.
    with patch.dict("sys.modules", text_generation=MagicMock()):
        model_kwargs = {"model_arg_1": "foo"}
        huggingface_text_gen_interface = HuggingFaceTextGenInference(
            model_kwargs=model_kwargs
        )

    assert (
        model_kwargs.items() <= huggingface_text_gen_interface._default_params.items()
    )


def test__invocation_params(
    huggingface_text_gen_interface: HuggingFaceTextGenInference,
):
    invocation_params = huggingface_text_gen_interface._invocation_params(
        runtime_stop=None
    )

    stop_sequences = ["</s>"]
    kwargs = {"foo": "bar"}
    invocation_params_with_stop_sequences = (
        huggingface_text_gen_interface._invocation_params(
            runtime_stop=stop_sequences, **kwargs
        )
    )

    assert (
        invocation_params_with_stop_sequences["stop_sequences"]
        == invocation_params["stop_sequences"] + stop_sequences
    )
    assert "foo" in invocation_params_with_stop_sequences
    assert invocation_params_with_stop_sequences["foo"] == "bar"
