import warnings
from typing import Literal
from unittest.mock import Mock

import httpx
import pytest
from openai import RateLimitError

from agentlab.llm import llm_utils
from agentlab.llm.chat_api import make_system_message

yaml_str = """Analysis:
This is the analysis

Summary: This is the summary

Confidence Score: 7
"""


def test_yaml_parser():
    ans, _, _ = llm_utils.yaml_parser(yaml_str)
    print(ans)
    assert ans["Analysis"] == "This is the analysis"
    assert ans["Summary"] == "This is the summary"
    assert ans["Confidence Score"] == 7


def test_truncate_tokens():
    text = "This is a simple test."
    truncated = llm_utils.truncate_tokens(text, max_tokens=3)
    assert truncated == "This is a"


def test_count_tokens():
    text = "This is a simple test."
    assert llm_utils.count_tokens(text) == 6


def test_json_parser():
    # Testing valid JSON
    message = '{"test": "Hello, World!"}'

    # deactivate warnings
    warnings.filterwarnings("ignore")

    value, valid, retry_message = llm_utils.json_parser(message)
    assert value == {"test": "Hello, World!"}
    assert valid == True
    assert retry_message == ""

    # Testing invalid JSON
    message = '{"test": "Hello, World!"'  # missing closing brace
    value, valid, retry_message = llm_utils.json_parser(message)
    assert value == {}
    assert valid == False
    assert len(retry_message) > 3

    # reactivate warnings
    warnings.filterwarnings("default")


def test_compress_string():
    text = """
This is a test
for paragraph.

This is a second test.
hola
This is a second test.

This is a test
for paragraph.
"""

    expected_output = """\
<definitions>
§-0:
This is a test
for paragraph.
¶-0:
This is a second test.
</definitions>
§-0
¶-0
hola
¶-0
§-0"""

    compressed_text = llm_utils.compress_string(text)
    assert compressed_text == expected_output


# Mock ChatOpenAI class
class MockChatOpenAI:
    def call(self, messages):
        return "mocked response"

    def __call__(self, messages):
        return self.call(messages)


def mock_parser(answer):
    if answer == "correct content":
        return "Parsed value"
    else:
        raise llm_utils.ParseError("Retry message")


def mock_rate_limit_error(message: str, status_code: Literal[429] = 429) -> RateLimitError:
    """
    Create a mocked instantiation of RateLimitError with a specified message and status code.

    Args:
        message (str): The error message.
        status_code (Literal[429]): The HTTP status code, default is 429 for rate limiting.

    Returns:
        RateLimitError: A mocked RateLimitError instance.
    """
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = status_code
    mock_response.json.return_value = {"error": {"message": message}}
    mock_response.headers = {"x-request-id": "test-request-id"}  # Add headers attribute

    return RateLimitError(message, response=mock_response, body=mock_response.json())


# Test to ensure function stops retrying after reaching the max wait time
# def test_rate_limit_max_wait_time():
#     mock_chat = MockChatOpenAI()
#     mock_chat.call = Mock(
#         side_effect=mock_rate_limit_error("Rate limit reached. Please try again in 2s.")
#     )

#     with pytest.raises(RateLimitError):
#         llm_utils.retry(
#             mock_chat,
#             [],
#             n_retry=4,
#             parser=mock_parser,
#             rate_limit_max_wait_time=6,
#             min_retry_wait_time=1,
#         )

#     # The function should stop retrying after 2 attempts (6s each time, 12s total which is greater than the 10s max wait time)
#     assert mock_chat.call.call_count == 3


# def test_rate_limit_success():
#     mock_chat = MockChatOpenAI()
#     mock_chat.call = Mock(
#         side_effect=[
#             mock_rate_limit_error("Rate limit reached. Please try again in 2s."),
#             make_system_message("correct content"),
#         ]
#     )

#     result = llm_utils.retry(
#         mock_chat,
#         [],
#         n_retry=4,
#         parser=mock_parser,
#         rate_limit_max_wait_time=6,
#         min_retry_wait_time=1,
#     )

#     assert result == "Parsed value"
#     assert mock_chat.call.call_count == 2


# Mock a successful parser response to test function exit before max retries
def test_successful_parse_before_max_retries():
    mock_chat = MockChatOpenAI()

    # mock a chat that returns the wrong content the first 2 time, but the right
    # content  on the 3rd time
    mock_chat.call = Mock(
        side_effect=[
            make_system_message("wrong content"),
            make_system_message("wrong content"),
            make_system_message("correct content"),
        ]
    )

    result = llm_utils.retry(mock_chat, llm_utils.Discussion(), 5, mock_parser)

    assert result == "Parsed value"
    assert mock_chat.call.call_count == 3


def test_unsuccessful_parse_before_max_retries():
    mock_chat = MockChatOpenAI()

    # mock a chat that returns the wrong content the first 2 time, but the right
    # content  on the 3rd time
    mock_chat.call = Mock(
        side_effect=[
            make_system_message("wrong content"),
            make_system_message("wrong content"),
            make_system_message("correct content"),
        ]
    )
    with pytest.raises(llm_utils.ParseError):
        result = llm_utils.retry(mock_chat, llm_utils.Discussion(), 2, mock_parser)

    assert mock_chat.call.call_count == 2


def test_retry_parse_raises():
    mock_chat = MockChatOpenAI()
    mock_chat.call = Mock(return_value=make_system_message("mocked response"))
    parser_raises = Mock(side_effect=ValueError("Parser error"))

    with pytest.raises(ValueError):
        llm_utils.retry(mock_chat, llm_utils.Discussion(), 3, parser_raises)


def test_extract_code_blocks():
    text = """\
This is some text.
```python
def hello_world():
    print("Hello, world!")
```
Some more text.
```
More code without a language.
```
Another block of code:
```javascript
console.log("Hello, world!");
```
An inline code block ```click()```
"""

    expected_output = [
        ("python", 'def hello_world():\n    print("Hello, world!")'),
        ("", "More code without a language."),
        ("javascript", 'console.log("Hello, world!");'),
        ("", "click()"),
    ]

    assert llm_utils.extract_code_blocks(text) == expected_output


def test_message_merge_only_text():
    content = [
        {"type": "text", "text": "Hello, world!"},
        {"type": "text", "text": "This is a test."},
    ]
    message = llm_utils.BaseMessage(role="system", content=content)
    message.merge()
    assert message["content"] == "Hello, world!\nThis is a test."


def test_message_merge_text_image():
    content = [
        {"type": "text", "text": "Hello, world!"},
        {"type": "text", "text": "This is a test."},
        {"type": "image_url", "image_url": "this is a base64 image"},
        {"type": "text", "text": "This is another test."},
        {"type": "text", "text": "Goodbye, world!"},
    ]
    message = llm_utils.BaseMessage(role="system", content=content)
    message.merge()
    assert len(message["content"]) == 3
    assert message["content"][0]["text"] == "Hello, world!\nThis is a test."
    assert message["content"][1]["image_url"] == "this is a base64 image"
    assert message["content"][2]["text"] == "This is another test.\nGoodbye, world!"


class FakeRateLimitError(Exception):
    pass


class FakeAPIError(Exception):
    def __init__(self, status_code):
        super().__init__(f"api error with status {status_code}")
        self.status_code = status_code


def get_fake_status_code(exception):
    return exception.status_code


@pytest.fixture
def sleep_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(llm_utils.time, "sleep", lambda delay: calls.append(delay))
    return calls


def call_with_retries(client_function, is_response_valid_fn=lambda response: True, **kwargs):
    return llm_utils.generic_call_api_with_retries(
        client_function=client_function,
        api_params={},
        is_response_valid_fn=is_response_valid_fn,
        rate_limit_exceptions=(FakeRateLimitError,),
        api_error_exceptions=(FakeAPIError,),
        get_status_code_fn=get_fake_status_code,
        max_retries=kwargs.pop("max_retries", 4),
        initial_retry_delay_seconds=kwargs.pop("initial_retry_delay_seconds", 1),
        **kwargs,
    )


def test_generic_retries_returns_valid_response_without_sleeping(sleep_calls):
    client_function = Mock(return_value="response")

    assert call_with_retries(client_function) == "response"
    assert client_function.call_count == 1
    assert sleep_calls == []


def test_generic_retries_recovers_from_rate_limit_error(sleep_calls):
    client_function = Mock(side_effect=[FakeRateLimitError("slow down"), "response"])

    assert call_with_retries(client_function) == "response"
    assert client_function.call_count == 2
    assert sleep_calls == [2]


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_generic_retries_recovers_from_retriable_status_codes(sleep_calls, status_code):
    client_function = Mock(side_effect=[FakeAPIError(status_code), "response"])

    assert call_with_retries(client_function) == "response"
    assert client_function.call_count == 2
    assert sleep_calls == [2]


@pytest.mark.parametrize("status_code", [400, 401, 404])
def test_generic_retries_reraises_non_retriable_status_codes(sleep_calls, status_code):
    client_function = Mock(side_effect=FakeAPIError(status_code))

    with pytest.raises(FakeAPIError):
        call_with_retries(client_function)

    assert client_function.call_count == 1
    assert sleep_calls == []


def test_generic_retries_reraises_when_status_code_is_unavailable(sleep_calls):
    client_function = Mock(side_effect=FakeAPIError(429))

    with pytest.raises(FakeAPIError):
        llm_utils.generic_call_api_with_retries(
            client_function=client_function,
            api_params={},
            is_response_valid_fn=lambda response: True,
            rate_limit_exceptions=(FakeRateLimitError,),
            api_error_exceptions=(FakeAPIError,),
            get_status_code_fn=Mock(side_effect=AttributeError("no status code")),
            max_retries=4,
            initial_retry_delay_seconds=1,
        )

    assert client_function.call_count == 1
    assert sleep_calls == []


def test_generic_retries_reraises_unexpected_exception(sleep_calls):
    client_function = Mock(side_effect=KeyError("unexpected"))

    with pytest.raises(KeyError):
        call_with_retries(client_function)

    assert client_function.call_count == 1
    assert sleep_calls == []


def test_generic_retries_on_invalid_response_content(sleep_calls):
    client_function = Mock(side_effect=["invalid", "invalid", "valid"])

    result = call_with_retries(
        client_function, is_response_valid_fn=lambda response: response == "valid"
    )

    assert result == "valid"
    assert client_function.call_count == 3
    assert sleep_calls == [1, 2]


def test_generic_retries_raises_runtime_error_when_exhausted_by_invalid_responses(sleep_calls):
    client_function = Mock(return_value="invalid")

    with pytest.raises(RuntimeError):
        call_with_retries(
            client_function,
            is_response_valid_fn=lambda response: False,
            max_retries=3,
        )

    assert client_function.call_count == 3
    assert sleep_calls == [1, 2]


def test_generic_retries_raises_runtime_error_when_exhausted_by_rate_limits(sleep_calls):
    client_function = Mock(side_effect=FakeRateLimitError("slow down"))

    with pytest.raises(RuntimeError):
        call_with_retries(client_function, max_retries=3)

    assert client_function.call_count == 3
    assert sleep_calls == [2, 4]


def test_generic_retries_raises_runtime_error_when_exhausted_by_api_errors(sleep_calls):
    client_function = Mock(side_effect=FakeAPIError(500))

    with pytest.raises(RuntimeError):
        call_with_retries(client_function, max_retries=3)

    assert client_function.call_count == 3
    assert sleep_calls == [2, 4]


def test_generic_retries_backoff_is_exponential_and_capped(sleep_calls):
    client_function = Mock(side_effect=FakeRateLimitError("slow down"))

    with pytest.raises(RuntimeError):
        call_with_retries(
            client_function,
            max_retries=6,
            initial_retry_delay_seconds=10,
            max_retry_delay_seconds=100,
        )

    assert sleep_calls == [20, 40, 80, 100, 100]


if __name__ == "__main__":
    # test_retry_parallel()
    # test_rate_limit_max_wait_time()
    # test_successful_parse_before_max_retries()
    # test_unsuccessful_parse_before_max_retries()
    # test_extract_code_blocks()
    # test_message_merge_only_text()
    test_message_merge_text_image()
