from src.translator import query_llm_robust
import pytest
from unittest.mock import patch
import openai

# 1. Test unexpected language response
@patch.object(openai.ChatCompletion, 'create')
def test_unexpected_language_response(mock_create):
    # Mock the language detection response to be unexpected
    mock_create.return_value.choices[0].message.content = "I don't understand your request"

    result = query_llm_robust("Hier ist dein erstes Beispiel.")
    assert result == (True, "Hier ist dein erstes Beispiel."), "Expected fallback to default English"

# 2. Test empty translation response
@patch.object(openai.ChatCompletion, 'create')
def test_empty_translation_response(mock_create):
    # First call for language detection, assume it detects a non-English language
    mock_create.side_effect = [
        type('obj', (object,), {'choices': [type('obj', (object,), {'message': {'content': 'German'}})]}),
        # Second call for translation returns an empty response
        type('obj', (object,), {'choices': [type('obj', (object,), {'message': {'content': ''}})]}),
    ]

    result = query_llm_robust("Hier ist dein erstes Beispiel.")
    assert result == (False, "Translation unavailable due to processing error."), "Expected fallback message for empty translation"

# 3. Test non-string translation response
@patch.object(openai.ChatCompletion, 'create')
def test_non_string_translation_response(mock_create):
    # First call for language detection, assume it detects a non-English language
    mock_create.side_effect = [
        type('obj', (object,), {'choices': [type('obj', (object,), {'message': {'content': 'German'}})]}),
        # Second call for translation returns a non-string response
        type('obj', (object,), {'choices': [type('obj', (object,), {'message': {'content': None}})]}),
    ]

    result = query_llm_robust("Hier ist dein erstes Beispiel.")
    assert result == (False, "Translation unavailable due to processing error."), "Expected fallback message for non-string translation"

# 4. Test rate limit error handling
@patch.object(openai.ChatCompletion, 'create')
def test_rate_limit_error_handling(mock_create):
    # Mock a rate limit error initially, then return a valid response on retry
    mock_create.side_effect = [
        openai.error.RateLimitError("Rate limit exceeded"),
        type('obj', (object,), {'choices': [type('obj', (object,), {'message': {'content': 'German'}})]}),
        type('obj', (object,), {'choices': [type('obj', (object,), {'message': {'content': 'Here is your first example.'}})]}),
    ]

    result = query_llm_robust("Hier ist dein erstes Beispiel.")
    assert result == (False, "Here is your first example."), "Expected successful retry after rate limit error"





def test_chinese():
    is_english, translated_content = query_llm_robust("这是一条中文消息")
    assert is_english == False
    assert translated_content == "This is a Chinese message"

def test_llm_normal_response():
    pass

def test_llm_gibberish_response():
    pass
