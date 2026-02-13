import pytest
from unittest.mock import patch, MagicMock
from knowledge_base.processing.retry import retry_with_backoff


class TestRetryWithBackoff:
    """Unit tests for the retry_with_backoff utility function."""

    @patch("knowledge_base.processing.retry.time.sleep")
    def test_succeeds_on_first_attempt(self, mock_sleep):
        """Verifies that when fn() succeeds immediately, the result is returned with no retries or sleeps."""
        mock_fn = MagicMock(return_value="success")

        result = retry_with_backoff(
            fn=mock_fn,
            max_retries=3,
            initial_delay=2,
            max_delay=60,
            retryable_on=(ValueError,),
        )

        assert result == "success"
        mock_fn.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("knowledge_base.processing.retry.time.sleep")
    def test_retries_then_succeeds(self, mock_sleep):
        """Verifies that fn() is retried on retryable exceptions and returns the result once it succeeds."""
        mock_fn = MagicMock(
            side_effect=[ValueError("rate limited"), ValueError("rate limited"), "success"]
        )

        result = retry_with_backoff(
            fn=mock_fn,
            max_retries=3,
            initial_delay=2,
            max_delay=60,
            retryable_on=(ValueError,),
        )

        assert result == "success"
        assert mock_fn.call_count == 3
        assert mock_sleep.call_count == 2
        # Verify exponential backoff delays: 2*2^0=2, 2*2^1=4
        mock_sleep.assert_any_call(2)
        mock_sleep.assert_any_call(4)

    @patch("knowledge_base.processing.retry.time.sleep")
    def test_exhausts_retries_and_raises(self, mock_sleep):
        """Verifies that the original exception is raised after all retry attempts are exhausted."""
        mock_fn = MagicMock(side_effect=ValueError("rate limited"))

        with pytest.raises(ValueError, match="rate limited"):
            retry_with_backoff(
                fn=mock_fn,
                max_retries=3,
                initial_delay=2,
                max_delay=60,
                retryable_on=(ValueError,),
            )

        # 1 initial + 3 retries = 4 total attempts
        assert mock_fn.call_count == 4
        # Sleeps happen after attempts 0, 1, 2 (not after attempt 3 which raises)
        assert mock_sleep.call_count == 3

    @patch("knowledge_base.processing.retry.time.sleep")
    def test_non_retryable_exception_propagates_immediately(self, mock_sleep):
        """Verifies that exceptions not in retryable_on propagate immediately without retrying."""
        mock_fn = MagicMock(side_effect=TypeError("bad input"))

        with pytest.raises(TypeError, match="bad input"):
            retry_with_backoff(
                fn=mock_fn,
                max_retries=3,
                initial_delay=2,
                max_delay=60,
                retryable_on=(ValueError,),
            )

        mock_fn.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("knowledge_base.processing.retry.time.sleep")
    def test_delay_is_capped_by_max_delay(self, mock_sleep):
        """Verifies that the backoff delay never exceeds max_delay, even at high attempt counts."""
        mock_fn = MagicMock(side_effect=ValueError("rate limited"))

        with pytest.raises(ValueError):
            retry_with_backoff(
                fn=mock_fn,
                max_retries=6,
                initial_delay=2,
                max_delay=10,
                retryable_on=(ValueError,),
            )

        # With initial_delay=2, max_delay=10: delays are 2, 4, 8, 10, 10, 10
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [2, 4, 8, 10, 10, 10]

    @patch("knowledge_base.processing.retry.time.sleep")
    def test_multiple_retryable_exception_types(self, mock_sleep):
        """Verifies that retry works when multiple exception types are specified in retryable_on."""
        mock_fn = MagicMock(
            side_effect=[ValueError("val error"), RuntimeError("runtime error"), "success"]
        )

        result = retry_with_backoff(
            fn=mock_fn,
            max_retries=3,
            initial_delay=2,
            max_delay=60,
            retryable_on=(ValueError, RuntimeError),
        )

        assert result == "success"
        assert mock_fn.call_count == 3
        assert mock_sleep.call_count == 2
