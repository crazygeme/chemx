import io
import logging
import unittest
from contextlib import redirect_stderr

from chemx.core.observability import LOGGER_NAMESPACE, configure_logging


class ObservabilityTests(unittest.TestCase):
    def tearDown(self) -> None:
        logger = logging.getLogger(LOGGER_NAMESPACE)
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.ERROR)

    def test_default_logging_shows_errors_only(self) -> None:
        stream = io.StringIO()
        with redirect_stderr(stream):
            level = configure_logging(0)
            logger = logging.getLogger("chemx.test")
            logger.info("hidden status")
            logger.error("visible error")

        self.assertEqual(level, logging.ERROR)
        self.assertNotIn("hidden status", stream.getvalue())
        self.assertIn("visible error", stream.getvalue())

    def test_double_verbose_enables_debug_status(self) -> None:
        stream = io.StringIO()
        with redirect_stderr(stream):
            level = configure_logging(2)
            logging.getLogger("chemx.test").debug("detailed status")

        self.assertEqual(level, logging.DEBUG)
        self.assertIn("DEBUG chemx.test: detailed status", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
