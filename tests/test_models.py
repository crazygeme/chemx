import os
import unittest
from unittest.mock import Mock, call, patch

from chemx.backends import (
    DeepSeekBackend,
    OllamaBackend,
    OpenAIBackend,
    available_backends,
    create_backend,
    get_backend_registration,
    prepare_backend,
    register_backend,
)
from chemx.backends.base import ModelError
from chemx.backends.ollama.runtime import ensure_ollama_running


class ModelFactoryTests(unittest.TestCase):
    def test_lists_built_in_backends(self) -> None:
        self.assertEqual(available_backends(), ("deepseek", "ollama", "openai"))

    def test_creates_local_backend(self) -> None:
        backend = create_backend("ollama", "llama3.2")

        self.assertIsInstance(backend, OllamaBackend)
        self.assertEqual(backend.model, "llama3.2")

    def test_creates_remote_backend_from_environment(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}):
            backend = create_backend("openai", "example-model")

        self.assertIsInstance(backend, OpenAIBackend)
        self.assertEqual(backend.api_key, "secret")

    def test_remote_backend_requires_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                create_backend("openai", "example-model")

    def test_creates_deepseek_backend_from_shared_api_key(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "deepseek-secret"}):
            backend = create_backend("deepseek", "deepseek-v4-flash")

        self.assertIsInstance(backend, DeepSeekBackend)
        self.assertEqual(backend.api_key, "deepseek-secret")
        self.assertEqual(backend.base_url, "https://api.deepseek.com")

    def test_deepseek_registration_uses_current_default_model(self) -> None:
        registration = get_backend_registration("deepseek")

        self.assertEqual(registration.default_model, "deepseek-v4-flash")

    def test_deepseek_requires_shared_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
                create_backend("deepseek", "deepseek-v4-flash")

    @patch("chemx.backends.deepseek.backend.post_json")
    def test_deepseek_uses_openai_compatible_chat_endpoint(
        self,
        post_json: Mock,
    ) -> None:
        post_json.return_value = {
            "choices": [{"message": {"content": "DeepSeek response"}}]
        }
        backend = DeepSeekBackend(
            model="deepseek-v4-flash",
            api_key="secret",
        )

        response = backend.complete([])

        self.assertEqual(response, "DeepSeek response")
        post_json.assert_called_once_with(
            url="https://api.deepseek.com/chat/completions",
            payload={
                "model": "deepseek-v4-flash",
                "messages": [],
                "stream": False,
            },
            headers={"Authorization": "Bearer secret"},
            timeout=60.0,
        )

    def test_registration_adds_backend_without_factory_branching(self) -> None:
        class ExampleBackend:
            def complete(self, messages: object) -> str:
                return "example"

        @register_backend("test-example", default_model="test-model")
        def create_example_backend(**options: object) -> ExampleBackend:
            self.assertEqual(options["model"], "selected-model")
            return ExampleBackend()

        backend = create_backend("test-example", "selected-model")
        registration = get_backend_registration("test-example")

        self.assertIsInstance(backend, ExampleBackend)
        self.assertEqual(registration.default_model, "test-model")

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported model provider"):
            create_backend("missing", "example-model")

    def test_prepare_backend_runs_optional_lifecycle(self) -> None:
        backend = Mock()

        prepare_backend(backend)

        backend.prepare.assert_called_once_with()


class OllamaRuntimeTests(unittest.TestCase):
    @patch("chemx.backends.ollama.runtime.is_ollama_running", return_value=True)
    @patch("chemx.backends.ollama.runtime.subprocess.Popen")
    def test_running_service_is_reused(
        self,
        popen: Mock,
        is_running: Mock,
    ) -> None:
        ensure_ollama_running("http://localhost:11434")

        popen.assert_not_called()
        is_running.assert_called_once_with("http://localhost:11434")

    @patch(
        "chemx.backends.ollama.runtime.is_ollama_running",
        side_effect=[False, False, True],
    )
    @patch("chemx.backends.ollama.runtime.shutil.which", return_value="/usr/bin/ollama")
    @patch("chemx.backends.ollama.runtime.time.sleep")
    @patch("chemx.backends.ollama.runtime.subprocess.Popen")
    def test_installed_local_service_is_started(
        self,
        popen: Mock,
        sleep: Mock,
        which: Mock,
        is_running: Mock,
    ) -> None:
        process = popen.return_value
        process.poll.return_value = None

        ensure_ollama_running("http://localhost:11434")

        popen.assert_called_once()
        self.assertEqual(popen.call_args.args[0], ["/usr/bin/ollama", "serve"])
        self.assertEqual(
            is_running.call_args_list,
            [
                call("http://localhost:11434"),
                call("http://localhost:11434"),
                call("http://localhost:11434"),
            ],
        )
        sleep.assert_called_once_with(0.2)
        which.assert_called_once_with("ollama")

    @patch("chemx.backends.ollama.runtime.is_ollama_running", return_value=False)
    @patch("chemx.backends.ollama.runtime.shutil.which", return_value=None)
    def test_missing_ollama_has_installation_guidance(
        self,
        which: Mock,
        is_running: Mock,
    ) -> None:
        with self.assertRaisesRegex(ModelError, "ollama.com/download"):
            ensure_ollama_running("http://localhost:11434")

        which.assert_called_once_with("ollama")
        is_running.assert_called_once_with("http://localhost:11434")

    @patch("chemx.backends.ollama.runtime.is_ollama_running", return_value=False)
    @patch("chemx.backends.ollama.runtime.shutil.which")
    def test_unreachable_remote_service_is_not_started(
        self,
        which: Mock,
        is_running: Mock,
    ) -> None:
        with self.assertRaisesRegex(ModelError, "not reachable"):
            ensure_ollama_running("https://ollama.example.com")

        which.assert_not_called()
        is_running.assert_called_once_with("https://ollama.example.com")


if __name__ == "__main__":
    unittest.main()
