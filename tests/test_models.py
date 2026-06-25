import os
import unittest
from unittest.mock import Mock, call, patch

from chemx.backends import (
    DeepSeekBackend,
    OllamaBackend,
    OpenAIBackend,
    ToolDefinition,
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
        self.assertEqual(backend.context_window_tokens, 131_072)

    def test_creates_remote_backend_from_environment(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}):
            backend = create_backend("openai", "example-model")

        self.assertIsInstance(backend, OpenAIBackend)
        self.assertEqual(backend.api_key, "secret")
        self.assertEqual(backend.context_window_tokens, 1_047_576)

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
        self.assertEqual(backend.context_window_tokens, 128_000)

    def test_deepseek_registration_uses_current_default_model(self) -> None:
        registration = get_backend_registration("deepseek")

        self.assertEqual(registration.default_model, "deepseek-v4-flash")

    def test_deepseek_requires_shared_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
                create_backend("deepseek", "deepseek-v4-flash")

    @patch("chemx.backends.deepseek.backend.post_json")
    def test_deepseek_allows_unstructured_prose_responses(
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

    @patch("chemx.backends.deepseek.backend.post_json")
    def test_deepseek_uses_json_mode_when_schema_is_requested(
        self,
        post_json: Mock,
    ) -> None:
        post_json.return_value = {
            "choices": [{"message": {"content": '{"result":"ok"}'}}]
        }
        backend = DeepSeekBackend(
            model="deepseek-v4-flash",
            api_key="secret",
        )

        backend.complete([], response_schema={"type": "object"})

        self.assertEqual(
            post_json.call_args.kwargs["payload"]["response_format"],
            {"type": "json_object"},
        )

    @patch("chemx.backends.openai.backend.post_json")
    def test_openai_allows_unstructured_prose_responses(
        self,
        post_json: Mock,
    ) -> None:
        post_json.return_value = {
            "choices": [{"message": {"content": '{"result":"ok"}'}}]
        }
        backend = OpenAIBackend(model="example-model", api_key="secret")

        response = backend.complete([])

        self.assertEqual(response, '{"result":"ok"}')
        self.assertNotIn(
            "response_format",
            post_json.call_args.kwargs["payload"],
        )

    @patch("chemx.backends.openai.backend.post_json")
    def test_openai_uses_strict_json_schema_when_supplied(
        self,
        post_json: Mock,
    ) -> None:
        schema = {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
            "additionalProperties": False,
        }
        post_json.return_value = {
            "choices": [{"message": {"content": '{"result":"ok"}'}}]
        }
        backend = OpenAIBackend(model="example-model", api_key="secret")

        backend.complete([], response_schema=schema)

        self.assertEqual(
            post_json.call_args.kwargs["payload"]["response_format"],
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "chemx_response",
                    "strict": True,
                    "schema": schema,
                },
            },
        )

    @patch("chemx.backends.openai.backend.post_json")
    def test_openai_forces_one_strict_native_tool_call(
        self,
        post_json: Mock,
    ) -> None:
        post_json.return_value = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"src/app.py"}',
                                }
                            }
                        ]
                    }
                }
            ]
        }
        backend = OpenAIBackend(model="example-model", api_key="secret")
        tool = ToolDefinition(
            "read_file",
            "Read one file.",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        )

        selected = backend.complete_tool([], [tool])

        self.assertEqual(selected.name, "read_file")
        self.assertEqual(selected.arguments, {"path": "src/app.py"})
        payload = post_json.call_args.kwargs["payload"]
        self.assertEqual(payload["tool_choice"], "required")
        self.assertFalse(payload["parallel_tool_calls"])
        self.assertTrue(payload["tools"][0]["function"]["strict"])

    @patch("chemx.backends.deepseek.backend.post_json")
    def test_deepseek_reads_one_native_tool_call(self, post_json: Mock) -> None:
        post_json.return_value = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "git_status",
                                    "arguments": "{}",
                                }
                            }
                        ]
                    }
                }
            ]
        }
        backend = DeepSeekBackend(
            model="deepseek-v4-flash",
            api_key="secret",
        )
        tool = ToolDefinition("git_status", "Show status.", {"type": "object"})

        selected = backend.complete_tool([], [tool])

        self.assertEqual(selected.name, "git_status")
        self.assertEqual(selected.arguments, {})
        payload = post_json.call_args.kwargs["payload"]
        self.assertIn("tools", payload)
        self.assertEqual(payload["tool_choice"], "required")
        self.assertEqual(payload["thinking"], {"type": "disabled"})

    @patch("chemx.backends.ollama.backend.post_json")
    def test_ollama_allows_unstructured_prose_responses(
        self,
        post_json: Mock,
    ) -> None:
        post_json.return_value = {"message": {"content": '{"result":"ok"}'}}
        backend = OllamaBackend(model="llama3.2")

        response = backend.complete([])

        self.assertEqual(response, '{"result":"ok"}')
        self.assertNotIn("format", post_json.call_args.kwargs["payload"])

    @patch("chemx.backends.ollama.backend.post_json")
    def test_ollama_uses_json_schema_when_supplied(self, post_json: Mock) -> None:
        schema = {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
        }
        post_json.return_value = {"message": {"content": '{"result":"ok"}'}}
        backend = OllamaBackend(model="llama3.2")

        backend.complete([], response_schema=schema)

        self.assertIs(post_json.call_args.kwargs["payload"]["format"], schema)

    @patch("chemx.backends.ollama.backend.post_json")
    def test_ollama_reads_one_native_tool_call(self, post_json: Mock) -> None:
        post_json.return_value = {
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_text",
                            "arguments": {"query": "needle"},
                        }
                    }
                ]
            }
        }
        backend = OllamaBackend(model="llama3.2")
        tool = ToolDefinition(
            "search_text",
            "Search text.",
            {"type": "object"},
        )

        selected = backend.complete_tool([], [tool])

        self.assertEqual(selected.name, "search_text")
        self.assertEqual(selected.arguments, {"query": "needle"})
        self.assertIn("tools", post_json.call_args.kwargs["payload"])

    def test_registration_adds_backend_without_factory_branching(self) -> None:
        class ExampleBackend:
            context_window_tokens = 8_192

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
