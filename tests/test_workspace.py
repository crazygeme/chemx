import subprocess
import tempfile
import unittest
from pathlib import Path

from chemx.core.coding import ActionKind, CodingAction, LocalWorkspace


class LocalWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        (self.root / "example.py").write_text("value = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "example.py"], cwd=self.root, check=True)
        self.command = (
            "python3",
            "-c",
            "from example import value; assert value == 2",
        )
        self.approvals: list[tuple[str, ...]] = []
        self.workspace = LocalWorkspace(
            self.root,
            command_approval=self._approve,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_read_replace_command_and_diff(self) -> None:
        read = self.workspace.execute(
            CodingAction(ActionKind.READ_FILE, path="example.py")
        )
        replace = self.workspace.execute(
            CodingAction(
                ActionKind.REPLACE_TEXT,
                path="example.py",
                old_text="value = 1",
                new_text="value = 2",
            )
        )
        command = self.workspace.execute(
            CodingAction(
                ActionKind.RUN_COMMAND,
                command=self.command,
            )
        )

        self.assertTrue(read.success)
        self.assertEqual(read.output, "value = 1\n")
        self.assertTrue(replace.success)
        self.assertTrue(command.success)
        self.assertEqual(self.approvals, [self.command])
        self.assertIn("+value = 2", self.workspace.changes())

    def test_replace_requires_one_exact_match(self) -> None:
        result = self.workspace.execute(
            CodingAction(
                ActionKind.REPLACE_TEXT,
                path="example.py",
                old_text="missing",
                new_text="replacement",
            )
        )

        self.assertFalse(result.success)
        self.assertIn("found 0", result.output)

    def test_path_cannot_escape_workspace(self) -> None:
        result = self.workspace.execute(
            CodingAction(ActionKind.READ_FILE, path="../outside.txt")
        )

        self.assertFalse(result.success)
        self.assertIn("escapes workspace", result.output)

    def test_create_file_refuses_overwrite(self) -> None:
        result = self.workspace.execute(
            CodingAction(
                ActionKind.CREATE_FILE,
                path="example.py",
                content="replacement",
            )
        )

        self.assertFalse(result.success)
        self.assertIn("Refusing to overwrite", result.output)

    def test_create_file_result_retains_content_for_internal_review(self) -> None:
        result = self.workspace.execute(
            CodingAction(
                ActionKind.CREATE_FILE,
                path="docs/guide.md",
                content="# Guide\n\nPrivate draft.",
            )
        )

        self.assertTrue(result.success)
        self.assertIn("Document content retained", result.output)
        self.assertIn("# Guide\n\nPrivate draft.", result.output)

    def test_list_files_action_honors_requested_directory(self) -> None:
        (self.root / "src" / "chemx").mkdir(parents=True)
        (self.root / "src" / "chemx" / "agent.py").write_text(
            "value = 1\n",
            encoding="utf-8",
        )
        (self.root / ".venv" / "bin").mkdir(parents=True)
        (self.root / ".venv" / "bin" / "activate").write_text(
            "generated\n",
            encoding="utf-8",
        )

        result = self.workspace.execute(
            CodingAction(ActionKind.LIST_FILES, path="src/chemx")
        )

        self.assertTrue(result.success)
        self.assertEqual(result.output, "src/chemx/agent.py")

    def test_list_files_action_honors_gitignore(self) -> None:
        (self.root / ".gitignore").write_text(".venv/\n", encoding="utf-8")
        (self.root / ".venv" / "bin").mkdir(parents=True)
        (self.root / ".venv" / "bin" / "activate").write_text(
            "generated\n",
            encoding="utf-8",
        )

        result = self.workspace.execute(CodingAction(ActionKind.LIST_FILES))

        self.assertTrue(result.success)
        self.assertNotIn(".venv", result.output)

    def test_command_requires_approval(self) -> None:
        workspace = LocalWorkspace(
            self.root,
            command_approval=lambda command: False,
        )
        result = workspace.execute(
            CodingAction(
                ActionKind.RUN_COMMAND,
                command=("python3", "-c", "print('not approved')"),
            )
        )

        self.assertFalse(result.success)
        self.assertIn("not approved", result.output)

    def _approve(self, command: tuple[str, ...]) -> bool:
        self.approvals.append(command)
        return True

    def test_git_status_action_uses_optional_git_tool(self) -> None:
        result = self.workspace.execute(CodingAction(ActionKind.GIT_STATUS))

        self.assertTrue(result.success)
        self.assertIn("example.py", result.output)


class NonGitWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "notes.txt").write_text("alpha\n", encoding="utf-8")
        self.workspace = LocalWorkspace(self.root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_non_git_folder_supports_read_edit_and_change_summary(self) -> None:
        inspection = self.workspace.inspect("Update notes")
        edit = self.workspace.execute(
            CodingAction(
                ActionKind.REPLACE_TEXT,
                path="notes.txt",
                old_text="alpha",
                new_text="beta",
            )
        )

        self.assertIn("Workspace type: directory", inspection)
        self.assertTrue(edit.success)
        self.assertEqual(self.workspace.changes(), "modified: notes.txt")

    def test_git_status_fails_honestly_outside_git_repository(self) -> None:
        result = self.workspace.execute(CodingAction(ActionKind.GIT_STATUS))

        self.assertFalse(result.success)
        self.assertIn("not a Git repository", result.output)


if __name__ == "__main__":
    unittest.main()
