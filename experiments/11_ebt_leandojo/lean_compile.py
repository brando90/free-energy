#!/usr/bin/env python3
"""Lean compile helpers for proof validation."""

from __future__ import annotations

import json
import re
import tempfile
import threading
from pathlib import Path

import pexpect


class InteractiveThread(threading.Thread):
    def __init__(
        self,
        session_id: int,
        repl_path: str,
        lean_env_path: str,
        initial_context: str | None = None,
        timeout: int = 600,
        expect_timeout: int = 120,
    ) -> None:
        super().__init__()
        self.session_id = session_id
        self.repl_path = repl_path
        self.lean_env_path = lean_env_path
        self.context = initial_context
        self.session = None
        self.expect_timeout = expect_timeout
        self.cmd_response_condition = threading.Event()
        self.cmd_query_condition = threading.Event()
        self.init_complete = threading.Event()
        self.response = None
        self.stop_flag = False
        self.timer = threading.Timer(timeout, self.stop)

    def initialize_check(self) -> None:
        try:
            if self.context is None:
                self.send_cmd({"cmd": "def init_check : Nat := 42"})
            self.session.expect('"env": 0}\r\n\r\n', timeout=self.expect_timeout)
            self.init_complete.set()
        except Exception:
            self.init_complete.set()
            self.stop()

    def send_cmd(self, cmd: dict[str, object]) -> None:
        self.session.sendline(json.dumps(cmd, ensure_ascii=False) + "\n")

    def submit_and_receive(self, cmd: dict[str, object]) -> dict[str, object] | None:
        if self.stop_flag:
            return None
        self.init_complete.wait()
        self.send_cmd(cmd)
        self.cmd_query_condition.set()
        self.cmd_response_condition.wait()
        self.cmd_response_condition.clear()
        if self.response:
            output = self.response
            self.response = None
            return output
        return None

    def process_responses(self) -> None:
        while not self.stop_flag:
            self.cmd_query_condition.wait()
            self.cmd_query_condition.clear()
            if self.stop_flag:
                break
            try:
                self.session.expect("\r\n\r\n", timeout=self.expect_timeout)
                self.session.expect(["\r\n\r\n", pexpect.EOF], timeout=self.expect_timeout)
                self.response = json.loads(self.session.before.strip())
                self.cmd_response_condition.set()
            except Exception:
                self.cmd_response_condition.set()
                break

    def _remove_last_comment(self) -> None:
        if self.context is None:
            return
        self.context = re.sub(r"/--[^/]*?-/(\n*)$", "", self.context, flags=re.DOTALL)

    def run(self) -> None:
        self.timer.start()
        try:
            self.session = pexpect.spawn("bash", encoding="utf-8", cwd=self.lean_env_path)
            if self.context is not None:
                self._remove_last_comment()
                with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp:
                    json.dump({"cmd": self.context}, temp, ensure_ascii=False)
                    temp.write("\n\n")
                    temp.flush()
                command = f"lake env {self.repl_path}/.lake/build/bin/repl < <(cat {temp.name} -)"
            else:
                command = f"lake env {self.repl_path}/.lake/build/bin/repl"
            self.session.sendline(command)
            self.initialize_check()
            self.process_responses()
        finally:
            self.stop()

    def stop(self) -> None:
        self.stop_flag = True
        self.init_complete.set()
        self.cmd_query_condition.set()
        self.cmd_response_condition.set()
        if self.session is not None and self.session.isalive():
            try:
                self.session.terminate(force=True)
            except Exception:
                pass
        self.timer.cancel()


def compile_theorem(
    theorem_code: str,
    *,
    repl_path: Path,
    lean_env_path: Path,
    timeout: int = 600,
    expect_timeout: int = 120,
) -> tuple[bool, dict[str, object] | None]:
    thread = InteractiveThread(
        0,
        str(repl_path.resolve()),
        str(lean_env_path.resolve()),
        initial_context="import Mathlib",
        timeout=timeout,
        expect_timeout=expect_timeout,
    )
    thread.start()
    thread.init_complete.wait()
    try:
        outcome = thread.submit_and_receive({"cmd": theorem_code, "env": 0})
    finally:
        thread.stop()
        thread.join()
    if outcome is None:
        return False, None
    messages = outcome.get("messages", [])
    has_error = any(msg.get("severity") == "error" for msg in messages)
    has_sorry = any(msg.get("severity") == "sorries" for msg in messages) or ("sorries" in outcome)
    return (not has_error and not has_sorry), outcome
