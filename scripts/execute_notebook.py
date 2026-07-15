#!/usr/bin/env python3
"""Execute a notebook with jupyter_client and save outputs without nbformat."""

from __future__ import annotations

import json
import os
from pathlib import Path

from jupyter_client import KernelManager


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "test1.ipynb"


def output_from_message(message_type: str, content: dict) -> dict | None:
    if message_type == "stream":
        return {
            "output_type": "stream",
            "name": content["name"],
            "text": content["text"],
        }
    if message_type in {"display_data", "execute_result"}:
        output = {
            "output_type": message_type,
            "data": content.get("data", {}),
            "metadata": content.get("metadata", {}),
        }
        if message_type == "execute_result":
            output["execution_count"] = content.get("execution_count")
        return output
    if message_type == "error":
        return {
            "output_type": "error",
            "ename": content.get("ename", "Error"),
            "evalue": content.get("evalue", ""),
            "traceback": content.get("traceback", []),
        }
    return None


def main() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = "/private/tmp/crop_recommendation_mpl"

    manager = KernelManager(kernel_name="python3")
    manager.start_kernel(cwd=str(ROOT), env=environment)
    client = manager.client()
    client.start_channels()

    executed = 0
    errors = []
    try:
        client.wait_for_ready(timeout=60)
        for cell_index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue

            source = "".join(cell.get("source", []))
            cell["outputs"] = []
            cell["execution_count"] = None
            message_id = client.execute(source, stop_on_error=True)

            while True:
                message = client.get_iopub_msg(timeout=180)
                if message.get("parent_header", {}).get("msg_id") != message_id:
                    continue

                message_type = message["header"]["msg_type"]
                content = message["content"]
                if message_type == "status" and content.get("execution_state") == "idle":
                    break
                if message_type == "execute_input":
                    cell["execution_count"] = content.get("execution_count")
                    continue
                if message_type == "clear_output":
                    cell["outputs"] = []
                    continue

                output = output_from_message(message_type, content)
                if output is not None:
                    cell["outputs"].append(output)
                    if output["output_type"] == "error":
                        errors.append((cell_index, output["ename"], output["evalue"]))

            executed += 1

        NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)

    if errors:
        raise RuntimeError(f"Notebook execution failed: {errors}")
    print(f"Executed and saved {executed} code cells in {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
