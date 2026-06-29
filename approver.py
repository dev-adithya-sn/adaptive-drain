"""Human-in-the-loop approval gate between the LLM decision and execution.

Sits in the pipeline's background LLM worker: every proposed decision is shown
to the operator, who can accept it, reject it (fall back to 'keep'), or edit it
field-by-field. A module-level lock serialises prompts so concurrent worker
decisions never interleave on the terminal. Stdlib only.
"""

from __future__ import annotations

import threading

_BAR = "━" * 47

_VALID_DECISIONS = ("keep", "merge", "split", "reset")


class HumanApprover:
    """Interactively confirms/modifies/rejects LLM decisions before they execute."""

    def __init__(self, auto_approve: bool = False) -> None:
        self._auto_approve = auto_approve
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def review(self, item: dict, decision: dict) -> dict:
        """Present the decision to the operator; return the (possibly edited) decision.

        With auto_approve=True the decision is returned unchanged and no prompt
        is shown — useful for CI / tests / unattended runs.
        """
        if self._auto_approve:
            return decision

        # Serialise prompts: the LLM worker may surface several decisions at once.
        with self._lock:
            self._render(item, decision)
            try:
                return self._prompt_action(decision)
            except EOFError:
                # stdin closed / non-interactive: don't loop forever — fall back
                # to the safe, no-op decision.
                print("(no input stream — defaulting to keep)")
                return {"decision": "keep", "reasoning": "no_input"}

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, item: dict, decision: dict) -> None:
        print(f"\n{_BAR}")
        print(f"[APPROVAL REQUIRED] cluster={item.get('cluster_id')} type={item.get('type')}")
        print()
        print(f'Template:   "{item.get("template", "")}"')
        samples = item.get("samples") or []
        print("Sample logs (up to 3):")
        if samples:
            for s in samples[:3]:
                print(f"  • {s}")
        else:
            print("  (none)")
        print()
        print(f"LLM decision: {decision.get('decision')}")
        print(f"LLM reasoning: {decision.get('reasoning', '')}")

        kind = decision.get("decision")
        if kind == "merge":
            print(f"  Merge target: cluster={decision.get('target_cluster_id')}")
            print(f'  Merged template: "{decision.get("merged_template", "")}"')
        elif kind == "split":
            print("  Sub-templates:")
            for i, sub in enumerate(decision.get("sub_templates") or []):
                print(f'    [{i}] "{sub}"')
        elif kind == "reset":
            print(f'  Reset to: "{decision.get("reset_template", "")}"')
        print(_BAR)
        print("Options:")
        print("  [y] Accept LLM decision")
        print("  [n] Reject → keep template as-is")
        print("  [e] Edit decision manually")

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------

    def _prompt_action(self, decision: dict) -> dict:
        while True:
            choice = self._input("Your choice (y/n/e): ").strip().lower()
            if choice == "y":
                return decision
            if choice == "n":
                return {"decision": "keep", "reasoning": "user_rejected"}
            if choice == "e":
                return self._edit(decision)
            print("  ! Please enter 'y', 'n', or 'e'.")

    def _edit(self, decision: dict) -> dict:
        while True:
            print("\nEdit menu:")
            print("  [1] Change decision (keep/merge/split/reset)")
            print("  [2] Edit merged_template")
            print("  [3] Edit reset_template")
            print("  [4] Edit sub_templates")
            print("  [5] Edit reasoning")
            print("  [0] Done editing")
            choice = self._input("Select field to edit: ").strip()

            if choice == "0":
                return decision
            elif choice == "1":
                value = self._input("New decision (keep/merge/split/reset): ").strip().lower()
                if value in _VALID_DECISIONS:
                    decision["decision"] = value
                else:
                    print(f"  ! Invalid decision; must be one of {_VALID_DECISIONS}.")
            elif choice == "2":
                decision["merged_template"] = self._input("New merged_template: ")
            elif choice == "3":
                decision["reset_template"] = self._input("New reset_template: ")
            elif choice == "4":
                decision["sub_templates"] = self._read_list()
            elif choice == "5":
                decision["reasoning"] = self._input("New reasoning: ")
            else:
                print("  ! Please choose 0-5.")

    def _read_list(self) -> list[str]:
        """Read sub-templates one per line until a blank line is entered."""
        print("  Enter sub-templates one per line; blank line to finish:")
        items: list[str] = []
        while True:
            line = self._input("  sub> ")
            if line.strip() == "":
                break
            items.append(line)
        return items

    # ------------------------------------------------------------------
    # Input helper (isolates input() so EOF/non-interactive runs don't hang)
    # ------------------------------------------------------------------

    @staticmethod
    def _input(prompt: str) -> str:
        # EOFError propagates to review(), which converts it to a safe 'keep'
        # fallback so non-interactive runs never hang re-prompting.
        return input(prompt)


class WebApprover:
    """Non-blocking approver for web UI. Stores batch decisions for UI display."""

    def __init__(self):
        self._pending  = {}
        self._batches  = {}
        self._lock     = threading.Lock()

    def review(self, item: dict, decision: dict) -> dict:
        """Block the LLM worker thread until the web user responds."""
        import uuid
        decision_id = str(uuid.uuid4())
        event = threading.Event()

        with self._lock:
            self._pending[decision_id] = {
                "id": decision_id,
                "item": item,
                "decision": decision,
                "event": event,
                "result": None,
            }

        event.wait(timeout=300)

        with self._lock:
            entry = self._pending.pop(decision_id, None)

        if entry and entry["result"] is not None:
            return entry["result"]
        return {"decision": "keep", "reasoning": "web_timeout"}

    def get_pending(self) -> list:
        """Return all pending decisions for the web UI (without threading internals)."""
        with self._lock:
            return [
                {
                    "id": v["id"],
                    "item": {
                        "cluster_id": v["item"].get("cluster_id"),
                        "type": v["item"].get("type"),
                        "template": v["item"].get("template"),
                        "samples": (v["item"].get("samples") or [])[:3],
                    },
                    "decision": v["decision"],
                }
                for v in self._pending.values()
            ]

    # ------------------------------------------------------------------
    # Batch API (new flow)
    # ------------------------------------------------------------------

    def set_batch(self, batch_id: str, decisions: list[dict], templates: list[dict]) -> None:
        """Store a batch of LLM decisions for UI display."""
        with self._lock:
            self._batches[batch_id] = {
                "id":        batch_id,
                "decisions": decisions,
                "templates": {t["cluster_id"]: t for t in templates},
                "approved":  False,
            }

    def get_batch(self, batch_id: str) -> list[dict]:
        with self._lock:
            b = self._batches.get(batch_id)
            return b["decisions"] if b else []

    def get_all_batches(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "batch_id":  b["id"],
                    "count":     len(b["decisions"]),
                    "decisions": b["decisions"],
                    "approved":  b["approved"],
                }
                for b in self._batches.values()
            ]

    def clear_batch(self, batch_id: str) -> None:
        with self._lock:
            self._batches.pop(batch_id, None)

    def update_decision(self, batch_id: str, cluster_id: str, updated: dict) -> bool:
        """Update a single decision within a batch before approval."""
        with self._lock:
            b = self._batches.get(batch_id)
            if not b:
                return False
            for dec in b["decisions"]:
                if str(dec.get("cluster_id")) == str(cluster_id):
                    dec.update(updated)
                    return True
            return False

    # ------------------------------------------------------------------
    # Legacy per-decision API (kept for backward compatibility)
    # ------------------------------------------------------------------

    def respond(self, decision_id: str, action: str, edited_decision=None) -> bool:
        """Called by Flask route when user responds.
        action: 'accept' | 'reject' | 'edit'
        Returns True if found, False if not found (already timed out).
        """
        with self._lock:
            entry = self._pending.get(decision_id)
            if not entry:
                return False

            if action == "accept":
                entry["result"] = entry["decision"]
            elif action == "reject":
                entry["result"] = {"decision": "keep", "reasoning": "user_rejected"}
            elif action == "edit" and edited_decision:
                entry["result"] = edited_decision
            else:
                entry["result"] = {"decision": "keep", "reasoning": "user_rejected"}

            entry["event"].set()
            return True
