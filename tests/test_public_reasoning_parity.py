"""Audit public-model behavior against the sealed published d4631fb source.

The external baseline is deliberately optional for ordinary repository users.
Set WINNOW_PUBLIC_SOURCE to the verified published source extraction to run it.
"""
import copy
import importlib.util
import itertools
import json
import os
from pathlib import Path
import unittest

from test_adaptive_pipeline import FakeTransport, body, generation, identity, native
from adaptive_policy import DecisionPipeline, load_policy


PUBLIC = os.environ.get("WINNOW_PUBLIC_SOURCE")


@unittest.skipUnless(PUBLIC, "Set WINNOW_PUBLIC_SOURCE for the sealed public comparison")
class PublicReasoningParity(unittest.TestCase):
    def test_all_public_recipes_keep_prompts_augmentation_mapping_and_math(self):
        spec = importlib.util.spec_from_file_location(
            "sealed_public_adaptive", Path(PUBLIC or "") / "scripts/adaptive_policy.py")
        public = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(public)
        import hashlib
        public_runtime = hashlib.sha256((Path(PUBLIC) / "runtime.lock.json").read_bytes()).hexdigest()
        policies = ("nvfp4-entropy-v1", "q8-fixed50-v1",
                    "e4b-calibrated50-v1", "e4b-calibrated75-g95-v1")
        count = 0
        for policy, kind, state, route, mtp in itertools.product(
                policies, ("choice", "noul", "score"),
                ("plain <marker> text", {"model_reasoning": "owner value", "items": [1]}, ["record", {"x": 2}]),
                (False, True), ("off", "on")):
            with self.subTest(policy=policy, kind=kind, state=state, route=route, mtp=mtp):
                request = body(kind, copy.deepcopy(state))
                n = 3 if kind == "score" else 2
                logits = [0] * n if route else [100] + [0] * (n - 1)
                replies = [identity(policy), native(request, logits, policy)]
                replies[0]["runtime"]["chat_context"] = 8192
                replies[0]["runtime"]["resident_mtp"] = mtp == "on"
                if mtp == "off":
                    replies[0]["runtime"]["assistant_sha256"] = None
                replies[1]["winnow"]["resident_speculative_chat"] = mtp == "on"
                if route:
                    replies += [generation(), native(request, list(range(n)), policy)]
                    replies[-1]["winnow"]["resident_speculative_chat"] = mtp == "on"
                old_replies = copy.deepcopy(replies)
                old_replies[0]["runtime"]["runtime_sha256"] = public_runtime
                old, new = FakeTransport(old_replies), FakeTransport(replies)
                before = public.DecisionPipeline(old, "experimental-adaptive", policy, mtp).decide(request)
                after = DecisionPipeline(new, "experimental-adaptive", policy, mtp).decide(request)
                self.assertEqual(before["answers"], after["answers"])
                self.assertEqual(before["usage"], after["usage"])
                self.assertEqual(before["winnow"]["adaptive"]["completed_blend"], route)
                self.assertEqual(after["winnow"]["adaptive"]["completed_blend"], route)
                calls = copy.deepcopy(new.calls)
                if route:
                    self.assertEqual(calls[2][1].pop("chat_template_kwargs"), {"enable_thinking": False})
                self.assertEqual(old.calls, calls)
                self.assertEqual(public.load_policy(policy)[1]["policy"], load_policy(policy)[1]["policy"])
                count += 1
        self.assertEqual(count, 144)


if __name__ == "__main__":
    unittest.main()
