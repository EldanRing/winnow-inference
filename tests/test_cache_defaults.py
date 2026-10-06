"""Resolved target/draft cache contracts across shipped models, without inference."""
import argparse
import copy
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import assets
import decision_client
import profiles
import serve
import winnow
from reasoning_contract import load_profile, check_inspection
from test_image_reasoning import inspection
from test_adaptive_pipeline import FakeTransport, body, native


class CacheDefaults(unittest.TestCase):
    def resolved(self, argv, system="Linux"):
        out = io.StringIO()
        with patch.object(sys, "argv", ["serve", *argv, "--dry-run"]), \
             patch("platform.system", return_value=system), patch("platform.machine", return_value="arm64"), \
             patch.object(serve, "verify"), patch.object(sys, "stdout", out), patch.object(sys, "stderr", io.StringIO()):
            serve.main()
        return json.loads(out.getvalue())

    def assert_cache(self, result, expected):
        args = result["command"]
        self.assertEqual(result["environment"]["WINNOW_CACHE"], expected)
        for flag in ("--cache-type-k", "--cache-type-v"):
            self.assertEqual(args[args.index(flag)+1], expected)
        if "--spec-draft-model" in args:
            for flag in ("--spec-draft-type-k", "--spec-draft-type-v"):
                self.assertEqual(args[args.index(flag)+1], "q8_0")

    def test_every_shipped_preset_defaults_to_f16_and_keeps_explicit_q8(self):
        with tempfile.TemporaryDirectory() as td:
            paths = [Path(td)/name for name in ("target", "assistant", "projector", "server")]
            for p in paths: p.touch()
            for name, preset in serve.RUNTIME_PRESETS.items():
                self.assertEqual(preset["settings"]["cache"], "f16", name)
                mtp = "on" if preset.get("resident_mtp", True) else "off"
                argv = ["--preset", name, "--mtp", mtp, "--model", str(paths[0]), "--server", str(paths[3])]
                if mtp == "on": argv += ["--assistant", str(paths[1])]
                if not preset["text_only"]: argv += ["--mmproj", str(paths[2])]
                for cache in (None, "q8_0"):
                    with self.subTest(preset=name, cache=cache):
                        result = self.resolved(argv + (["--cache", cache] if cache else []))
                        self.assert_cache(result, cache or "f16")
                        if name.startswith("e2b-") and cache == "q8_0":
                            self.assertNotIn("--backend-sampling", result["command"])

    def test_platform_defaults_and_explicit_cache_override(self):
        with tempfile.TemporaryDirectory() as td:
            model, server = Path(td)/"model", Path(td)/"server"
            model.touch(); server.touch()
            for system in ("Linux", "Darwin"):
                for cache in (None, "q8_0"):
                    argv = ["--model", str(model), "--server", str(server), "--text-only"]
                    self.assert_cache(self.resolved(argv + (["--cache", cache] if cache else []), system), cache or "f16")
            self.assertTrue(all(p["cache"] == "f16" for p in profiles.PROFILES.values()))

    def test_unified_model_launches_match_cache_and_draft_settings(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            server = root/"server"; server.touch()
            for spec in assets.MODELS.values():
                for key in ("model", "assistant", "projector"):
                    if key in spec:
                        p = root/spec[key]["file"]; p.parent.mkdir(parents=True, exist_ok=True); p.touch()
            for model, reasoning, mtp, vision, cache in itertools.product(
                    assets.MODEL_ALIASES, ("off", "selective"), ("off", "on"), ("off", "on"), ("f16", "q8_0")):
                try:
                    assets.selection(model, reasoning, mtp, vision)
                except ValueError:
                    continue
                with self.subTest(model=model, reasoning=reasoning, mtp=mtp, vision=vision, cache=cache):
                    a = argparse.Namespace(model=model, reasoning=reasoning, mtp=mtp, vision=vision,
                                           cache=cache, context=8192, model_dir=root, server=server)
                    with patch("platform.system", return_value="Linux"):
                        command = winnow.serve_command(a)
                    result = self.resolved(command[2:])
                    self.assert_cache(result, cache)

    def test_profile_cache_override_is_isolated_and_mismatches_fail(self):
        for name, preset in serve.RUNTIME_PRESETS.items():
            if preset["status"] == "blocked":
                continue
            if preset["text_only"] and not preset.get("decision_profile"):
                continue
            before = load_profile(name)
            self.assertEqual(before["runtime"]["cache_type"], "f16")
            quantized = load_profile(name, cache="q8_0")
            self.assertEqual(quantized["runtime"]["cache_type"], "q8_0")
            self.assertEqual(load_profile(name), before)
            images = not preset["text_only"]
            native = inspection(quantized, images=images)
            check_inspection(native, quantized, int(images))
            with self.assertRaisesRegex(ValueError, "cache_type"):
                check_inspection(native, before, int(images))
        with self.assertRaisesRegex(ValueError, "Target cache"):
            load_profile("e2b-q8-text8k-mtp", cache="q4_0")

    def test_cli_passes_explicit_cache_to_profile_checked_client(self):
        for cache in ("f16", "q8_0"):
            argv = ["winnow", "decide", "--model", "e2b", "--cache", cache, "--input", "request.json"]
            with patch.object(sys, "argv", argv), patch.object(winnow.subprocess, "run") as run:
                winnow.main()
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--cache")+1], cache)

    def test_lower_client_uses_matching_explicit_cache_contract(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"request.json"
            request = body()
            path.write_text(json.dumps(request))
            policy = "e2b-raw99-blend50-v3"
            for cache in ("f16", "q8_0"):
                profile = load_profile("e2b-q8-text8k-mtp", cache=cache)
                transport = FakeTransport([inspection(profile, images=False), native(request, [0, 0], policy)])
                argv = ["decide", "--input", str(path), "--runtime-profile", profile["id"], "--cache", cache]
                with patch.object(sys, "argv", argv), patch.object(decision_client, "HTTPTransport", return_value=transport), \
                     patch.object(sys, "stdout", io.StringIO()):
                    decision_client.main()
                self.assertFalse(transport.replies)


if __name__ == "__main__":
    unittest.main()
