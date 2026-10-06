"""Publication bindings and download transitions, without network or inference."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import assets

REV = 'f0f9931c4d44e97091d14c0b8dba118eef32ff1b'
PREFIX = f'https://huggingface.co/EldanRing/Winnow-E2B/resolve/{REV}/gguf/'
EXPECTED = {
    'model': ('Winnow-E2B-Q8_0.gguf', 4954594336, 'cb37414b46b5c4ac54ea9a44f71bb6147991c34e6be414a8cdd359cc10f0dfb1'),
    'projector': ('mmproj-Winnow-E2B.gguf', 985653600, '7faa5282cff7250380c191bdea4e13a7b397ad851e2d883873cd9c323a5e192f'),
    'assistant': ('Gemma-4-E2B-IT-Assistant-BF16.gguf', 170194016, '0772dcc50761a47a2da87151d3105ad2b82994d9dd1dae3ad3c5c0cd26086b16'),
}

class ReleaseBindings(unittest.TestCase):
    def test_all_e2b_bindings_match_verified_distribution_and_presets(self):
        spec = assets.MODELS['e2b-q8']
        for kind, (name, size, digest) in EXPECTED.items():
            entry = spec[kind]
            self.assertEqual((entry['bytes'], entry['sha256']), (size, digest))
            self.assertEqual(entry['revision'], REV)
            self.assertEqual(entry['repository'], 'EldanRing/Winnow-E2B')
            self.assertEqual(entry['url'], PREFIX + name)
            self.assertIn(entry['availability'], {'private_verified', 'published'})
        assistant = json.loads((ROOT/'manifests/assistants-v1.json').read_text())['assets']['e2b']
        self.assertEqual(assistant['distribution_revision'], REV)
        self.assertEqual(assistant['url'], spec['assistant']['url'])
        self.assertEqual(assistant['sha256'], spec['assistant']['sha256'])
        self.assertEqual(assistant['revision'], '2d874ef7d29f9a30599a1e4b3c1cbc9595f005df')
        presets = json.loads((ROOT/'manifests/runtime-presets-v1.json').read_text())['presets']
        found = 0
        for name, preset in presets.items():
            if name.startswith('e2b-'):
                found += 1
                self.assertEqual(preset['assistant'], assistant)
                self.assertEqual(preset['target']['sha256'], spec['model']['sha256'])
                self.assertEqual(preset['settings']['cache'], 'f16')
        self.assertEqual(found, 8)

    def test_proposed_published_downloads_use_pinned_urls_and_verify_files(self):
        registry = copy.deepcopy(assets.MODELS)
        payload = b'CPU-only verified fixture'
        for kind in EXPECTED:
            registry['e2b-q8'][kind].update(availability='published', bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest())
        requested = []
        def fetch(path, artifact):
            requested.append(artifact['url'])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        with tempfile.TemporaryDirectory() as td, patch.object(assets, 'MODELS', registry), \
                patch.object(assets, 'fetch', side_effect=fetch):
            result = assets.acquire('e2b', Path(td), vision='on', mtp='on')
            self.assertEqual(requested, [PREFIX + EXPECTED[k][0] for k in EXPECTED])
            self.assertEqual(set(result), set(EXPECTED))
            for value in result.values():
                self.assertEqual(Path(value).read_bytes(), payload)

    def test_private_assistant_blocks_network_until_publication(self):
        registry = copy.deepcopy(assets.MODELS)
        for kind in EXPECTED:
            registry['e2b-q8'][kind]['availability'] = 'private_verified'
        with tempfile.TemporaryDirectory() as td, patch.object(assets, 'MODELS', registry), \
                patch.object(assets, 'fetch') as fetch:
            with self.assertRaisesRegex(ValueError, 'not published'):
                assets.acquire('e2b', Path(td), vision='on', mtp='on')
            fetch.assert_not_called()

if __name__ == '__main__':
    unittest.main()
