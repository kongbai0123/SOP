"""Release metadata compatibility without a compiler, model, camera or real payload."""
from __future__ import annotations

import ast
from contextlib import ExitStack, redirect_stdout
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sop_app.release_version import parse_release_version, windows_file_version, windows_version_tuple


ROOT = Path(__file__).resolve().parents[1]


class ReleaseVersionTests(unittest.TestCase):
    def test_three_and_four_part_versions_keep_the_same_numeric_identity(self):
        examples = [('2026.10.19', (2026, 10, 19), (2026, 10, 19, 0), '2026.10.19.0'),
                    ('2026.10.19.1', (2026, 10, 19, 1), (2026, 10, 19, 1), '2026.10.19.1'),
                    ('0.0.0', (0, 0, 0), (0, 0, 0, 0), '0.0.0.0'),
                    ('0.0.0.0', (0, 0, 0, 0), (0, 0, 0, 0), '0.0.0.0'),
                    ('65535.65535.65535', (65535,) * 3, (65535, 65535, 65535, 0), '65535.65535.65535.0'),
                    ('65535.65535.65535.65535', (65535,) * 4, (65535,) * 4, '65535.65535.65535.65535'),
                    ('01.002.0003.0004', (1, 2, 3, 4), (1, 2, 3, 4), '1.2.3.4')]
        for text, parsed, padded, formatted in examples:
            with self.subTest(version=text):
                self.assertEqual(parse_release_version(text), parsed)
                self.assertEqual(windows_version_tuple(text), padded)
                self.assertEqual(windows_file_version(text), formatted)

    def test_malformed_non_numeric_and_out_of_range_values_are_rejected(self):
        bad = ['', '1', '1.2', '1.2.3.4.5', '1..3', '1.2.3.', '.1.2.3', 'v1.2.3',
               '1.2.3-rc1', '-1.2.3', '+1.2.3', '1. 2.3', ' 1.2.3', '1.2.3 ', '1.2.3\n',
               '１.2.3', '1.٢.3', '65536.0.0', '0.65536.0', '0.0.65536', '0.0.0.65536',
               None, 123, (1, 2, 3)]
        for value in bad:
            for helper in (parse_release_version, windows_version_tuple, windows_file_version):
                with self.subTest(version=value, helper=helper.__name__):
                    with self.assertRaises(ValueError):
                        helper(value)


class ReleaseToolMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Match direct script execution so build_payload's sibling import stays
        # unchanged. Neither script runs main() when imported.
        with patch.object(sys, 'path', [str(ROOT / 'tools'), *sys.path]):
            cls.build = importlib.import_module('build_payload')
            cls.installer = importlib.import_module('prepare_installer')

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='sop-release-version-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def version_source(self, value):
        package = self.root / 'sop_app'
        package.mkdir(exist_ok=True)
        (package / '__init__.py').write_text(f'__version__ = {value!r}\n', encoding='utf-8')

    def test_both_tools_read_three_and_four_part_metadata(self):
        for version in ('2026.10.19', '2026.10.19.1'):
            self.version_source(version)
            with self.subTest(version=version), patch.object(self.build, 'ROOT', self.root), \
                    patch.object(self.installer, 'ROOT', self.root):
                self.assertEqual(self.build.read_version(), version)
                self.assertEqual(self.installer.app_version(), version)

    def test_both_tools_reject_invalid_application_metadata(self):
        for version in ('2026.10', '2026.10.19.65536', '2026.10.19rc1', 2026):
            self.version_source(version)
            with self.subTest(version=version), patch.object(self.build, 'ROOT', self.root), \
                    patch.object(self.installer, 'ROOT', self.root):
                with self.assertRaises(SystemExit):
                    self.build.read_version()
                with self.assertRaises(ValueError):
                    self.installer.app_version()

    def test_pe_resources_have_four_component_tuple_and_matching_string(self):
        source = self.root / 'sops'
        source.mkdir()
        (source / 'example_sop.json').write_text(json.dumps({'model_path': 'private.pt', 'model_version_id': 'private'}), encoding='utf-8')
        for version, expected in (('2026.10.19', (2026, 10, 19, 0)), ('2026.10.19.1', (2026, 10, 19, 1))):
            generated = self.root / version
            with self.subTest(version=version), patch.object(self.build, 'ROOT', self.root), \
                    patch.object(self.build, 'GENERATED', generated), patch.object(self.build, 'collect_notices'):
                self.build.generate_resources(version)
            tree = ast.parse((generated / 'version-info.txt').read_text(encoding='utf-8'))
            fixed = next(node for node in ast.walk(tree) if isinstance(node, ast.Call) and
                         isinstance(node.func, ast.Name) and node.func.id == 'FixedFileInfo')
            fields = {keyword.arg: ast.literal_eval(keyword.value) for keyword in fixed.keywords}
            strings = {ast.literal_eval(node.args[0]): ast.literal_eval(node.args[1])
                       for node in ast.walk(tree) if isinstance(node, ast.Call) and
                       isinstance(node.func, ast.Name) and node.func.id == 'StringStruct'}
            self.assertEqual(fields['filevers'], expected)
            self.assertEqual(fields['prodvers'], expected)
            self.assertEqual(strings['FileVersion'], '.'.join(map(str, expected)))
            self.assertEqual(strings['ProductVersion'], version)

    def test_invalid_version_fails_before_creating_resource_output(self):
        generated = self.root / 'never-created'
        with patch.object(self.build, 'GENERATED', generated):
            with self.assertRaises(ValueError):
                self.build.generate_resources('2026.10.19.65536')
        self.assertFalse(generated.exists())

    def test_nsis_define_and_release_manifest_use_the_same_four_part_file_version(self):
        payload = self.root / 'payload'
        payload.mkdir()
        (payload / 'SOP.exe').write_bytes(b'inert unit-test payload; never executed')
        installer = self.root / 'installer'
        (installer / '.build/generated').mkdir(parents=True)
        (installer / '.build/generated/SOP.ico').write_bytes(b'unit-test icon placeholder')
        (installer / 'README.md').write_text('Unit-test installer documentation', encoding='utf-8')
        (self.root / 'tools').mkdir()
        (self.root / 'tools/check_windows_trust.ps1').write_text('# unit-test fixture', encoding='utf-8')
        config = {'productId': 'SOP.WorkInstruction.Smoke.VersionTest', 'productName': 'Version Unit Test',
                  'entryExe': 'SOP.exe', 'publisher': 'Unit test', 'nsisVersion': 'mock',
                  'architecture': 'x64', 'installScope': 'current-user', 'channel': 'unit-test',
                  'signingPolicy': 'local-test', 'updateMode': 'manual', 'distribution': 'offline'}
        for version, expected in (('2026.10.19', '2026.10.19.0'), ('2026.10.19.1', '2026.10.19.1')):
            output = self.root / version
            args = SimpleNamespace(payload=payload, output=output, rebuild=False)
            setup = output / f'SOP-{version}-windows-x64-Setup.exe'
            with self.subTest(version=version), ExitStack() as mocks:
                for patcher in (patch.object(self.installer, 'ROOT', self.root),
                                patch.object(self.installer, 'INSTALLER', installer),
                                patch.object(self.installer, 'compiler', return_value=Path('mock-compiler-never-executed')),
                                patch.object(self.installer.subprocess, 'run', side_effect=lambda *_a, **_k: setup.write_bytes(b'mock compiler output; not an installer')),
                                patch.object(self.installer.subprocess, 'check_output', side_effect=['test-commit\n', '']),
                                patch.object(self.installer.importlib.metadata, 'version', return_value='unit-test')):
                    mocks.enter_context(patcher)
                with redirect_stdout(io.StringIO()):
                    self.installer.prepare(args, config, version)
            header = (installer / '.build' / ('nsis-' + config['productId']) / 'Product.nsh').read_text(encoding='utf-8')
            manifest = json.loads((output / 'release-manifest.json').read_text(encoding='utf-8'))
            self.assertIn(f'!define PRODUCT_VERSION "{version}"', header)
            self.assertIn(f'!define FILE_VERSION "{expected}"', header)
            self.assertEqual(manifest['version'], version)
            self.assertEqual(manifest['fileVersion'], expected)

    def test_invalid_version_fails_before_installer_io_or_compiler(self):
        args = SimpleNamespace(payload=self.root / 'missing-payload', output=self.root / 'never-created', rebuild=False)
        with patch.object(self.installer, 'compiler') as compiler:
            with self.assertRaises(ValueError):
                self.installer.prepare(args, {}, '2026.10.19.65536')
        compiler.assert_not_called()
        self.assertFalse(args.output.exists())


if __name__ == '__main__':
    unittest.main()
