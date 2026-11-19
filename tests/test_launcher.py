import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from launcher import Download, download, extract_source, instance_lock


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.content = b"verified asset bytes"
        self.asset = Download("model.safetensors", "https://example.com/model",
                              hashlib.sha256(self.content).hexdigest(), len(self.content))

    def test_resumes_then_reuses_verified_download(self):
        partial = self.directory / "model.safetensors.part"
        partial.write_bytes(self.content[:4])
        with patch("launcher.urlopen", return_value=Response(
            self.content[4:], 206, {"Content-Range": f"bytes 4-19/{len(self.content)}"},
        )) as request:
            result = download(self.asset, self.directory)
            self.assertEqual(request.call_args.args[0].get_header("Range"), "bytes=4-")
        self.assertEqual(result.read_bytes(), self.content)
        self.assertFalse(partial.exists())
        with patch("launcher.urlopen", side_effect=AssertionError("downloaded again")):
            self.assertEqual(download(self.asset, self.directory), result)

    def test_rejects_corruption(self):
        with patch("launcher.urlopen", side_effect=lambda *args, **kwargs: Response(
            b"x" * len(self.content),
        )), patch("launcher.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "Checksum mismatch"):
                download(self.asset, self.directory)
        self.assertFalse((self.directory / self.asset.name).exists())

    def test_installs_archive_without_overwriting_existing_install(self):
        archive = self.directory / "source.zip"
        destination = self.directory / "installed"
        with ZipFile(archive, "w") as contents:
            contents.writestr("folder/main.py", "print('ready')")
        extract_source(archive, destination)
        source = destination / "main.py"
        self.assertEqual(source.read_text(), "print('ready')")
        source.write_text("existing")
        extract_source(archive, destination)
        self.assertEqual(source.read_text(), "existing")

    def test_instance_lock_releases_after_use(self):
        with instance_lock(self.directory):
            with self.assertRaisesRegex(RuntimeError, "already running"), instance_lock(self.directory):
                self.fail("Second instance acquired the lock.")
        with instance_lock(self.directory):
            self.assertTrue((self.directory / "launcher.lock").is_file())
