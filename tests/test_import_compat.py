from __future__ import annotations

import unittest


class TestImportCompat(unittest.TestCase):
    def test_stt_alias_import(self) -> None:
        from stt import STTService

        self.assertIsNotNone(STTService)


if __name__ == "__main__":
    unittest.main()
