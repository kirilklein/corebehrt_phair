import json
import os
import tempfile
import unittest
from unittest.mock import mock_open, patch

import yaml

from ehr2vec.common.config import Config, get_function, instantiate, load_config


class TestConfigClass(unittest.TestCase):
    def test_config_init(self):
        """Test initialization of Config with a dictionary."""
        data = {"a": "1", "b": {"c": "2"}}
        cfg = Config(data)
        self.assertEqual(cfg.a, 1)
        self.assertEqual(cfg.b.c, 2)
        self.assertIsInstance(cfg.b, Config)

    def test_str_to_num(self):
        """Test str_to_num method."""
        cfg = Config()
        self.assertEqual(cfg.str_to_num("10"), 10)
        self.assertEqual(cfg.str_to_num("10.5"), 10.5)
        self.assertEqual(cfg.str_to_num("abc"), "abc")

    def test_setattr(self):
        """Test setting attributes."""
        cfg = Config()
        cfg.new_attr = "100"
        self.assertEqual(cfg.new_attr, 100)
        self.assertEqual(cfg["new_attr"], 100)

    def test_setitem(self):
        """Test setting items."""
        cfg = Config()
        cfg["new_item"] = "200"
        self.assertEqual(cfg.new_item, 200)
        self.assertEqual(cfg["new_item"], 200)

    def test_delattr(self):
        """Test deleting attributes."""
        cfg = Config({"a": 1})
        del cfg.a
        self.assertFalse(hasattr(cfg, "a"))
        self.assertNotIn("a", cfg)

    def test_delitem(self):
        """Test deleting items."""
        cfg = Config({"a": 1})
        del cfg["a"]
        self.assertFalse(hasattr(cfg, "a"))
        self.assertNotIn("a", cfg)

    def test_to_dict(self):
        """Test to_dict method."""
        data = {"a": 1, "b": {"c": 2}}
        cfg = Config(data)
        expected = {"a": 1, "b": {"c": 2}}
        self.assertEqual(cfg.to_dict(), expected)

    def test_save_to_yaml(self):
        """Test saving to a YAML file."""
        cfg = Config({"a": 1, "b": {"c": 2}})
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmpfile:
            cfg.save_to_yaml(tmpfile.name)
            tmpfile.seek(0)
            content = tmpfile.read()
        loaded_cfg = yaml.safe_load(content)
        self.assertEqual(loaded_cfg, cfg.to_dict())
        os.remove(tmpfile.name)

    def test_save_pretrained(self):
        """Test saving to a JSON file."""
        cfg = Config({"a": 1, "b": {"c": 2}})
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.save_pretrained(tmpdir)
            filepath = os.path.join(tmpdir, "model_config.json")
            with open(filepath, "r") as f:
                data = json.load(f)
            self.assertEqual(data, cfg.to_dict())

    def test_update(self):
        """Test updating the config."""
        cfg1 = Config({"a": 1, "b": 2})
        cfg2 = Config({"b": 3, "c": 4})
        cfg1.update(cfg2)
        self.assertEqual(cfg1.a, 1)
        self.assertEqual(cfg1.b, 2)  # Should not be updated
        self.assertEqual(cfg1.c, 4)


class TestHelperFunctions(unittest.TestCase):
    @patch("importlib.import_module")
    def test_instantiate(self, mock_import_module):
        """Test the instantiate function."""

        class DummyClass:
            def __init__(self, x, y=0):
                self.x = x
                self.y = y

        # Create a mock module with DummyClass
        mock_module = unittest.mock.Mock()
        mock_module.DummyClass = DummyClass
        mock_import_module.return_value = mock_module

        cfg = Config({"_target_": "dummy.module.DummyClass", "x": 5})
        instance = instantiate(cfg, y=10)
        self.assertEqual(instance.x, 5)
        self.assertEqual(instance.y, 10)

    @patch("importlib.import_module")
    def test_get_function(self, mock_import_module):
        """Test the get_function function."""

        def dummy_function():
            return "dummy"

        # Create a mock module with dummy_function
        mock_module = unittest.mock.Mock()
        mock_module.dummy_function = dummy_function
        mock_import_module.return_value = mock_module

        cfg = Config({"_target_": "dummy.module.dummy_function"})
        func = get_function(cfg)
        self.assertEqual(func(), "dummy")

    def test_load_config(self):
        """Test loading a YAML configuration file."""
        yaml_content = """
        a: 1
        b:
          c: 2
        """
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            cfg = load_config("dummy_path.yaml")
            self.assertEqual(cfg.a, 1)
            self.assertEqual(cfg.b.c, 2)

    def test_load_config_nonexistent_file(self):
        """Test loading a non-existent configuration file."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):
                load_config("nonexistent.yaml")


# Run the unit tests
if __name__ == "__main__":
    unittest.main()
