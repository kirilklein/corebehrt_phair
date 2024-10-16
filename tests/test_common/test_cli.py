import sys
import unittest
from unittest.mock import patch

from ehr2vec.common.cli import override_config_from_cli, parse_cli_args, set_nested_attr
from ehr2vec.common.config import Config


# Unit tests
class TestConfigOverride(unittest.TestCase):
    def test_parse_cli_args_normal(self):
        """Test parsing normal key=value arguments."""
        test_args = ["script.py", "key1=value1", "key2=value2"]
        with patch.object(sys, "argv", test_args):
            overrides = parse_cli_args()
            expected = [("key1", "value1", False), ("key2", "value2", False)]
            self.assertEqual(overrides, expected)

    def test_parse_cli_args_new_attributes(self):
        """Test parsing new attributes with '+' prefix."""
        test_args = ["script.py", "+newkey=newvalue"]
        with patch.object(sys, "argv", test_args):
            overrides = parse_cli_args()
            expected = [("newkey", "newvalue", True)]
            self.assertEqual(overrides, expected)

    def test_parse_cli_args_invalid_format(self):
        """Test that invalid argument formats raise ValueError."""
        test_args = ["script.py", "invalid_argument"]
        with patch.object(sys, "argv", test_args):
            with self.assertRaises(ValueError):
                parse_cli_args()

    def test_set_nested_attr_existing(self):
        """Test setting an existing attribute."""
        cfg = Config({"existing": "old_value"})
        set_nested_attr(cfg, "existing", "new_value")
        self.assertEqual(cfg.existing, "new_value")

    def test_set_nested_attr_new_not_allowed(self):
        """Test that setting a new attribute without allow_new raises AttributeError."""
        cfg = Config()
        with self.assertRaises(AttributeError):
            set_nested_attr(cfg, "new_attr", "value")

    def test_set_nested_attr_new_allowed(self):
        """Test setting a new attribute with allow_new=True."""
        cfg = Config()
        set_nested_attr(cfg, "new_attr", "value", allow_new=True)
        self.assertEqual(cfg.new_attr, "value")

    def test_set_nested_attr_nested_existing(self):
        """Test setting an existing nested attribute."""
        cfg = Config({"level1": {"level2": "old_value"}})
        set_nested_attr(cfg, "level1.level2", "new_value")
        self.assertEqual(cfg.level1.level2, "new_value")

    def test_set_nested_attr_nested_new_not_allowed(self):
        """Test that setting a new nested attribute without allow_new raises AttributeError."""
        cfg = Config({"level1": {}})
        with self.assertRaises(AttributeError):
            set_nested_attr(cfg, "level1.new_attr", "value")

    def test_set_nested_attr_nested_new_allowed(self):
        """Test setting a new nested attribute with allow_new=True."""
        cfg = Config({"level1": {}})
        set_nested_attr(cfg, "level1.new_attr", "value", allow_new=True)
        self.assertEqual(cfg.level1.new_attr, "value")

    def test_override_config_from_cli_existing(self):
        """Test overriding existing configuration values."""
        cfg = Config({"param1": "old_value", "nested": {"param2": 2}})
        test_args = ["script.py", "param1=new_value", "nested.param2=3"]
        with patch.object(sys, "argv", test_args):
            override_config_from_cli(cfg)
            self.assertEqual(cfg.param1, "new_value")
            self.assertEqual(cfg.nested.param2, 3)

    def test_override_config_from_cli_new_attribute(self):
        """Test adding a new attribute using '+' prefix."""
        cfg = Config({"param1": "value1"})
        test_args = ["script.py", "+new_param=42"]
        with patch.object(sys, "argv", test_args):
            override_config_from_cli(cfg)
            self.assertEqual(cfg.new_param, 42)

    def test_override_config_from_cli_type_conversion(self):
        """Test that values are converted to appropriate types."""
        cfg = Config({"int_param": "0", "float_param": "0.0"})
        test_args = ["script.py", "int_param=10", "float_param=5.5"]
        with patch.object(sys, "argv", test_args):
            override_config_from_cli(cfg)
            self.assertEqual(cfg.int_param, 10)
            self.assertEqual(cfg.float_param, 5.5)
            self.assertIsInstance(cfg.int_param, int)
            self.assertIsInstance(cfg.float_param, float)

    def test_override_config_from_cli_error_on_nonexistent(self):
        """Test that overriding a nonexistent attribute without '+' raises AttributeError."""
        cfg = Config({"param1": "value1"})
        test_args = ["script.py", "nonexistent_param=123"]
        with patch.object(sys, "argv", test_args):
            with self.assertRaises(AttributeError):
                override_config_from_cli(cfg)

    def test_override_config_from_cli_multiple_overrides(self):
        """Test multiple overrides including new and existing attributes."""
        cfg = Config({"param1": "value1", "nested": {"param2": 2}})
        test_args = [
            "script.py",
            "param1=new_value1",
            "+param3=3",
            "nested.param2=4",
            "+nested.param4=4",
        ]
        with patch.object(sys, "argv", test_args):
            override_config_from_cli(cfg)
            self.assertEqual(cfg.param1, "new_value1")
            self.assertEqual(cfg.param3, 3)
            self.assertEqual(cfg.nested.param2, 4)
            self.assertEqual(cfg.nested.param4, 4)

    def test_override_config_from_cli_invalid_argument(self):
        """Test that invalid command-line arguments raise ValueError."""
        cfg = Config()
        test_args = ["script.py", "invalid_argument"]
        with patch.object(sys, "argv", test_args):
            with self.assertRaises(ValueError):
                override_config_from_cli(cfg)

    def test_override_config_from_cli_nested_new_attribute(self):
        """Test adding a new nested attribute using '+' prefix."""
        cfg = Config({"level1": {}})
        test_args = ["script.py", "+level1.level2=42"]
        with patch.object(sys, "argv", test_args):
            override_config_from_cli(cfg)
            self.assertEqual(cfg.level1.level2, 42)


# Run the unit tests
if __name__ == "__main__":
    unittest.main()
