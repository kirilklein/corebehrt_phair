import unittest
from ehr2vec.common.utils import match_pattern, match_patterns


class TestPatternMatching(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.vocabulary = {
            "CONTROL": 1000,
            "CASE": 2000,
            "CONTROL_1": 1001,
            "CONTROL_2": 1002,
            "OTHER_CONTROL": 1003,
            "TEST": 3000,
        }

    def test_match_pattern_exact(self):
        """Test exact pattern matching"""
        result = match_pattern("^CONTROL$", self.vocabulary)
        self.assertEqual(result, {1000})

    def test_match_pattern_wildcard(self):
        """Test pattern matching with wildcards"""
        result = match_pattern("CONTROL_.*", self.vocabulary)
        self.assertEqual(set(result), {1001, 1002})

    def test_match_pattern_no_matches(self):
        """Test pattern with no matches"""
        result = match_pattern("NONEXISTENT", self.vocabulary)
        self.assertEqual(result, set())

    def test_match_pattern_partial(self):
        """Test partial pattern matching"""
        result = match_pattern(".*CONTROL.*", self.vocabulary)
        self.assertEqual(set(result), {1000, 1001, 1002, 1003})

    def test_match_patterns_multiple(self):
        """Test matching multiple patterns"""
        patterns = ["^CONTROL$", "^CASE$"]
        result = match_patterns(patterns, self.vocabulary)
        self.assertEqual(set(result), {1000, 2000})

    def test_match_patterns_overlapping(self):
        """Test matching patterns with overlapping results"""
        patterns = ["CONTROL", "CONTROL_.*"]
        result = match_patterns(patterns, self.vocabulary)
        self.assertEqual(set(result), {1000, 1001, 1002})

    def test_match_patterns_empty(self):
        """Test matching with empty pattern list"""
        patterns = []
        result = match_patterns(patterns, self.vocabulary)
        self.assertEqual(result, set())

    def test_match_patterns_no_matches(self):
        """Test matching patterns with no matches"""
        patterns = ["NONEXISTENT1", "NONEXISTENT2"]
        result = match_patterns(patterns, self.vocabulary)
        self.assertEqual(result, set())

    def tearDown(self):
        """Clean up after each test method."""
        self.vocabulary = None


if __name__ == "__main__":
    unittest.main()
