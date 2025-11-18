import unittest
from unittest.mock import MagicMock
import sys
from pathlib import Path

# --- Make project root importable ---
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simple_controller import SimpleController

class TestSimpleController(unittest.TestCase):
    def test_analyse_market_conditions(self):
        # Mock the original controller
        mock_controller = MagicMock()

        # Instantiate the simple controller with the mock
        simple_controller = SimpleController(mock_controller)

        # Call the method to be tested
        simple_controller.analyse_market_conditions()

        # Assert that the correct method was called on the original controller
        mock_controller.analyse_market_conditions.assert_called_once()

    def test_analyse_current_portfolio(self):
        # Mock the original controller
        mock_controller = MagicMock()

        # Instantiate the simple controller with the mock
        simple_controller = SimpleController(mock_controller)

        # Call the method to be tested
        simple_controller.analyse_current_portfolio()

        # Assert that the correct method was called on the original controller
        mock_controller.analyse_current_portfolio.assert_called_once()

    def test_generate_new_portfolio(self):
        # Mock the original controller
        mock_controller = MagicMock()

        # Instantiate the simple controller with the mock
        simple_controller = SimpleController(mock_controller)

        # Call the method to be tested
        simple_controller.generate_new_portfolio()

        # Assert that the correct method was called on the original controller
        mock_controller.generate_new_portfolio.assert_called_once()

if __name__ == '__main__':
    unittest.main()
