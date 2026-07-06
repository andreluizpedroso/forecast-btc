import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from forecast_btc.data import load_price_history


class LoadPriceHistoryTests(unittest.TestCase):
    def test_loads_only_requested_ticker_in_date_order(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "test.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    "CREATE TABLE historico "
                    "(ativo TEXT, data DATE, fechamento REAL)"
                )
                connection.executemany(
                    "INSERT INTO historico VALUES (?, ?, ?)",
                    [
                        ("BTC-USD", "2025-01-02", 20.0),
                        ("ETH-USD", "2025-01-01", 99.0),
                        ("BTC-USD", "2025-01-01", 10.0),
                    ],
                )
                connection.commit()

            result = load_price_history("BTC-USD", database)

            self.assertEqual([10.0, 20.0], result["Close"].tolist())
            self.assertEqual("2025-01-01", result["Date"].iloc[0].date().isoformat())

    def test_rejects_unknown_ticker(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "test.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    "CREATE TABLE historico "
                    "(ativo TEXT, data DATE, fechamento REAL)"
                )
                connection.commit()

            with self.assertRaisesRegex(ValueError, "Nenhum histórico"):
                load_price_history("BTC-USD", database)


if __name__ == "__main__":
    unittest.main()
