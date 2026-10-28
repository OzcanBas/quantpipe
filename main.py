from src.backtest import Backtester
from src.train import train_model
from src.utils import set_seed
from src.data import DataHandler


if __name__ == "__main__":
    set_seed(42)
    windows = [5, 10, 20]
    start_date = "2015-01-01"
    end_date = "2025-01-01"
    split_date = "2023-01-01"
    target_horizon = 1
    model, data_handler = train_model(
        target_ticker='AAPL',
        windows=windows,
        start_date=start_date,
        end_date=end_date,
        split_date=split_date,
        target_horizon=target_horizon,
    )

    market_index = DataHandler('^GSPC', start=split_date, end=end_date)
    market_index.download_data()

    backtester = Backtester()
    backtester.run_backtest(data_handler, model, start_date=split_date)

    backtester.generate_performance_report(market_index)
    backtester.plot_results()
