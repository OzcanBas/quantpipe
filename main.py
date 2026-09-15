from src.backtest import Backtester
from src.train import train_model
from src.utils import set_seed


if __name__ == "__main__":
    set_seed(42)
    windows = [5, 10, 20]
    model, data_handler, split_date = train_model(target_ticker='AAPL', windows=windows)

    backtester = Backtester()
    backtester.run_backtest(data_handler, model, start_date=split_date)

    backtester.generate_performance_report()
    backtester.plot_results()
