import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Backtester:
    def __init__(self, initial_capital=100000, transaction_cost=0.0001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.portfolio_value = []
        self.positions = []
        self.dates = []
        self.trades = []
        self.cash = initial_capital
        self.current_position = 'flat'

    OPEN_ORDER_SIDE = {
        "long": "buy",
        "short": "sell",
    }

    CLOSE_ORDER_SIDE = {
        "long": "sell",
        "short": "buy",
    }

    POSITION_MULTIPLIER = {
        "long": 1,
        "short": -1,
    }

    def __str__(self):
        return f"Backtester(capital={self.initial_capital})"

    def run_backtest(self, data_handler, model, start_date=None, end_date=None):
        """Main backtesting loop with walk-forward analysis."""
        df = data_handler.features

        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        self.prices = data_handler.data["Close"].loc[df.index]
        self.portfolio_value = []
        self.positions = []
        self.dates = []
        self.trades = []
        self.cash = self.initial_capital
        self.current_position = 'flat'

        for i in range(len(df)):
            features = df.iloc[i:i+1, :-1].values

            prediction = model.predict(features)[0]

            if prediction == 1:
                signal = 'long'
            elif prediction == -1:
                signal = 'short'
            else:
                signal = 'flat'

            # execute at next day's open to avoid look-ahead bias
            if i + 1 < len(df):
                next_date = df.index[i + 1]
                execution_price = data_handler.data["Open"].loc[next_date]
            else:
                continue

            self._execute_trade(signal, execution_price, next_date)

            portfolio_val = self._calculate_portfolio_value(execution_price)
            self.portfolio_value.append(portfolio_val)
            self.positions.append(self.current_position)
            self.dates.append(next_date)

        return self
    

    def _execute_trade(self, signal, price, date):
        """Execute trade based on signal"""
        if signal != self.current_position: # only trade if the signal is different from current position
            if self.current_position != 'flat':
                self._close_position(price, date) # close existing position if you have one

            if signal != 'flat': # open new position if the signal is not 0
                self._open_position(signal, price, date)

    def _open_position(self, signal, price, date):
        """Open a new position"""
        portfolio_value = self._calculate_portfolio_value(price)
        position_value = portfolio_value * 0.95
        shares = position_value/price

        transaction_cost = position_value * self.transaction_cost

        if signal == 'long':
            self.cash -= (position_value + transaction_cost)
        elif signal == 'short':
            self.cash += (position_value - transaction_cost)
        else:
            return

        # record the trade
        self.trades.append({
            'date': date,
            'action': 'open',
            'order_side': self.OPEN_ORDER_SIDE[signal],
            'position_side': signal,
            'price': price,
            'shares': shares,
            'value': position_value,
            'cost': transaction_cost,
        })

        self.current_position = signal

    def _close_position(self, price, date):
        """Close current position"""
        if self.current_position == 'flat':
            return
        
        last_trade = next((t for t in reversed(self.trades) if t['action'] == 'open'), None)
        if not last_trade:
            return
        
        shares = last_trade['shares']
        position_value = shares*price
        transaction_cost = position_value * self.transaction_cost

        # keep track of profits and losses
        entry_price = last_trade["price"]
        exit_price = price
        position_side = last_trade["position_side"]

        if position_side == 'long':
            self.cash += (position_value - transaction_cost)
        elif position_side == 'short':
            self.cash -= (position_value + transaction_cost)
        else:
            return

        direction = self.POSITION_MULTIPLIER[position_side]
        gross_pnl = (exit_price - entry_price)*shares*direction
        net_pnl = gross_pnl - last_trade["cost"] - transaction_cost


        # record the trade
        self.trades.append({
            'date': date,
            'action': 'close',
            'position_side': 'flat',
            'closed_position_side': position_side,
            'order_side': self.CLOSE_ORDER_SIDE[position_side],
            'price': price,
            'shares': shares,
            'value': position_value,
            'cost': transaction_cost,
            'pnl': net_pnl
        })

        self.current_position = 'flat'

    def _calculate_portfolio_value(self, current_price):
        """Calculate portfolio value"""
        if self.current_position == 'flat':
            return self.cash
        
        # find last open trade to get shares
        last_trade = next((t for t in reversed(self.trades) if t['action'] == 'open'), None)
        if last_trade:
            current_position_value = last_trade['shares'] * current_price
            if last_trade['position_side'] == 'long':
                return self.cash + current_position_value
            if last_trade['position_side'] == 'short':
                return self.cash - current_position_value
        
        return self.cash


    def calculate_returns(self):
        """Calculate portfolio returns accounting for transaction costs"""
        if len(self.portfolio_value) < 2:
            return []
        
        portfolio_series = pd.Series(self.portfolio_value)
        returns = portfolio_series.pct_change().dropna()
        return returns

    def buy_and_hold_value(self, prices):
        initial_shares = self.initial_capital/prices.iloc[0]
        final_value = initial_shares*prices.iloc[-1]
        return final_value


    def generate_performance_report(self):
        """Comprehensive performance analysis with all the key metrics (Sharpe Ratio, Maximum Drawdown...)."""
        returns = self.calculate_returns()
        if len(returns) == 0:
            print("No returns to analyse")
            return
        
        # basic metrics
        total_return = ((self.portfolio_value[-1] / self.initial_capital) -1) * 100
        annualized_return = ((self.portfolio_value[-1] / self.initial_capital) ** (252/len(returns)) - 1) * 100

        # comparison metrics
        bh_value = self.buy_and_hold_value(self.prices)
        bh_return = ((bh_value / self.initial_capital) - 1) * 100

        # risk metrics
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        # drawdown calculation
        portfolio_series = pd.Series(self.portfolio_value)
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        # trade analysis
        closed_trades = [t for t in self.trades if t['action'] == 'close']
        winning_trades = len([t for t in closed_trades if t["pnl"] > 0])
        losing_trades = len([t for t in closed_trades if t["pnl"] < 0])
        total_trades = len(closed_trades)
        
        print("=== PERFORMANCE REPORT ===")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${self.portfolio_value[-1]:,.2f}")
        print(f"Strategy Return: {total_return:.2f}%")
        print(f"Buy & Hold Return: {bh_return:.2f}%")
        print(f"Annualized Return: {annualized_return:.2f}%")
        print(f"Volatility (annualized): {volatility:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
        }

    def plot_results(self):
        """Visualize portfolio performance, drawdowns, and trade analysis."""
        if not len(self.portfolio_value):
            print("Not enough data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        portfolio_series = pd.Series(self.portfolio_value, index=self.dates)

        ax1.plot(portfolio_series.index, portfolio_series.values, label='Portfolio Value', linewidth=2)
        ax1.axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.7, label='Initial Capital')
        ax1.set_title('Portfolio Performance Over Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max * 100

        ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax2.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        ax2.set_title('Drawdown Over Time')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
