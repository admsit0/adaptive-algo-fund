import matplotlib.pyplot as plt
import pandas as pd

def plot_equity_curve(df, benchmark=None, title='Equity Curve'):
	plt.figure(figsize=(10,5))
	plt.plot(df['equity'], label='Strategy')
	if benchmark is not None:
		plt.plot(benchmark, label='Benchmark')
	plt.title(title)
	plt.xlabel('Date')
	plt.ylabel('Equity')
	plt.legend()
	plt.grid()
	plt.tight_layout()
	plt.show()

def plot_drawdown(df, title='Drawdown'):
	plt.figure(figsize=(10,3))
	plt.plot(df['drawdown'], color='red')
	plt.title(title)
	plt.xlabel('Date')
	plt.ylabel('Drawdown')
	plt.grid()
	plt.tight_layout()
	plt.show()
