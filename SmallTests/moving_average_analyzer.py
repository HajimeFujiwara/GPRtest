
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from dataclasses import dataclass

@dataclass
class DataPreprocessor:
    filename: str

    def __post_init__(self):
        self.df = pd.read_csv(self.filename)
        self.df = self.df.rename(columns={'日付け': 'Date'})
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.set_index('Date')
        self.df = self.df.rename(columns={'終値': 'Closing'})
        self.df = self.df.sort_values('Date')

    def calculate_moving_average(self, n: int, m: int):
        # n日間の移動平均の計算
        self.df['Moving_Average'] = self.df['Closing'].rolling(window=n).mean()

        # m日間のリターンの計算（移動平均に対して）
        self.df['Return'] = self.df['Moving_Average'].pct_change(periods=m)
        # self.df['Return'] = self.df['Moving_Average'].shift(m) / self.df['Moving_Average'] - 1
        # 計算結果をメンバ変数に保持
        self.moving_average = self.df['Moving_Average']
        self.moving_average_return = self.df['Return']
        self.mlt = np.sqrt(m/20)

    def plot_data(self):
        if self.moving_average is None or self.moving_average_return is None:
            raise ValueError("移動平均と移動平均リターンが計算されていません。calculate_moving_average関数を呼び出してください。")

        # データに含まれているすべての年を抽出
        years = self.df.index.year.unique()

        # 各年のデータをプロット
        for year in sorted(years, reverse=True):
            self.plot_data_by_year(year)

    def plot_data_by_year(self, year: int):
        if self.moving_average is None or self.moving_average_return is None:
            raise ValueError("移動平均と移動平均リターンが計算されていません。calculate_moving_average関数を呼び出してください。")

        # 指定した年のデータを抽出
        year_data = self.df[self.df.index.year == year]

        # グラフの作成
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1x3のサブプロットを作成

        # 為替レートの推移
        axs[0].plot(year_data['Closing'])
        axs[0].set_title(f'{year} Exchange Rate')
        axs[0].set_xlabel('Date')
        axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%m'))  # x軸のラベルを年に設定
        axs[0].set_ylabel('Exchange Rate')
        axs[0].set_ylim([75,150])
        axs[0].grid(True)

        # リターンの推移
        axs[1].plot(year_data['Return'])
        axs[1].set_title(f'{year} Return')
        axs[1].set_xlabel('Date')
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%m'))  # x軸のラベルを年に設定
        axs[1].set_ylabel('Return')
        axs[1].set_ylim([-0.08*self.mlt, 0.08*self.mlt])
        axs[1].grid(True)

        # リターンのヒストグラム
        axs[2].hist(year_data['Return'].dropna(), bins=20,range=[-0.08*self.mlt,0.08*self.mlt])  # NaNを除去
        axs[2].set_title(f'{year} Return Histogram')
        axs[2].set_xlabel('Return')
        axs[2].set_ylabel('Frequency')
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()
