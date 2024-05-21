import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings

hel_25 = [ 
    "NOKIA.HE", #Nokia Oyj
    "KNEBV.HE", #KONE Oyj
    "STERV.HE", #Stora Enso Oyj
    "FORTUM.HE", #Fortum Oyj
    "TIETO.HE", #TietoEVRY Oyj
    "TYRES.HE", #Nokian Renkaat Oyj
    "METSB.HE", #Metsä Board Oyj
    "KESKOB.HE", #Kesko Oyj
    "HUH1V.HE", #Huhtamäki Oyj
    "ELISA.HE", #Elisa Oyj
    "TELIA1.HE", #Telia Company AB (publ)
    "ORNBV.HE", #Orion Oyj
    "NESTE.HE", #Neste Oyj  
    "UPM.HE", #UPM-Kymmene Oyj
    "NDA-FI.HE", #Nordea Bank Abp
    "WRT1V.HE", #Wärtsilä Oyj Abp
    "METSO.HE", #Metso Oyj
    "SAMPO.HE", #Sampo Oyj
    "KCR.HE", #Konecranes Plc
    "OUT1V.HE", #Outokumpu Oyj
    "CGCBV.HE" #Cargotec Corporation
]

nifty_50 = [
    "TATASTEEL.NS",
    "TATACONSUM.NS",
    "RELIANCE.NS",
    "HDFCLIFE.NS",
    "LT.NS",
    "TCS.NS",
    "KOTAKBANK.NS",
    "HINDALCO.NS",
    "LTIM.NS",
    "MARUTI.NS",
    "ULTRACEMCO.NS",
    "BAJFINANCE.NS",
    "WIPRO.NS",
    "BRITANNIA.NS",
    "INDUSINDBK.NS",
    "BAJAJ-AUTO.NS",
    "SHRIRAMFIN.NS",
    "BAJAJFINSV.NS",
    "ITC.NS",
    "HEROMOTOCO.NS",
    "CIPLA.NS",
    "NTPC.NS",
    "ADANIENT.NS",
    "ONGC.NS",
    "COALINDIA.NS",
    "NESTLEIND.NS",
    "TITAN.NS",
    "APOLLOHOSP.NS",
    "BHARTIARTL.NS"
]

sma_lags = {"short": 10, "long": 50}

def get_data(symbols):
    for symbol in symbols:
        data = yf.download(symbol, start="2010-01-01", end="2024-01-01")
        name = symbol.split(".")[0]
        data.to_csv(f"data/{name}.csv", sep = ";")

def get_from_csv(csv_file):
    try:
        df = pd.read_csv(csv_file, sep=";")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    return df

def calculate_log_returns(df):
    df['Percentage Change'] = df['Adj Close'].pct_change()
    df['Log Returns'] = np.log(1 + df['Percentage Change'])
    return df

def calculate_sma(df, window): 
    df[f'SMA {window}'] = df['Adj Close'].rolling(window=window).mean()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
    return df

def sma_crossover(stock_data, short, long):
    stock_data = calculate_sma(stock_data,short)
    stock_data = calculate_sma(stock_data,long)
    stock_data['SMA diff'] = stock_data[f'SMA {long}'] - stock_data[f'SMA {short}']
    stock_data['Previous SMA diff'] = stock_data['SMA diff'].shift(1)
    stock_data['SMA Log Returns'] = 0
    stock_data.loc[stock_data['Previous SMA diff'] >= 0, 'SMA Log Returns'] = stock_data.loc[stock_data['Previous SMA diff'] >= 0, 'Log Returns']
    stock_data.loc[stock_data['Previous SMA diff'] < 0, 'SMA Log Returns'] = -stock_data.loc[stock_data['Previous SMA diff'] < 0, 'Log Returns']
    
    stock_data.drop(columns=['Previous SMA diff', 'SMA diff'], inplace=True)
    return stock_data

def bollinger_bands(df):    
    df["BB upper"], df["BB lower"] = calculate_bollinger_bands(df)
    df['Previous BB upper'] = df['BB upper'].shift(1)
    df['Previous BB lower'] = df['BB lower'].shift(1)
    df['Previous Adj Close'] = df['Adj Close'].shift(1)
    df["BB Position"] = 0
    df["BB Log Returns"] = 0
    
    if df.loc[1, 'Previous Adj Close'] < df.loc[1, 'Previous BB lower']: 
        df.loc[1, 'BB Position'] = 1
    elif df.loc[1, 'Previous Adj Close'] > df.loc[1, 'Previous BB upper']:
        df.loc[1, 'BB Position'] = 2
    else:
        df.loc[1, 'BB Position'] = 0
    
    for i in range(2,len(df)-2):
        if df.loc[i, 'Previous Adj Close'] < df.loc[i, 'Previous BB lower']: 
            df.loc[i, 'BB Position'] = 1
        elif df.loc[i, 'Previous Adj Close'] > df.loc[i, 'Previous BB upper']:
            df.loc[i, 'BB Position'] = 2
        else: 
            df.loc[i, 'BB Position'] = df.loc[i-1, 'BB Position']
   
    df.loc[df['BB Position'] == 1, 'BB Log Returns'] = df.loc[df['BB Position'] == 1, 'Log Returns']
    df.loc[df['BB Position'] == 2, 'BB Log Returns'] = -df.loc[df['BB Position'] == 2, 'Log Returns']
    
    df.drop(columns=['Previous BB upper', 'Previous BB lower', 'Previous Adj Close'], inplace=True)
    return df

def filter_by_date(df,year):
    df_after = df[df['Date'] > f'{year}-01-01']
    return df_after

def calculate_cumulative_log_returns(df):
    df['Cumulative Log Returns'] = df['Log Returns'].cumsum()
    df['Cumulative SMA Log Returns'] = df['SMA Log Returns'].cumsum()
    df['Cumulative BB Log Returns'] = df['BB Log Returns'].cumsum()
    return df

def calculate_excess_returns(excess_df, stock_data, name, type):
    if 'Date' not in excess_df.columns:
        excess_df['Date'] = stock_data['Date']

    excess_df[name] = stock_data[f'Cumulative {type} Log Returns'] - stock_data['Cumulative Log Returns']
    
    return excess_df

def plot_cumulative_returns(df, name):
    plt.plot(df['Date'], df['Cumulative SMA Log Returns'], label='Cumulative SMA Log Returns')
    plt.plot(df['Date'], df['Cumulative BB Log Returns'], label='Cumulative BB Log Returns')
    plt.plot(df['Date'], df['Cumulative Log Returns'], label='Cumulative Log Returns')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title(name)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_deviation_line_chart(df, type):
    df['Average'] = df.iloc[:, 1:].mean(axis=1)
    df['Date'] = pd.to_datetime(df['Date'])
    blue = (0.2, 0.4, 0.8)
    
    for column in df.columns:
        if column != 'Date' and column != 'Average':
            plt.plot(df['Date'], df[column], color=blue, alpha=0.5, label=None, linewidth=0.8) 
    
    plt.plot(df['Date'], df['Average'], color='red', linewidth=1.2)
    
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Values', fontsize=12)
    plt.title(f'Excess Log Returns ({type})', fontsize=14, fontweight='bold')
   
    custom_legend = [Line2D([0], [0], color=blue, lw=2, label='Individual Stocks'),
                     Line2D([0], [0], color='red', lw=2, label='Average Excess Log Returns')]
    plt.legend(handles=custom_legend, loc='lower left', fontsize=10)
    
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'Figures/Final/Excess_Returns_{type}.png')
    plt.show()

def plot_deviation_histogram(df, type):
    last_row = df.iloc[-1, 1:]

    blue = (0.2, 0.4, 0.8)
    red = (0.8, 0.2, 0.2)

    plt.hist(last_row, bins=12, color=blue, edgecolor='black', alpha=0.7)

    average_value = last_row.mean()

    plt.axvline(x=average_value, color=red, linestyle='--', label='Deviation on average')

    plt.xlabel('Values', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Distribution of Excess Log Returns ({type})', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'Figures/Final/Distribution_Excess_Returns_{type}.png')
    plt.show()


def calculate_bollinger_bands(data, window=20, num_std=2):
    
    rolling_mean = data['Adj Close'].rolling(window=window).mean()
    rolling_std = data['Adj Close'].rolling(window=window).std()

    
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)

    return upper_band, lower_band

def plot_bollinger_bands(df):
    plt.figure(figsize=(12, 6))
    df['Date'] = pd.to_datetime(df['Date'])
    plt.plot(df['Date'], df['Adj Close'], label='Close Price', color='black', linewidth=0.8)
    plt.plot(df['Date'], df['BB upper'], label='Upper Bollinger Band', color='red', linewidth=0.8)
    plt.plot(df['Date'], df['BB lower'], label='Lower Bollinger Band', color='green', linewidth=0.8)
    plt.title('Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_sma_crossover(df):
    plt.figure(figsize=(12, 6))
    df['Date'] = pd.to_datetime(df['Date'])
    plt.plot(df['Date'], df['Adj Close'], label='Close Price', color='black', linewidth=0.8)
    plt.plot(df['Date'], df['SMA 10'], label='SMA 10', color='red', linewidth=0.8)
    plt.plot(df['Date'], df['SMA 50'], label='SMA 50', color='green', linewidth=0.8)
    plt.title('Simple Moving Average Crossover')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

warnings.filterwarnings("ignore")  # Ignore all warnings

""" Used only when fetching the data, Remove # in front of which data you want to download from yfinance"""
# get_data(nifty_50)
# get_data(hel_25)


for symbol in hel_25:

    name = symbol.split(".")[0] # Turn symbol into name
    stock_data = get_from_csv(f"data/{name}.csv") # Get stock data from csv
    calculate_log_returns(stock_data) # Calculate log returns of the stock

    stock_data = sma_crossover(stock_data, sma_lags['short'], sma_lags['long']) # Calculate SMA crossover returns with short and long term MA:s
    
    stock_data = bollinger_bands(stock_data)

    """ Used only when charts of indicators are needed """
    # plot_sma_crossover(stock_data)
    # plot_bollinger_bands(stock_data)

    stock_data = filter_by_date(stock_data, 2011) # Filter data with missing data at the start of the time series
   
    stock_data = calculate_cumulative_log_returns(stock_data) # Turn log returns into cumulative ones
    
    """ Used only when charts of cumulative returns are needed """
    # plot_cumulative_returns(stock_data, name) 

    stock_data.to_csv(f"Results/results_{name}.csv", sep = ";") # Write data into csv
    print(f"results for {name} written to csv")

excess_returns_df_sma = pd.DataFrame()
excess_returns_df_bb = pd.DataFrame()

for symbol in hel_25:

    name = symbol.split(".")[0] # Turn symbol into name
    stock_data = get_from_csv(f"Results/results_{name}.csv") # Get stock data from csv
    excess_returns_df_sma = calculate_excess_returns(excess_returns_df_sma, stock_data, name, "SMA")
    excess_returns_df_bb = calculate_excess_returns(excess_returns_df_bb, stock_data, name, "BB")

index_name = "OMXH 25"
plot_deviation_line_chart(excess_returns_df_sma, f'SMA, {index_name}')
plot_deviation_histogram(excess_returns_df_sma, f'SMA, {index_name}')

plot_deviation_line_chart(excess_returns_df_bb, f'BB, {index_name}')
plot_deviation_histogram(excess_returns_df_bb, f'BB, {index_name}')