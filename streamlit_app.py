import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm
import requests


#to do:

##database?
##study RBA doco and update the tool, the three different methods
## stop monte carlo from rerunning?? but thats the whole point
## put it on online/streamlit cloud

@st.cache_data
# Function to fetch cryptocurrency data
def get_return(tickers,weights,lookback_period):

    ### Set time from to a certain number of years

    endDate = dt.datetime.now()
    startDate = endDate - dt.timedelta(days=lookback_period)

    ### Download the daily adjusted close prices for the tickers
    adj_close_df = pd.DataFrame()

    for ticker in tickers:
        data = yf.download(ticker)
        data_length = len(data)

        # Check if the lookback period is larger than the available data
        if lookback_period > data_length:
            st.error(f"Error: The lookback period of {lookback_period} days exceeds the available data of {ticker} "
                             f"period of {data_length} days. Please adjust the lookback period or start date.")
        else:
            # Proceed with calculations if lookback period is within available data length
            # Example: calculate rolling standard deviation
            data = yf.download(ticker, start=startDate, end=endDate)
            adj_close_df[ticker] = data['Adj Close']



    ### Calculate the daily log returns and drop any NAs
    log_returns = np.log(adj_close_df / adj_close_df.shift(1))
    for ticker in tickers:
        log_returns[ticker] = log_returns[ticker] * weights[tickers.index(ticker)]

    return log_returns.dropna()

def calculate_parametric_var(returns, confidence_level,holding_period, portfolio_value):

    days = holding_period
    rolling_returns = returns.rolling(window=days).sum()
    expected_return = np.sum(rolling_returns.mean())
    weights = np.full(rolling_returns.shape[1],1)
    portfolio_variance = weights.T @ rolling_returns.cov() @ weights
    standard_deviation = np.sqrt(portfolio_variance)
    confidence_interval = confidence_level

    simulations = 1000
    z_scores = np.random.normal(0, 1, simulations)
    scenario_return = portfolio_value * expected_return * days + portfolio_value * standard_deviation * z_scores * \
                      np.sqrt(days)
    # Calculate VaR
    VaR = -np.percentile(scenario_return, 100 * (1 - confidence_level))

    plt.hist(scenario_return, bins=50, density=True)
    plt.xlabel('Scenario Gain/Loss ($)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Portfolio Gain/Loss Over {days} Days')
    plt.axvline(-VaR, color='r', linestyle='dashed', linewidth=2,
                label=f'VaR at {confidence_interval:.0%} confidence level')
    plt.legend()

    plt.show()

    return VaR,plt_x

def calculate_historical_var(returns, confidence_level, holding_period, portfolio_value):

    confidence_interval = confidence_level
    return_window = holding_period
    range_returns = returns.rolling(window=return_window).sum()
    range_returns = range_returns.dropna().sum(axis=1)
    var = -np.percentile(range_returns, 100 - (confidence_interval * 100)) * portfolio_value
    range_returns_dollar = range_returns * portfolio_value

    fig, ax = plt.subplots()
    plt.hist(range_returns_dollar.dropna(), bins=50, density=True)
    ax.set_xlabel('Scenario Gain/Loss ($)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of Portfolio Gain/Loss Over {return_window} Days')
    plt.axvline(-var, color='r', linestyle='dashed', linewidth=2,
                label=f'VaR at {confidence_interval:.0%} confidence level')
    plt.legend()
    plt.show()

    return var,fig

def calculate_var(returns, confidence_level, holding_period, method,portfolio_value):
    if method == 'Historical':
        return calculate_historical_var(returns, confidence_level, holding_period, portfolio_value)
    elif method == 'Parametric':
        return calculate_parametric_var(returns, confidence_level, holding_period,portfolio_value)
    else:
        return None

#Risk free rate data
# URL of the page
url = 'https://www.macrotrends.net/2492/1-year-treasury-rate-yield-chart'

# Add headers to mimic a browser request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'
}

# Fetch the HTML page
if 'response' not in st.session_state:
    response = requests.get(url, headers=headers)
    st.session_state.response = response
else:
    response = st.session_state.response

# Parse the HTML content using pandas
tables = pd.read_html(response.text)

# Print the first table (or choose the appropriate one)
rf =tables[0]
rf = rf['1 Year Treasury - Historical Annual Data']
rf = rf[['Year','Year Close']]
rf['Year Close']= rf['Year Close']/100/365

def calculate_crypto_alpha(portfolio_return, benchmark_return):

    portfolio_df = portfolio_return.to_frame(name="portfolio_return")
    benchmark_return.rename(columns={benchmark_return.columns[0]: "benchmark_return"}, inplace=True)
    port = portfolio_df.merge(benchmark_return,on ="Date")
    merged = port.merge(rf,left_on = port.index.year, right_on="Year")

    merged['r-rf'] = merged["portfolio_return"] - merged["Year Close"]
    merged['rm-rf'] = merged["benchmark_return"] - merged["Year Close"]
    X = merged['rm-rf']
    X = sm.add_constant(X)
    y= merged['r-rf']

    model = sm.OLS(y, X).fit()
    alpha = model.params[0]  # The intercept is alpha
    beta = model.params[1]   # The slope is beta
    p_value_alpha = model.pvalues[0]
    p_value_beta = model.pvalues[1]

    return alpha, beta, p_value_alpha, p_value_beta


# Streamlit interface
st.set_page_config(page_title="Portfolio Risk Tool", page_icon="📊")
st.title("Portfolio Risk Tool")

if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=['Asset', 'Allocation($)'])
if 'beta' not in st.session_state:
    st.session_state.beta = 0
if 'alpha' not in st.session_state:
    st.session_state.alpha = 0

with st.sidebar:
    st.title("📊 Portfolio Risk Tool")
    st.markdown("Created by:[Keeeeeyoti](https://github.com/Keeeeeyoti)")

    confidence_level = st.slider("Select the VaR confidence interval:", 0.90, 0.99, 0.95)
    holding_period = st.number_input("Enter the VaR holding period (days):", min_value=1, value=1)
    lookback_period = st.number_input("Enter the VaR lookback period (days):", min_value=1, value=300)
    method = st.selectbox("Select the VaR calculation method:", ['Historical', 'Parametric'])
    benchmark = st.selectbox("Select the benchmark index:", ['SP500', 'BTC','ASX','DJI'])


tab1, tab2 = st.tabs(["Enter portfolio", "Portfolio analysis"])

with tab1:
    st.write("Input Portfolio")
    asset = st.text_input("Enter the ticker (e.g., BTC-USD):", "BTC-USD")
    allocation = st.number_input("Enter USD value of current allocation", min_value=1, value=1)
    if st.button('Enter allocation'):
        data = yf.Ticker(asset).history(period="1d")
        if data.empty:
            st.write(f"Error: Ticker '{asset}' could not be found on Yahoo Finance.")
        else:
            st.session_state.df.loc[len(st.session_state.df)] = [asset, allocation]
            st.write('Input success')
    if st.button('Clear alloction'):
        st.session_state.df = st.session_state.df.iloc[0:0]

    st.write(st.session_state.df)

with tab2:

    st.write("Portfolio analysis")
    st.write(st.session_state.df)

    if not st.session_state.df.empty:

        # VAR
        tickers = st.session_state.df.iloc[:, 0].tolist()
        allocation = st.session_state.df.iloc[:, 1].tolist()
        weights = allocation / np.sum(allocation)
        returns = get_return(tickers, weights,lookback_period)
        portfolio_value = np.sum(allocation)
        var_value, var_plt = calculate_var(returns, confidence_level, holding_period, method, portfolio_value)


        # alpha,beta,annual return
        if benchmark == 'SP500':
            benchmark_returns = get_return(["^GSPC"], [1], lookback_period)
        if benchmark == 'BTC':
            benchmark_returns = get_return(["BTC-USD"], [1], lookback_period)
        if benchmark == 'ASX':
            benchmark_returns = get_return(["^AXJO"], [1], lookback_period)
        if benchmark == 'DJI':
            benchmark_returns = get_return(["^DJI"], [1], lookback_period)

        portfolio_return = returns.sum(axis=1)
        alpha, beta,p_value_alpha, p_value_beta = calculate_crypto_alpha(portfolio_return, benchmark_returns)


        # charting portfolio vs benchmark
        portfolio_cum_return = (portfolio_return + 1).cumprod() - 1
        benchmark_cum_returns = (benchmark_returns + 1).cumprod() - 1
        chart_data = pd.concat([portfolio_cum_return, benchmark_cum_returns], axis=1)
        chart_data.columns = ['Portfolio return', 'Benchmark return']
        # st.line_chart(data=chart_data)

        fig, ax = plt.subplots()
        chart_data.plot(ax=ax)
        ax.set_ylabel('Return (%)')  # Label for y-axis
        ax.set_title('Portfolio vs Benchmark')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))  # Format y-axis as percentages
        st.pyplot(fig)


        #metrics
        annual_return = portfolio_cum_return[-1]/lookback_period * 365
        st.write(f"The annualised return is {annual_return*100:.3f}%.")
        st.write(f"The Beta is {beta:.4f}. P value is {p_value_beta:.4f}")
        st.write(f"The Yearly Alpha is {alpha*365:.4f} or {alpha*365*100:.2f}% return. P value is {p_value_alpha:.4f}")
        st.session_state.beta = beta
        st.session_state.alpha = alpha

        #VAR chart
        st.write(
            f"The {confidence_level * 100:.0f}% VaR over a holding period of {holding_period} days is: $ "
            f"{var_value:.2f} or {var_value / np.sum(allocation) * 100:.2f}%")
        st.pyplot(var_plt)
