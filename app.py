
# IMPORT LIBRARIES


import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

from plotly.subplots import make_subplots

import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from streamlit_searchbox import st_searchbox

import warnings
import datetime

warnings.filterwarnings("ignore")


# STREAMLIT CONFIG


st.set_page_config(

    page_title="AI Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"

)


# AUTO REFRESH


st.markdown(

    """
    <script>
    setTimeout(function(){
       window.location.reload();
    }, 15000);
    </script>
    """,

    unsafe_allow_html=True

)


# SESSION STATE


if "last_refresh" not in st.session_state:

    st.session_state.last_refresh = datetime.datetime.now()


# CACHE DOWNLOAD


@st.cache_data(ttl=2)
def cached_download(symbol, period, interval):

    try:

        data = yf.download(

            symbol,
            period=period,
            interval=interval,

            auto_adjust=False,
            progress=False,
            threads=True,
            prepost=True

        )

        if data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):

            data.columns = data.columns.get_level_values(0)

        data.index = pd.to_datetime(
            data.index
        ).tz_localize(None)

        data = data[~data.index.duplicated(keep='first')]

        return data

    except Exception as e:

        print(e)
        return pd.DataFrame()


# YAHOO FINANCE SEARCH


@st.cache_data(ttl=3600)
def search_yahoo_stocks(query):

    if not query or len(query) < 2:
        return []

    try:

        url = "https://query2.finance.yahoo.com/v1/finance/search"

        params = {

            "q": query,
            "quotesCount": 20,
            "newsCount": 0

        }

        headers = {

            "User-Agent": "Mozilla/5.0"

        }

        response = requests.get(

            url,
            params=params,
            headers=headers,
            timeout=5

        )

        data = response.json()

        results = []

        for item in data.get("quotes", []):

            symbol = item.get("symbol", "")

            if symbol.endswith(".NS"):

                shortname = item.get(
                    "shortname",
                    symbol
                )

                results.append(
                    f"{shortname} ({symbol})"
                )

        return results

    except Exception as e:

        print(e)
        return []


# THEME SWITCH


theme_mode = st.sidebar.toggle(
    "🌙 Dark Trading Mode",
    value=True
)


# THEME COLORS


if theme_mode:

    bg_primary = "#0f1720"
    bg_secondary = "#1a2332"
    card_bg = "rgba(28,35,48,0.92)"
    text_color = "#f8fafc"
    sidebar_bg = "rgba(15,23,32,0.98)"
    chart_bg = "rgba(15,23,42,0.92)"

else:

    bg_primary = "#2b2621"
    bg_secondary = "#3a342d"
    card_bg = "rgba(58,52,45,0.92)"
    text_color = "#f8f4ee"
    sidebar_bg = "rgba(43,38,33,0.98)"
    chart_bg = "rgba(58,52,45,0.92)"


# CUSTOM CSS


st.markdown(f"""

<style>

.stApp {{

    background:
    radial-gradient(
        circle at top left,
        {bg_secondary} 0%,
        {bg_primary} 50%,
        #1e1b18 100%
    );

    color: {text_color};
    font-family: 'Inter', sans-serif;

}}

section[data-testid="stSidebar"] {{

    background:
    linear-gradient(
        180deg,
        {sidebar_bg},
        rgba(20,20,20,0.98)
    );

}}

div[data-testid="metric-container"] {{

    background:
    linear-gradient(
        145deg,
        {card_bg},
        rgba(255,255,255,0.03)
    );

    border: 1px solid rgba(255,255,255,0.06);

    padding: 22px;
    border-radius: 24px;

}}

.js-plotly-plot {{

    border-radius: 20px;
    overflow: hidden;

}}

</style>

""", unsafe_allow_html=True)


# MODEL CACHE


@st.cache_resource
def get_model():

    return RandomForestRegressor(

        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1

    )


# STOCK ANALYZER


class StockAnalyzer:

    def __init__(self):

        self.scaler = StandardScaler()
        self.model = get_model()

    
    # FETCH DATA
    

    def fetch_stock_data(self, symbol, period="1y"):

        try:

            if period == "1mo":

                interval = "5m"
                fetch_period = "1mo"

            elif period == "3mo":

                interval = "1h"
                fetch_period = "3mo"

            elif period == "6mo":

                interval = "1d"
                fetch_period = "6mo"

            elif period == "1y":

                interval = "1d"
                fetch_period = "1y"

            elif period == "2y":

                interval = "1d"
                fetch_period = "2y"

            else:

                interval = "1wk"
                fetch_period = "5y"

            data = cached_download(

                symbol,
                fetch_period,
                interval

            )

            if data.empty:
                return pd.DataFrame(), {}

            data = data.dropna()

            return data, {}

        except Exception as e:

            print(e)
            return pd.DataFrame(), {}

    
    # TECHNICAL INDICATORS
    

    def calculate_technical_indicators(self, data):

        df = data.copy()

        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()

        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()

        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()

        delta = df['Close'].diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss

        df['RSI'] = 100 - (100 / (1 + rs))

        return df

    
    # MACHINE LEARNING MODEL
    

    def train_prediction_model(self, data):

        try:

            df = data.copy()

            df['Returns'] = df['Close'].pct_change()
            df['MA_5'] = df['Close'].rolling(5).mean()
            df['MA_10'] = df['Close'].rolling(10).mean()
            df['Volatility'] = df['Returns'].rolling(5).std()

            for lag in [1, 2, 3, 5]:

                df[f'Lag_{lag}'] = df['Close'].shift(lag)

            df = df.dropna()

            if len(df) < 50:
                return None

            features = [

                'Returns',
                'MA_5',
                'MA_10',
                'Volatility',
                'Lag_1',
                'Lag_2',
                'Lag_3',
                'Lag_5'

            ]

            X = df[features]
            y = df['Close'].shift(-1)

            X = X[:-1]
            y = y[:-1]

            if len(X) == 0:
                return None

            X_train, X_test, y_train, y_test = train_test_split(

                X,
                y,

                test_size=0.2,
                shuffle=False

            )

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            self.model.fit(

                X_train_scaled,
                y_train

            )

            predictions = self.model.predict(
                X_test_scaled
            )

            mae = mean_absolute_error(
                y_test,
                predictions
            )

            accuracy = self.model.score(
                X_test_scaled,
                y_test
            )

            next_features = X.iloc[-1:]

            return {

                "accuracy": accuracy,
                "mae": mae,
                "next_features": next_features

            }

        except Exception as e:

            print(e)
            return None

    
    # PREDICT
    

    def predict_next_price(self, model_info):

        features_scaled = self.scaler.transform(
            model_info['next_features']
        )

        prediction = self.model.predict(
            features_scaled
        )[0]

        return float(prediction)

    
    # AI ANALYSIS
    

    def generate_market_analysis(self, data):

        latest = data.iloc[-1]

        insights = []

        if float(latest['RSI']) > 70:

            insights.append(
                "⚠️ RSI indicates overbought conditions."
            )

        elif float(latest['RSI']) < 30:

            insights.append(
                "💡 RSI indicates oversold conditions."
            )

        else:

            insights.append(
                "📊 RSI indicates neutral momentum."
            )

        if float(latest['MACD']) > float(latest['MACD_signal']):

            insights.append(
                "🚀 MACD indicates bullish momentum."
            )

        else:

            insights.append(
                "🔻 MACD indicates bearish momentum."
            )

        if float(latest['Close']) > float(latest['SMA_20']):

            insights.append(
                "📈 Price trading above short-term trend."
            )

        else:

            insights.append(
                "📉 Price trading below short-term trend."
            )

        return insights


# COMPARISON SYMBOLS


def get_comparison_symbol(compare_index):

    mapping = {

        "NIFTY 50": "^NSEI",
        "BANK NIFTY": "^NSEBANK",
        "SENSEX": "^BSESN"

    }

    return mapping.get(compare_index)


# CREATE CHART


def create_chart(
    data,
    chart_height,
    chart_bg,
    show_sma20,
    show_sma50,
    show_volume,
    show_macd,
    show_rsi,
    line_width,
    show_grid
):

    fig = make_subplots(

        rows=4,
        cols=1,

        shared_xaxes=True,

        vertical_spacing=0.06,

        row_heights=[0.50, 0.16, 0.17, 0.17],

        subplot_titles=(
            "Price Action",
            "Volume",
            "MACD",
            "RSI"
        )

    )

    fig.add_trace(

        go.Candlestick(

            x=data.index,

            open=data['Open'].astype(float),
            high=data['High'].astype(float),
            low=data['Low'].astype(float),
            close=data['Close'].astype(float),

            increasing_line_color='#22c55e',
            decreasing_line_color='#ef4444',

            increasing_fillcolor='#22c55e',
            decreasing_fillcolor='#ef4444',

            name='Price'

        ),

        row=1,
        col=1

    )

    if show_sma20:

        fig.add_trace(

            go.Scatter(

                x=data.index,
                y=data['SMA_20'],

                line=dict(
                    color='#38bdf8',
                    width=line_width
                ),

                name='SMA 20'

            ),

            row=1,
            col=1

        )

    if show_sma50:

        fig.add_trace(

            go.Scatter(

                x=data.index,
                y=data['SMA_50'],

                line=dict(
                    color='#818cf8',
                    width=line_width
                ),

                name='SMA 50'

            ),

            row=1,
            col=1

        )

    if show_volume:

        fig.add_trace(

            go.Bar(

                x=data.index,
                y=data['Volume'].astype(float),

                marker_color='#3b82f6',

                name='Volume'

            ),

            row=2,
            col=1

        )

    if show_macd:

        fig.add_trace(

            go.Scatter(

                x=data.index,
                y=data['MACD'],

                line=dict(
                    color='#22d3ee',
                    width=line_width
                ),

                name='MACD'

            ),

            row=3,
            col=1

        )

        fig.add_trace(

            go.Scatter(

                x=data.index,
                y=data['MACD_signal'],

                line=dict(
                    color='#f59e0b',
                    width=line_width
                ),

                name='Signal'

            ),

            row=3,
            col=1

        )

    if show_rsi:

        fig.add_trace(

            go.Scatter(

                x=data.index,
                y=data['RSI'],

                line=dict(
                    color='#f472b6',
                    width=line_width
                ),

                name='RSI'

            ),

            row=4,
            col=1

        )

    fig.update_layout(

        template="plotly_dark",

        height=chart_height,

        paper_bgcolor=chart_bg,
        plot_bgcolor=chart_bg,

        hovermode="x unified",

        dragmode="pan",

        xaxis_rangeslider_visible=False,

        font=dict(color=text_color),

        uirevision="constant",

        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )

    )

    if len(data) > 1:

        intraday_mode = (
            (data.index[1] - data.index[0]).total_seconds()
            < 86400
        )

        if intraday_mode:

            fig.update_xaxes(

                rangebreaks=[

                    dict(bounds=["sat", "mon"]),

                    dict(
                        bounds=[15.5, 9.15],
                        pattern="hour"
                    )

                ]

            )

        else:

            fig.update_xaxes(

                rangebreaks=[

                    dict(bounds=["sat", "mon"])

                ]

            )

    fig.update_yaxes(
        side="right",
        showgrid=show_grid
    )

    fig.update_xaxes(
        showgrid=show_grid
    )

    return fig


# MAIN APP


def main():

    st.title("📈 AI Stock Analysis Dashboard")

    st.markdown(
        "*Advanced AI + ML Technical Trading Dashboard*"
    )

    st.sidebar.header("📊 Dashboard Controls")

    
    # LIVE NSE SEARCH
    

    selected_stock = st_searchbox(

        search_function=search_yahoo_stocks,

        placeholder="🔍 Search Any NSE Stock",

        label="Search Stocks",

        key="stock_search"

    )

    if not selected_stock:

        symbol = "RELIANCE.NS"

    else:

        symbol = selected_stock.split("(")[-1].replace(")", "")

    
    # CONTROLS
    

    period = st.sidebar.selectbox(

        "📅 Time Frame",

        [
            "1mo",
            "3mo",
            "6mo",
            "1y",
            "2y",
            "5y"
        ]

    )

    chart_height = st.sidebar.slider(
        "📈 Chart Height",
        700,
        1400,
        1000
    )

    st.sidebar.markdown("---")

    st.sidebar.subheader("⚙️ Advanced Dashboard Controls")

    show_sma20 = st.sidebar.checkbox(
        "📘 Show SMA 20",
        value=True
    )

    show_sma50 = st.sidebar.checkbox(
        "📗 Show SMA 50",
        value=True
    )

    show_volume = st.sidebar.checkbox(
        "📊 Show Volume",
        value=True
    )

    show_macd = st.sidebar.checkbox(
        "📉 Show MACD",
        value=True
    )

    show_rsi = st.sidebar.checkbox(
        "📈 Show RSI",
        value=True
    )

    show_prediction = st.sidebar.checkbox(
        "🔮 Show ML Prediction",
        value=True
    )

    show_ai_analysis = st.sidebar.checkbox(
        "🧠 Show AI Analysis",
        value=True
    )

    enable_comparison = st.sidebar.checkbox(
        "📊 Enable Comparison",
        value=True
    )

    show_grid = st.sidebar.checkbox(
        "🔲 Show Chart Grid",
        value=True
    )

    line_width = st.sidebar.slider(
        "📏 Indicator Width",
        1,
        5,
        2
    )

    comparison_index = st.sidebar.selectbox(

        "📈 Compare With",

        [
            "None",
            "NIFTY 50",
            "BANK NIFTY",
            "SENSEX"
        ]

    )

    
    # MARKET STATUS
    

    now = datetime.datetime.now()

    market_open = (

        now.weekday() < 5 and

        (
            (now.hour > 9 or (now.hour == 9 and now.minute >= 15))
            and
            (now.hour < 15 or (now.hour == 15 and now.minute <= 30))
        )

    )

    if market_open:

        st.success("🟢 NSE Market Open")

    else:

        st.warning("🔴 NSE Market Closed")

    st.caption(
        f"Last Updated: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    
    # FETCH DATA
    

    analyzer = StockAnalyzer()

    with st.spinner("📡 Fetching Market Data..."):

        data, info = analyzer.fetch_stock_data(
            symbol,
            period
        )

    if data.empty:

        st.error(
            f"❌ Invalid or unavailable stock symbol: {symbol}"
        )

        return

    
    # TECHNICALS
    

    data = analyzer.calculate_technical_indicators(data)

    
    # METRICS
    

    latest_price = float(data['Close'].iloc[-1])

    prev_price = float(data['Close'].iloc[-2])

    change_pct = (
        (latest_price - prev_price)
        / prev_price
    ) * 100

    historical_data = cached_download(
        symbol,
        "1y",
        "1d"
    )

    week_52_high = float(historical_data['High'].max())
    week_52_low = float(historical_data['Low'].min())

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "💰 Current Price",
            f"₹{latest_price:.2f}",
            f"{change_pct:+.2f}%"
        )

    with col2:

        st.metric(
            "📊 Volume",
            f"{float(data['Volume'].iloc[-1]):,.0f}"
        )

    with col3:

        st.metric(
            "📈 52W High",
            f"₹{week_52_high:.2f}"
        )

    with col4:

        st.metric(
            "📉 52W Low",
            f"₹{week_52_low:.2f}"
        )

    st.markdown("---")

    
    # MAIN CHART
    

    st.subheader("📈 Advanced Technical Analysis")

    chart = create_chart(

        data,
        chart_height,
        chart_bg,
        show_sma20,
        show_sma50,
        show_volume,
        show_macd,
        show_rsi,
        line_width,
        show_grid

    )

    st.plotly_chart(

        chart,

        use_container_width=True,

        config={

            "scrollZoom": True,
            "displaylogo": False,
            "responsive": True

        }

    )

    
    # ML PREDICTION
    

    if show_prediction:

        st.markdown("---")

        st.subheader("🔮 Machine Learning Price Prediction")

        model_info = analyzer.train_prediction_model(
            data
        )

        if model_info is not None:

            prediction = analyzer.predict_next_price(
                model_info
            )

            prediction_change = (
                (
                    prediction - latest_price
                )
                / latest_price
            ) * 100

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "🎯 Predicted Price",
                    f"₹{prediction:.2f}",
                    f"{prediction_change:+.2f}%"
                )

            with c2:

                st.metric(
                    "🤖 Model Accuracy",
                    f"{model_info['accuracy']:.1%}"
                )

            with c3:

                st.metric(
                    "📉 Prediction Error",
                    f"{model_info['mae']:.2f}"
                )

            if prediction_change > 2:

                st.success(
                    "🟢 Bullish Signal Detected"
                )

                st.markdown("""
### 📈 What This Means

The ML model predicts upward momentum.

- strong bullish trend
- positive buying pressure
- momentum continuation possible

### ⚠️ Important

Predictions are probabilistic and not guaranteed.
""")

            elif prediction_change < -2:

                st.error(
                    "🔴 Bearish Signal Detected"
                )

                st.markdown("""
### 📉 What This Means

The ML model predicts downside movement.

- bearish momentum
- selling pressure
- weak technical structure

### ⚠️ Important

ML predictions may fail during volatile conditions.
""")

            else:

                st.info(
                    "🟡 Sideways Market Expected"
                )

                st.markdown("""
### ↔️ What This Means

The model predicts low movement.

- consolidation phase
- low volatility
- range-bound trading possible

### ⚠️ Important

Breakouts can still happen unexpectedly.
""")

        else:

            st.warning(
                "⚠️ Not enough historical data available for ML prediction."
            )

    
    # AI ANALYSIS
    

    if show_ai_analysis:

        st.markdown("---")

        st.subheader("🧠 AI-Powered Market Analysis")

        insights = analyzer.generate_market_analysis(
            data
        )

        for insight in insights:

            st.info(insight)

        latest = data.iloc[-1]

        st.markdown("## 📚 Indicator Explanation")

        # RSI

        rsi = float(latest['RSI'])

        if rsi > 70:

            st.markdown("""
### RSI Analysis

RSI above 70 suggests overbought conditions.

- strong buying momentum
- possible correction risk
""")

        elif rsi < 30:

            st.markdown("""
### RSI Analysis

RSI below 30 suggests oversold conditions.

- heavy selling pressure
- possible recovery bounce
""")

        else:

            st.markdown("""
### RSI Analysis

RSI is neutral.

- balanced momentum
- no extreme conditions
""")

        # MACD

        macd = float(latest['MACD'])
        signal = float(latest['MACD_signal'])

        if macd > signal:

            st.markdown("""
### MACD Analysis

MACD above signal line indicates bullish momentum.

- upward trend strength
- positive acceleration
""")

        else:

            st.markdown("""
### MACD Analysis

MACD below signal line indicates bearish momentum.

- weaker trend
- downside pressure
""")

        # TREND

        close = float(latest['Close'])
        sma20 = float(latest['SMA_20'])

        if close > sma20:

            st.markdown("""
### Trend Analysis

Price trading above SMA20.

- bullish short-term structure
- buyers dominating
""")

        else:

            st.markdown("""
### Trend Analysis

Price trading below SMA20.

- bearish short-term structure
- sellers dominating
""")


# RUN APP


if __name__ == "__main__":

    main()