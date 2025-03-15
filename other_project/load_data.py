
import pandas as pd

def transformer(tickers : pd.DataFrame, controverses : pd.DataFrame):

    tickers = tickers[tickers['Ticker'].str.contains("Not Collected")==False]

    tickers = tickers.dropna(axis = 1)
    tickers = tickers.dropna(axis = 0)

    tickers = tickers.reset_index(drop=True)
    tickers.index = range(1, len(tickers) + 1)

    ticker_extensions = {
        'USA': '',      # États-Unis : aucune extension
        'CAN': '.TO',   # Canada : Toronto Stock Exchange
        'SWE': '.ST',   # Suède : Stockholm
        'BRA': '.SA',   # Brésil : B3 (Sao Paulo)
        'SUI': '.SW',   # Suisse : Swiss Exchange
        'PHI': '.PS',   # Philippines : Philippine Stock Exchange
        'THA': '.BK',   # Thaïlande : Stock Exchange of Thailand (Bangkok)
        'TWN': '.TW',   # Taïwan
        'UK':  '.L',    # Royaume-Uni : London Stock Exchange
        'JPN': '.T',    # Japon : Tokyo Stock Exchange
        'NLD': '.AS',   # Pays-Bas : Euronext Amsterdam
        'FRA': '.PA',   # France : Euronext Paris
        'MEX': '.MX',   # Mexique : Bolsa Mexicana de Valores
        'FIN': '.HE',   # Finlande : Nasdaq Helsinki
        'AUT': '.VI',   # Autriche : Wiener Börse
        'AUS': '.AX'    # Australie : Australian Securities Exchange
    }

    # Fonction qui ajoute l'extension si nécessaire
    def add_extension(row):
        ticker = row['Ticker']
        # Si le ticker contient déjà un point, on considère qu'il est déjà complété
        if '.' in ticker:
            return ticker
        iso_code = row['ISOCountryCode3']
        # Récupération de l'extension à partir du dictionnaire (retourne '' si non trouvé)
        extension = ticker_extensions.get(iso_code, '')
        return ticker + extension

    # Création d'une nouvelle colonne 'Ticker_full' avec le ticker complet
    tickers['Ticker_full'] = tickers.apply(add_extension, axis=1)

    controverses['Tickers'] = tickers['Ticker']
    controverses = controverses[[controverses.columns[-1]] + list(controverses.columns[:-1])]

    # Filtrer les lignes où le ticker n'est pas None
    valid_tickers = controverses[controverses['Tickers'].notna()]['Tickers'].tolist()

    import yfinance as yf
    import pandas as pd

    # Example DataFrame 'controverses' with columns ['Tickers', ...]
    # We assume it's already loaded/created.

    # Initialize columns for returns, sector, and additional Yahoo Finance fields
    controverses['1_year_return'] = None
    controverses['6_month_return'] = None
    controverses['3_month_return'] = None
    controverses['Sector'] = None

    # Additional columns based on the Yahoo Finance screenshot
    controverses['previous_close'] = None
    controverses['open_price'] = None
    controverses['day_low'] = None
    controverses['day_high'] = None
    controverses['fifty_two_week_low'] = None
    controverses['fifty_two_week_high'] = None
    controverses['beta'] = None
    controverses['forward_dividend_rate'] = None
    controverses['forward_dividend_yield'] = None
    controverses['bid'] = None
    controverses['ask'] = None
    controverses['volume'] = None
    controverses['average_volume'] = None
    controverses['market_cap'] = None
    controverses['trailing_pe'] = None
    controverses['eps_ttm'] = None
    controverses['ex_dividend_date'] = None
    controverses['earnings_date'] = None
    controverses['target_mean_price'] = None

    # Function to calculate returns
    def calculate_returns(historical_data, period_days):
        try:
            # Get start and end prices for the specified period
            start_price = historical_data.iloc[-period_days]['Close']  # Close price at start
            end_price = historical_data.iloc[-1]['Close']              # Close price at the end
            return (end_price / start_price - 1) * 100                  # Calculate return percentage
        except Exception as e:
            print(f"Error calculating return: {e}")
            return None

    # Loop through each ticker to fetch data, calculate returns, and fetch additional info
    for i in range(100):
        ticker = controverses.loc[i, 'Tickers']
        print(f"Processing {ticker}...")

        try:
            # Fetch ticker info and historical data (last 2 years here)
            stock = yf.Ticker(ticker)
            historical_data = stock.history(period="2y")

            # If no historical data is available, skip this ticker
            if historical_data.empty:
                print(f"No historical data available for {ticker}.")
                continue

            # Calculate returns
            controverses.loc[i, '1_year_return'] = calculate_returns(historical_data, period_days=252)  # ~1 year
            controverses.loc[i, '6_month_return'] = calculate_returns(historical_data, period_days=126) # ~6 months
            controverses.loc[i, '3_month_return'] = calculate_returns(historical_data, period_days=63)  # ~3 months

            # Fetch main info dictionary
            info = stock.info

            # Sector
            controverses.loc[i, 'Sector'] = info.get('sector', 'Unknown')

            # Additional fields
            controverses.loc[i, 'previous_close'] = info.get('previousClose')
            controverses.loc[i, 'open_price'] = info.get('open')
            controverses.loc[i, 'day_low'] = info.get('dayLow')
            controverses.loc[i, 'day_high'] = info.get('dayHigh')
            controverses.loc[i, 'fifty_two_week_low'] = info.get('fiftyTwoWeekLow')
            controverses.loc[i, 'fifty_two_week_high'] = info.get('fiftyTwoWeekHigh')
            controverses.loc[i, 'beta'] = info.get('beta')
            controverses.loc[i, 'forward_dividend_rate'] = info.get('dividendRate')
            controverses.loc[i, 'forward_dividend_yield'] = info.get('dividendYield')
            controverses.loc[i, 'bid'] = info.get('bid')
            controverses.loc[i, 'ask'] = info.get('ask')
            controverses.loc[i, 'volume'] = info.get('volume')
            controverses.loc[i, 'average_volume'] = info.get('averageVolume')
            controverses.loc[i, 'market_cap'] = info.get('marketCap')
            controverses.loc[i, 'trailing_pe'] = info.get('trailingPE')
            controverses.loc[i, 'eps_ttm'] = info.get('trailingEps')
            controverses.loc[i, 'ex_dividend_date'] = info.get('exDividendDate')
            controverses.loc[i, 'earnings_date'] = info.get('earningsDate')
            controverses.loc[i, 'target_mean_price'] = info.get('targetMeanPrice')

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    # Drop rows where 1_year_return is None (if desired)
    controverses = controverses.dropna(subset=['1_year_return'])

    # Display the updated DataFrame
    print(controverses[[
        'Tickers', 'Sector',
        '1_year_return', '6_month_return', '3_month_return',
        'previous_close', 'open_price', 'day_low', 'day_high',
        'fifty_two_week_low', 'fifty_two_week_high', 'beta',
        'forward_dividend_rate', 'forward_dividend_yield',
        'bid', 'ask', 'volume', 'average_volume', 'market_cap',
        'trailing_pe', 'eps_ttm', 'ex_dividend_date', 'earnings_date',
        'target_mean_price'
    ]])

    # Suppression des lignes où '1_year_return' est None/NaN
    controverses = controverses.dropna(subset=['1_year_return'])

    return tickers, controverses