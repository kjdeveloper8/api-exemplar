import datetime
import joblib
import numpy as np
np.float_ = np.float64

import pandas as pd
import yfinance as yf
from pathlib import Path
from prophet import Prophet

BASE_DIR = Path(__file__).resolve(strict=True).parent     
TODAY = datetime.date.today()
st = "2024-01-01"
START = datetime.datetime.strptime(st, "%Y-%m-%d")

def train(ticker="MSFT",
          start_date:datetime = START,
          end_date:datetime = TODAY,
          save_csv:bool = False):
    """ Train ticker model.
    
        ticker (str): stock name  
        start_date (datetime): start date  
        end_date (datetime): end date (default: Today) 
        save_csv (bool):  flag to store csv (default: False)   
    """

    data = yf.download(ticker, start_date, end_date.strftime("%Y-%m-%d"))    # date: YYYY-MM-DD
    # print(data.head())

    df_forecast = data.copy()
    df_forecast.reset_index(inplace=True)
    df_forecast["ds"] = df_forecast["Date"]
    df_forecast["y"] = df_forecast["Adj Close"]
    df_forecast = df_forecast[["ds", "y"]]
    # print(df_forecast)

    Path(f"{BASE_DIR}/{ticker}").mkdir(parents=True, exist_ok=True) if True else None
    store_format = f"{ticker}_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}"

    if save_csv:
        df_forecast.to_csv(f"{BASE_DIR}/{ticker}/{store_format}.csv")

    model = Prophet()
    model.fit(df_forecast)

    model_path = Path(f"{BASE_DIR}/{ticker}").joinpath(f"{store_format}.joblib")
    joblib.dump(model, model_path)

def predict(ticker="MSFT", 
            days=7, 
            start_date:datetime = START,
            end_date:datetime = TODAY,
            save_plot=False):
    """ Predicts stock trend.
    
        ticker (str): stock name  
        days (int): period of days for prediction
        start_date (datetime): start date  
        end_date (datetime): end date (default: Today) 
        save_plot (bool):  flag to save plot (default: False)  # NOTE: enabling this causes type error in API calling
    """
        
    store_format = f"{ticker}_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}"

    model_file = Path(f"{BASE_DIR}/{ticker}").joinpath(f"{store_format}.joblib")

    if not model_file.exists():
        return False

    model = joblib.load(model_file)

    future = TODAY + datetime.timedelta(days=days)

    dates = pd.date_range(start=st, end=future.strftime("%m/%d/%Y"),)
    df = pd.DataFrame({"ds": dates})

    forecast = model.predict(df)

    if save_plot:
        model.plot(forecast).savefig(f"{BASE_DIR}/{ticker}/{store_format}_plot.png")
        model.plot_components(forecast).savefig(f"{BASE_DIR}/{ticker}/{store_format}_plot_components.png")

    prediction_list = forecast.tail(days).to_dict("records")
    output = {}
    for data in prediction_list:
        date = data["ds"].strftime("%m/%d/%Y")
        output[date] = data["trend"]
    return output


if __name__ == "__main__":

    # train("GOOG")
    result = predict(ticker='GOOG', days=3)
    print(result)
