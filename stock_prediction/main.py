import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from stock_model import predict

app = FastAPI()

@app.get("/msg")
def greet():
    return {"msg": "Welcome"}


# pydantic model
class StockIn(BaseModel):
    ticker: str

class StockOut(StockIn):
    forecast: dict


@app.post("/predict", response_model=StockOut, status_code=200)
def get_prediction(payload: StockIn):
    ticker = payload.ticker
    prediction_list = predict(ticker)

    if not prediction_list:
        raise HTTPException(status_code=400, detail="model not found!")

    response_object = {"ticker": ticker, "forecast":prediction_list}
    return response_object

if __name__ == "__main__":
    uvicorn.run(
        __name__ + ':app',
        reload = True,
        # workers = 1,
        host = 'localhost',
        port = 8080,
    )