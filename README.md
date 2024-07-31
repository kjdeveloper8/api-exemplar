# API Exemplar

Model deployment and API integration example


#### 📉 Stock Prediction

1. Create and activate env

```py
# brew install pipenv
cd <project-dir>
pipenv shell
```

2. Install requirements

```py
pipenv install -r requirements.txt
```

3. Training and prediction

```py
import joblib
from prophet import Prophet

# training
model = Prophet()
model.fit(data) # downloaded finance data

joblib.dump(model, "stock.joblib") # save model

# prediction

model = joblib.load("stock.joblib")  # load model

forecast = model.predict(df)  # data to predict
```

See `train()` and `predict()` module in `stock_model.py` for full implementation


4. API integration

```shell
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

or

```py
python main.py
```