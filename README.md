# big_mart_sales_prediction
Predict the sales of stores from structural data containing sale information features (Part of Analytics Vidhya Hackathon.) https://www.analyticsvidhya.com/datahack/leaderboard/practice-problem-big-mart-sales-iii/

## Pre-requisites
- I use the Python 3.12.3 version.
- Run ```pip install -r requirements.txt``` to install dependencies
- Run ```python big_sale.py```

## Methodology
- The data consists of columns ```Item_Identifier	Item_Weight	Item_Fat_Content	Item_Visibility	Item_Type	Item_MRP	Outlet_Identifier	Outlet_Establishment_Year	Outlet_Size	Outlet_Location_Type	Outlet_Type	Item_Outlet_Sales```
- For the final submission, I have used Gradient Boosting Regressor to predict the sales values of test data.
- In final submission, I have found that Item_Weight and Outlet_Size has NaN values. For this I have filled them 0 for earlier and "Missing" for the Outlet_Size NaN values.
- In final submission, I have used Ordinal Encoder to encode the non-numerical columns.

Achieved a ranking of #3235 among 52515 participants as of 23rd March with a public score of 1169.5250917601

## Contact
```nagarjun.gururaj@gmail.com```
