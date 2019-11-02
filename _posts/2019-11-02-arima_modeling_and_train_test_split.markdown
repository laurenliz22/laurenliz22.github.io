---
layout: post
title:      "ARIMA Modeling and Train/Test Split"
date:       2019-11-02 15:15:51 +0000
permalink:  arima_modeling_and_train_test_split
---


When looking at time series and considering fitting the ARIMA model to your data, as always it's important to develop train/test splits of your data.  However, when doing this for time series the process is a bit different.  Rather than using a random sample as you may do when fitting a regression model, you'll want to split the data based on your datetime.  

For example, if I had data from 1989 - 2019 I may use the data from 1989-2016 to train my model, the data from 2016-2018 to develop/validate my model and then the data from 2019 to test my model.  

Please find a brief overview of the steps and coding you'll use to do this:

Step 1: Fitting The ARIMA Time Series Model:


* Set up and plot your training data to look at trend and seasonality:

```
df_train = df_all[:'2016-01-01']
df_train.plot(figsize = (15,6))
```

* Determine the best model using a for loop.  Please note - we will look at p=d=q= range(0,2) for this blog.  The greater the range, the longer it will take to process your model, but you may find a better fit.

```
# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)
```
```
# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))
```
```
# Generate all different combinations of seasonal p, q and q triplets
pdqs = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
```
```
#determine what the best model would be using a for loop
ans = []
for comb in pdq:
    for combs in pdqs:
        try:
            mod = sm.tsa.statespace.SARIMAX(df_train,
                                            order=comb,
                                            seasonal_order=combs,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            output = mod.fit()
            ans.append([comb, combs, output.aic])
            print('ARIMA {} x {}12 : AIC Calculated ={}'.format(comb, combs, output.aic))
        except:
            continue
```
```
#print out the best model parameters
ans_df = pd.DataFrame(ans, columns=['pdq', 'pdqs', 'aic'])
ans_df.loc[ans_df['aic'].idxmin()]
```

Based on the output from our for loop we'll see a pdq, pdqs and aic returned.  Our output will suggest the ARIMA parameters pdq and pdqs that yield the lowest aic value and these are the parameters you'll want to use to fit your training data.  

* Fitting your model: Below please find an example with pdq=(1,1,1) and seasonal_order=(1,1,1,12)

```
#Use pdq and pdqs above to fit the data with the ARIMA model
ARIMA_MODEL = sm.tsa.statespace.SARIMAX(df_train,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

output_df_train = ARIMA_MODEL.fit()

print(output_df_train.summary().tables[1])
```

The code above will spit out a table with each coefficient, standard error, z-value, p-value and confidence intervals.  Each weight with a p-value less than 0.05 is significant to the model.

<blockquote class="imgur-embed-pub" lang="en" data-id="a/wGGnZkJ" data-context="false" ><a href="//imgur.com/a/wGGnZkJ"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

Lastly, you'll want to plot the diagnostics with the results to make sure the normality and correlation assumptions for the model hold.  The code below will provide you with 4 graphs (Standard residual and Correlogram to look at your correlation and Histogram plus estimated density and Normal Q-Q to look at normality)

```
#Use plot_diagnostics with results calculated above to make sure 
#our assumptions of normality and correlation hold
output_df_train.plot_diagnostics(figsize=(15, 18))
plt.show()
```

<blockquote class="imgur-embed-pub" lang="en" data-id="a/P1dEJto" data-context="false" ><a href="//imgur.com/a/P1dEJto"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

Step 2: Developing/Validating The Model

* Set up and plot your development/validation data to look at trend and seasonality:

```
df_dev = df_all['2016-01-01' : '2018-01-01']
```

* Repeat steps above to fit your model and review the model's output as well as the diagonostic plots to review if the normality and correlation assumptions hold.

Step 3: Testing The Model

* Repeat step 2 for the final year of your data

```
df_test = df_all[ '2019-01-01' :]
```

Once you are comfortable with your model you can now look at forecasting to determine your models predictive power!
