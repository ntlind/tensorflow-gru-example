# Forecasting Dish Subscribers Using Gated Recurrent Units

## Introduction
To date, Professor Peter Fader's "Applied Probability Models in Marketing" class has been the most impactful and rewarding course that I've taken while in graduate school. He is a master at weaving engaging stories using parametric modeling and, after seeing his magic up-close over the previous semester, I found myself wondering how his parametric models would compete with the new-age machine learning algorithms that are gathering steam in the private sector. In this repository, I will build a recurrent neural network (specifically a Gated Recurrent Unit, or GRU) to test its effectiveness against Fader's parametric models as described in his *Valuing Subscription-Based Businesses Using Publicly Disclosed Customer Data*.

## Background
Dish Network is a broadcast satellite service provider based out of Englewood, Colorado that has grown to serve over 14.2M subscribers since its inception in 1981. In this analysis, we will explore the number of pay-TV subscribers that Dish has added over the past decade and compare our findings to Professor Fader's previous analysis, where he reported a 16.0% MAPE when predicting 8 quarters out (Table 3 @ <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2701093>).

## Data Observations
Our initial dataset contained 53,819 observations of pay-TV customer acquisition data that were scraped from Dish Network’s annual investor reports. This data is:
* Discrete
* Asymmetric
* Bound between 0 and positive infinity (theoretically, there was no discernable limit on the number of customers that could be acquired in a single quarter)
* Aggregated by month and quarter
* “Count” data, as opposed to “timing” or “choice” data
* Ranging in date from Q1 1996 to Q4 2016
* Contractual (customers would have to call Dish and pick up their equipment to begin using Dish’s service, so Dish would be notified immediately in the case of a customer acquisition)

## Feature Selection
We begin by taking a step back and thinking from the perspective of a potential Dish customer: given all the digital media alternatives that exist on today’s market, what kind of customer would choose to become a Dish subscriber? I'd hypothesize that Dish customers were swayed by one or more of the following reasons:
* **They are “cost conscious”** – Dish costs $54.99/year, compared to $95.88 for Hulu, $120 for Netflix, and $65.99 for Comcast 
* **They are interested in Dish’s unique technological capabilities** – Dish has multiple innovative products on the market that may have spurred customer demand for their subscription services
* **They were influenced by ad spend or other promotions in their local area**

To test these hypotheses, I pulled data from a variety of first and third-party sources to use as covariates within our model. These features are as follows:
* **Economic**
  * **GDP Growth Rate** - We expect subscriptions to increase in periods of high economic growth. Data taken from the US Bureau of Economic Analysis.
  * **Recession Indicator** - Classified using data from the National Bureau of Economic Research. Customer spending on recreational activities decreased dramatically during the Great Recession (Q4 2007 – Q2 2009).
  * **Consumer Recreational Spending** - We expect subscriptions to increase as consumer recreational spending increases. Consumer recreational spending data scraped from the US Bureau of Economic Analysis.
* **Customer Preference**
  * **Promotional Subsidies** - Promotion data scraped from Dish Network's 10-Ks. We would expect higher promotional spend to correlate positively with increased customer acquisitions.
  * **Sling TV Indicator** - Sling TV’s delivery medium, low price point, and specially-selected offerings will attract an entirely new market of customers who desire access to a select group of premium channels at an affordable price.
* **Other**
  * **Seasonality** - Dish’s annual reports note that most subscriber activations occur in the second half of each calendar year. This covariate involves the inclusion of three separate indicators (one for Q1, one for Q2, and one for Q3). 

## Model Methodology
In applied machine learning, we generally use k-fold cross-validation to repeatedly split our data into randomly-sampled training and test sets and, in doing so, prevent overfitting. The problem with using this approach when dealing with time-series data, however, is that it requires individual observations to be independent of one another. Though several prominent statisticians have still found [uses for cross-validation in this context](https://robjhyndman.com/hyndsight/tscv/), we prefer to use the current gold standard for time-series forecasting: **walk-forward validation** (i.e., rolling window analysis, rolling forecast).

The basic steps in this process are as follows:
* The user defines a *window size* (the number of time periods that fall within our 'window'; called *span* in our model] and begins by selecting a training / test set starting at the beginning of the time series.
* The model makes a prediction for the next time step.
* The prediction is stored or evaluated against the known value.
* The window slides forward to the next period in which we have known values and the process is repeated.

Here's a visual to describe this approach:

![Walk-Forward Validation Example](https://i.stack.imgur.com/padg4.gif)

An alternative approach would be to use **forward chaining**, which has similiar mechanics but an expanding window.

![Forward Chain Example](https://i.stack.imgur.com/fXZ6k.png)

## GRU Layers
To predict acquired customers over time, we will create two separate GRU layers and merge them into one ensembled model:
* **Endogenous Layer** - Uses lagged time-series variables to construct a time-series model of acquired customers.
* **Exogeneous Layer** - Uses the covariates listed above (GDP, etc.) to predict customer acquisitions over time.

The final, ensembled model is weighted (using *yweight*) to empower users with the ability to control variate importance during the modeling process. The model's *predictions* (generated using out-of-sample observations so we can examine the OOS Mean Average Percent Error, or OOS MAPE) and *forecasts* (generated for future time periods in which we have no data) are then visualized and presented with the model's OOS MAPE terms for easy comparison.

## Repository Structure
* **Part 1. Building Dual GRUs** - Build the GRUs cell-by-cell and show that they produce the correct visualizations and error terms.  
* **Part 2. Optimizing Parameters & Forecasting Dish Subscribers** - Test parameter adjustments and create final forecasts for comparison against Professor Fader's 16% MAPE.

## Model Results
Our ensembled GRU performs extremely well, consistently delivering a **7.9% out-of-sample MAPE** when forecasting over a two-year period. When we compare this error term to the Weibull-Gamma's 18% OOS MAPE, we would prefer to use our ensembled model for forecasting purposes. 

If you weren't already impressed, **Dish's recently-released new customer acquisition number for Q1 2017 was 547K, putting our forecast within <0.2% of the real-world observed value.**

![Visual of Final Model](http://i65.tinypic.com/315kjeu.png)

This visual was generated using the code in **"Part 2..."**.

# Caveats
Despite these strong results, there are a few important caveats that should be kept in mind: 
* **This analysis includes additional covariates** - Professor Fader inspired my use of the Seasonality and Recession indicators, but the other features mentioned above were not tested in his study. Though these new covariates *were* tested in the Weibull-Gamma through our second in-class assignment (with a resulting ~20% OOS MdAPE), it is difficult to directly compare Fader's original forecasts to those presented in this analysis. 
* **Parametric models are useful for more than just forecasting** - Our neural network won't answer the same important questions as a well-built parametric model (i.e., how heterogeneous is our customer base?), which means that we may prefer parametric models to RNNs for their real-world applications.
* **Parametric models may perform better than GRUs when working with small datasets** - RNNs shine when given a large number of observations, but parametric models are likely to perform better when you have a limited number of observations.
