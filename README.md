# Bayesian deep neural network for trading with dynamic profit taking

This project implements a Bayesian deep neural network for trading. Gal and Ghahramani (2016) show that “a neural network with arbitrary depth and non-linearities, with dropout applied before every weight layer, is mathematically equivalent to an approximation to the probabilistic deep Gaussian process.” Simply training a neural network (NN) with dropout and then, contrary to common practice, using dropout at test time, performing a large number of stochastic forward passes approximates Bayesian inference. Using this method, uncertainty estimates can be obtained simply by computing the sample standard deviation of the predictions. These estimates, in turn, can be used to ignore trading signals that are highly uncertain, thereby increasing the win rate and reducing transaction costs. The model might learn a mean reversion strategy. If there is a large down move due to some event, the model might forecast a large return even if such a move has never occurred during training and is actually justified. In this case we don't want to buy into the market crash because we have no idea what will happen next since we never observed such a thing. Another example of the usefulness of the Bayesian uncertainty estimate is that the model may ignore predictions based on outliers due to data errors that creep through our detection process. 



Inputs are ohlc and in fact any time series that is desired. As an example, the user can input price series statistics, fundamentals, intermarket statistics, order book information or trading signals from other models or based on the trader's own discretion, etc.  
The model predicts the specified period's open to high (low) return. The return that is greater in absolute terms is used as the target in training. If the high (low) in the period is strictly higher (lower) than the target price of a long (short) trade, the profit is taken. The inequality is strict to err on the side of conservatism with respect to fill assumptions. In contrast to https://github.com/jpwoeltjen/BayesianDNN the profit taking can be dynamic by setting the profit_taking_threshold to np.nan. In this case, the target return is the predicted return. If a specific return threshold is defined by the user, the target return is this defined value and NOT dynamic. 
 
Transaction costs are configurable in the equity_curve() function. They are currently set to $2 ($1 commissions + $1 slippage) per $100000 trading volume. 

The model is fit on the training set. A grid search over the user defined hyper parameters is performed including the profit taking threshold, position taking threshold and Bayesian uncertainty threshold. The models are saved to the disk. The best model and hyperparameters are selected based on the validation set performance. This model is used to predict on the test set providing an unbiased estimate of expected future performance. The testing on the test set is necessary because the act of choosing the best performing model biases the expected future performance based on the validation set upwards. The highest care was taken to prevent any information leakage into the test set. 

As a demonstration the model is fit on the EUR.USD currency pair. Every ten minutes a prediction is made whether to enter, exit, or hold a long or short position. The maximum holding period is specified to be one hour. 1000 stochastic forward passes are performed to obtain uncertainty estimates and mean predictions. 

As an example output consider:
```

Win rate 100.00_sigma: 63.915857605178 %
Percentage of periods betting up 100.00_sigma : 0.852870488998 %
Percentage of periods betting down: 100.00_sigma  0.339932481086 %
Percentage of periods staying out of the market: 100.00_sigma  98.8071970299 %

There were 2474.0 total trades for 100.00_sigma.
The annualised_sharpe for 100.00_sigma. is: 1.12.
The CAGR for 100.00_sigma. is: 2.92 percent.
The annualised_sharpe for 100.00_sigma. after commissions is: 0.80.
The CAGR for 100.00_sigma. is: 2.07 percent. after commissions
Average winning trade: 0.0012098237068309985
Average losing trade: -0.0017546262082783674
Average trade: 0.00014012737823975482
Average long trade: 8.566419267203525e-05
Average short trade: 0.0002774490854404153
Average time in long trade: 3.112509834775767
Average time in short trade: 3.012784090909091
```
Refer to the log.txt files for more information. 

To improve the model one should consider adding potentially predictive data and trying many hyperparameter combinations to find out which ones work best. 
