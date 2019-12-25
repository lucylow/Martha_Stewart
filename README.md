# Martha Stewart


**Algorithmic Trading Software for High Frequency Trading (HFT)** (also known ask Automated Trading, Black-Box Trading, or Algo-Trading)

<div>
  
  [![Status](https://img.shields.io/badge/status-work--in--progress-success.svg)]()
  [![GitHub Issues](https://img.shields.io/github/issues/lucylow/Martha_Stewart.svg)](https://github.com/lucylow/salty-wet-man/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/lucylow/Martha_Stewart.svg)](https://github.com/lucylow/Martha_Stewart/pulls)
  [![License](https://img.shields.io/bower/l/bootstrap)]()

</div>

--------

## Motivation 
* Algorithms are important in trading
   * Computer follows **algorithm of pre-defined set of instructions to place trades**
   * Generates profits at a speed and frequency that is impossible for a human trader
   * **Makes trading more systematic** by ruling out human emotions
* Physics background
  * In finance world, **time scales of nanoseconds/speed of light are important**: 
  * Example, in high frequency trading, one-hundredth of a microsecond is enough time for most HFT trade decisions and executions and three milliseconds is an eternity to  high frequency trading firms
  * [Ex. Goldman Sachs is spending $100 million to shave milliseconds off stock trades](https://www.cnbc.com/2019/08/01/goldman-spending-100-million-to-shave-milliseconds-off-stock-trades.html)
* Building my own algo trading software 
  * Allows flexibility for configurability and customization
  * Ex: Quantopian

---

## Benefits of Algorithmic Trading

* **Money**
  * Trades are executed at the best possible prices
  * Reduced transaction costs
  * HFT is a way for traders to capitalize on infinitesimal price discrepancies that might exist only for a minuscule period.
* **Accuracy of trade orders**
  * Trade order placement is instant and accurate (high chance of execution at the desired levels)
  * Trades are timed correctly and instantly to avoid significant price changes
  * Simultaneous automated checks on multiple market conditions
  * Algo-trading can be backtested using available historical and real-time data to see if it is a viable trading strategy
  * Capitalize on placing a large number of orders at rapid speeds across multiple markets and multiple decision parameters based on preprogrammed instructions
* **Reduced possibility of mistakes** 
  * Reduced risk of manual errors when placing trades
  * Human traders based on emotional and psychological factors


---

## Finance World Algorithmic Trading 

1) **Quantitaitve Finance (ATQs)**
    * Determine prices, manage risk, and identify profitable opportunities by creating financial models and model validation
    * Futures, options, numerical methods, Monte Carlo methods, finite difference methods, binomial model and delta-hedging 
    * Signal processing, game theory, gambling Kelly criterion, market microstructure, econometrics, and time series analysis)
    * **Stocastic Calculus** to model systems that behave randomly
      * Modeling Brownian Motion, diffusion processes of particles subject to random forces in physics
      * Asset prices assumed to follow stochastic differential equations. Ex. [Black-Scholes Model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_equation) where asset prices follow geometric Brownian motion <3


2) **Trading / Sales Floor** 
    * Buy side uses machine learning
    * Option Volatility & Pricing Models
    * Volatility Considerations
    * Basic and advanced trading strategies
    * Portfolio risk management techniques (Value at risk, Financial stress test, or Economic capital analysis)


---

## Algorithmic Trading

* **Executing orders using automated trading instructions**: 
   * Parameters so traders do not need to manually watch a stock
      * Time, Price, and Volume
   * Parameters so traders can execute large orders
      * Send small slices of the order out to the market over time


* **Popular Algorithm Examples**
   * Percentage of Volume
   * Pegged
   * VWAP
   * TWAP
   * Implementation shortfall
   * Target close
   
* **Target Audience**
   * Investment banks
   * Pension funds
   * Mutual funds
   * Hedge funds (Renaissance Technologies and Two Sigma)
---

## Numerical Methods used in Finance

* **Finite difference method**  
  * Solve partial differential equations
* **Monte Carlo method** 
  * Solve partial differential equations as well 
  * Monte Carlo simulation common in risk management
* **Ordinary least squares** 
  * Estimate parameters in statistical regression analysis
* **Spline interpolation** 
  * Interpolate values from spot and forward interest rates curves, and volatility smiles
* **Bisection, Newton, and Secant methods** 
  * Find the roots, maxima and minima of functions 
  * Ex. Internal rate of return


---

## High Frequency Trading (HFT) 

**Define High Frequency Trading (HFT) and How it works**
* Goal of HFT is to benefit from bid-ask spreads
  * If they sense an opportunity, HFT algorithms then try to capitalize on large pending orders by adjusting prices to fill them and make profits.
  * It adds liquidity to the markets and eliminates small bid-ask spreads
  * High turnover and High order-to-trade ratios
* HFT algorithms typically involve two-sided order placements 
   * Buy-low
   * Sell-high
* Two types of HFT
  1) Market makers to set price
  2) **Statistical arbitrage to guess price (focus on this one)**

---

## HFT Examples

  * **Ex.1 Pair Trading Algorithms**
    * Time series of one stock
    * Convert to binary 
    * Look for correlations and diverges (delta)
    * If diverges, **makes trade and wins** 

  * **Ex.2 Moving Large Volume Stocks** 
    * Break up large stock ($200million)
    * Iceberging 

  * **Ex.3 Predator-Prey Algorithms** 
    * Ping and sharks
    * Reverse engineer stealth algorithms, **attack and win**
    * Put noise out there to confuse other algorithms

* **Ultra-Fast Black Swan**
  * When you have species that all look the same in the ecosystem, the system crashes
 

---
 
## Criticisms of HFT
* Allows institutional players to gain an upper hand in trading because they are able to trade in large blocks through the use of algorithms resulting in dramatic change of finance market microstructure)  
* By paying an additional exchange fee, trading firms get access to see pending orders a split-second before the rest of the market does
* Potentially destabilizing for financial markets,making the market more vulnerable to extreme events such asthe Flash Crash
* Liquidity produced by this type of trading is momentary as it disappears within seconds, making it impossible for traders to take advantage of it


---

## Technical Tools

**Python Data Science Toolkit**

* **Python Programming Language**
  * Beautiful
  * Statistical computing language
  * C++, Java, and Python are common among trader community
  
* **Pandas Library**
  * Data analysis library 
  * Created by Wes McKinney working at hedgefund AQR Capital Management
  
* Other Fun Tools: Numpy, Scipy, and Matplotlib
* Lame alternatives: Excel or R

---

## Financial data for Algorithmic Trading

* Accessing market data
  * Bloomberg data terminal
  * Historical and/or real-time data availability, to backtest their identified strategies
  * Network connectivity and access to trading platforms to place orders.
  * Technical and quantitative analysis tools for trading (like Bollinger bands, charts, etc.)
  * Real-time data feeds required to avoid microsecond delays that may impact profit

* Financial Data
  * Spreedsheet datasets with rows and columns
  * Finance problems more about manipulating data than math
  * Quality data is far more important than fancy analysis
  
----


## Computational Advantages of Discrete-Time Trading
* Discrete vs Continue time trading 
* Uniform-price  auctions  are  fast  to  compute
* Gives algorithmic traders a discrete block of timebetween when they receive a message
* Simplifies the market paper trail for reg-ulators  and  other  market  observers
  * Adjust for relativity
  * Sequence of events depends on the location of the observe
* Makes it technologically possible to dis-seminate  public  information  symmetrically

Formally, the processing time of the uniform-price auction isOðnlognÞ,wherenis the number of orders. Sorting bids and asks to compute the demandand supply curve isOðnlognÞ(Cormen et al. 2009), and then walking down thedemand curve and up the supply curve to compute the market clearing price isO(n).We also ran some simple computational simulations of uniform-price auctions,using randomly generated bids and asks, on an ordinary laptop using C++. Wefound that a uniform-price auction with 250,000 orders—the rate of messages persecond during the flash crash according to a Nanex analysis (2011)—cleared inabout 10 milliseconds in this simple computational environment

For instance, with a 100 millisecond batch interval, the first 10 millisecondsof each batch interval could be allocated to the exchange computers for computingand reporting outcomes from the previous batch interval.



----

## Backtesting the Trading Platform 
* Final component of algorithmic trading, backtest the system before it goes live on real markets
* Test on historical periods of past stock-market performance to see if potentially profitable
* Available historical data for backtesting depending on the complexity of rules implemented in the algorithm


---

## **Udacity course "Machine Learning for Trading” from Georgia Tech (CS 7646)**

* Implementing **ML based trading strateies** from info gathering to market orders
* Focus on how to apply probalistic approaches to trading decisions
* Statistical approaches - **linear regression, KNN, and regression tree** 
* **Data structures used for algorithmc trading** 
* Build stock trading software system that uses current daily data
* Build software to access **live equity data**, assess it, and **make trading decisions**

* Optomizing portfolios
* Computing at Hedgefunds. So you want to become a Hedgefund manager?


---

## References 
* Nassim Nichlas Taleb - AntiFragile Things That Gain From Disorder
* Sean Gourley - High frequency trading and the new algorithmic ecosystem
* Udacity course -  "Machine Learning for Trading” https://www.udacity.com/course/machine-learning-for-trading--ud501
* Wes McKinney - Python for Data Analysis
* HFT https://faculty.chicagobooth.edu/eric.budish/research/HFT-FrequentBatchAuctions.pdf
* What Hedge Funds Really Do by Romero and Balch
* Python for Finance by Yves Hilpisch
* Books by Sheldon Natenberg or Lawrence G. McMillan

