# Martha Stewart


**Quantitative Algorithmic Trading Software for High Frequency Trading (HFT)** aka Automated Trading, Black-Box Trading, or Algo-Trading. Martha Stewart's goal is to execute trades at the best possible prices to reduce transaction costs to capitalize on infinitesimal price discrepancies.

<div>
  
  [![Status](https://img.shields.io/badge/status-work--in--progress-success.svg)]()
  [![GitHub Issues](https://img.shields.io/github/issues/lucylow/Martha_Stewart.svg)](https://github.com/lucylow/salty-wet-man/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/lucylow/Martha_Stewart.svg)](https://github.com/lucylow/Martha_Stewart/pulls)
  [![License](https://img.shields.io/bower/l/bootstrap)]()

</div>

![dashboard](https://github.com/lucylow/Martha_Stewart/blob/master/martha.gif)



--------

## Motivation 
* **Algorithms are important in High Frequency Trading**
   * Computer follows **algorithm of pre-defined set of instructions to place trades**
   * Involves buying and selling securities such as stocks at extremely high speeds where traders may hold the shares they buy for only a fraction of a second before selling them again.
   * Generates profits at a speed and frequency that is impossible for a human trader by **making trading more systematic** by ruling out human emotions
* **Physics and Finance**
  * In finance world, **time scales of nanoseconds/speed of light are important** where one-hundredth of a microsecond is enough time for most HFT trade decisions and executions
  * [Ex. Goldman Sachs is spending $100 million to shave milliseconds off stock trades](https://www.cnbc.com/2019/08/01/goldman-spending-100-million-to-shave-milliseconds-off-stock-trades.html)
  * Joke: The next innovation will be to dig a tunnel to avoid Earth's curvature to speed up trading times 
* **Building your own Algo Trading Software** 
  * Allows flexibility for configurability and customization
  * Ex: Quantopian

---

## Martha Stewart

**Martha Stewart capitalizes on infinitesimal price discrepancies that exists for minuscule periods. Her goal is to execute trades at at the best possible prices to reduce transaction costs.**

By taking advantage of high frequency trading (which accounts for 50–60% of all trading activity), **trade order placements are timed correctly to avoid significant price changes on multiple market conditions**, there will be simultaneous automated checks on multiple market conditions, and capitalize on placing large number of orders at rapid speeds across multiple markets and multiple decision parameters based on preprogrammed instructions

Software **reduces risk of manual errors** when placing trades and trade orders made based on emotional and psychological factors  by human traders. Algo-trading can be **backtested using available historical real-time data to test viable trading stratgies.**



---

## Finance World Algorithmic Trading 

1) **Quantitaitve Finance (ATQs)**
    * Determine prices, manage risk, and identify profitable opportunities by creating financial models and model validation
    * Futures, options, numerical methods, Monte Carlo methods, finite difference methods, binomial model and delta-hedging 
    * Signal processing, game theory, gambling Kelly criterion, market microstructure, econometrics, and time series analysis)
    * **Stocastic Calculus** to model systems that behave randomly
      * Modeling Brownian Motion, diffusion processes of particles subject to random forces in physics
      * Asset prices assumed to follow stochastic differential equations. 
      * Ex. [Black-Scholes Model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_equation) where asset prices follow geometric Brownian motion


2) **Trading / Sales Floor** 
    * Buy side Machine Learning techniques
    * Option Volatility & Pricing Models
    * Volatility Considerations
    * Basic and advanced trading strategies
    * Portfolio risk management techniques (Value at risk, Financial stress test, or Economic capital analysis)


---

## Algorithmic Trading

* **Executing Orders using Automated Trading Instructions**
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

## High Frequency Trading (HFT) 

**Define High Frequency Trading (HFT) and How it Works**
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
    *  Splits large orders into smaller sub-orders that arrive at the same time to all the exchanges through the use of intentional delays
    * Prevents information leakage in the propagation of orders that high-speed traders can take advantage of

  * **Ex.3 Predator-Prey Algorithms** 
    * Ping and sharks
    * Reverse engineer stealth algorithms, **attack and win**
    * Put noise out there to confuse other algorithms

* **Ultra-Fast Black Swan**
  * In an ecosystem where species all look the same - the system crashes
 

---
 
## Criticisms of High Frequency Trading
* Allows institutional players to gain an upper hand in trading because they are able to trade in large blocks through the use of algorithms resulting in dramatic change of finance market microstructure) by paying an additional exchange fee, trading firms get access to see pending orders a split-second before the rest of the market does
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

## Financial Data for Algorithmic Trading

* **Accessing Market Data**
  * Bloomberg data terminal
  * Historical and/or real-time data availability, to backtest their identified strategies
  * Network connectivity and access to trading platforms to place orders.
  * Technical and quantitative analysis tools for trading (like Bollinger bands, charts, etc.)
  * Real-time data feeds required to avoid microsecond delays that may impact profit
  
* **Financial Data**
  * Spreedsheet datasets with rows and columns
  * Finance problems more about manipulating data than math
  * Quality data is far more important than fancy analysis
  
----


## Computational Advantages of Discrete-Time Trading
* Discrete vs Continuous time trading 
  * Continuous-time markets implicitly assume computers and communications are infinitely fast (false)
  * Discrete time respects the limits of computers
* Uniform-price  auctions are fast to compute
* Gives algorithmic traders a discrete block of time between when they receive a message
* Simplifies the market paper trail for regulators and other market observers
  * Adjust for relativity
  * Sequence of events depends on the location of the observe
* Makes it technologically possible to dis-seminate public information symmetrically
* Ran some simple computational simulations of uniform-price auctions,using randomly generated bids and asks, on an ordinary laptop using C++. 
* Ex. 100 millisecond batch interval, the first 10 milliseconds of each batch interval could be allocated to the exchange computers for computing and reporting outcomes from the previous batch interval

-----

 ## How to Execute an Order 
* Execute a market order for stocks 
* Execute a limit order 

---

## Settings for Automated Trading
* Timestamping in the milliseconds


----

## Backtesting the Trading Platform 
* Final component of algorithmic trading, backtest the system before it goes live on real markets
* Test on historical periods of past stock-market performance to see if potentially profitable
* Available historical data for backtesting depending on the complexity of rules implemented in the algorithm


---

## **Udacity "Machine Learning for Trading” from Georgia Tech (CS 7646)**

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
* Flash Boys: A Wall Street Revolt
* https://www.quora.com/Why-is-high-frequency-trading-legal
* Nassim Nichlas Taleb - AntiFragile Things That Gain From Disorder
* Sean Gourley - High frequency trading and the new algorithmic ecosystem
* Udacity course -  "Machine Learning for Trading” https://www.udacity.com/course/machine-learning-for-trading--ud501
* Wes McKinney - Python for Data Analysis
* HFT https://faculty.chicagobooth.edu/eric.budish/research/HFT-FrequentBatchAuctions.pdf
* Python for Finance by Yves Hilpisch
* Books by Sheldon Natenberg or Lawrence G. McMillan
* https://faculty.chicagobooth.edu/eric.budish/research/HFT-FrequentBatchAuctions-Slides.pdf

