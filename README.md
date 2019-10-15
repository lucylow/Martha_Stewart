# Martha Stewart

<div>
  
  [![Status](https://img.shields.io/badge/status-work--in--progress-success.svg)]()
  [![GitHub Issues](https://img.shields.io/github/issues/lucylow/Martha_Stewart.svg)](https://github.com/lucylow/salty-wet-man/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/lucylow/Martha_Stewart.svg)](https://github.com/lucylow/Martha_Stewart/pulls)
  [![License](https://img.shields.io/bower/l/bootstrap)]()

</div>

**Algorithmic Trading Software for High Frequency Trading (HFT)** (also known ask Automated Trading, Black-Box Trading, or Algo-Trading)
---

## Motivation 
* Algorithms are important in trading
   * Computer program follows **algorithm of pre-defined set of instructions to place trades**
   * Generates profits at a speed and frequency that is impossible for a human trader
   * **Makes trading more systematic** by ruling out human emotions
* Relates to **physic background** where in the finance world, **time scales of nanoseconds/speed of light are important**:
  * In high frequency trading, one-hundredth of a microsecond is enough time for most HFT trade decisions and executions
  * [Ex. Goldman Sachs is spending $100 million to shave milliseconds off stock trades](https://www.cnbc.com/2019/08/01/goldman-spending-100-million-to-shave-milliseconds-off-stock-trades.html)
* Build your own trading software allows full flexibility for Configurability and Customization
  * a good free source to explore algorithmic trading is Quantopian
  * Online platform for testing and developing algorithmic trading. Individuals can try and customize any existing algorithm or write a completely new one. The platform also offers built-in algorithmic trading software to be tested against market data.

---

## Benefits of Algorithmic Trading

* Trades are executed at the best possible prices
* Trade order placement is instant and accurate (there is a high chance of execution at the desired levels)
* Trades are timed correctly and instantly to avoid significant price changes
* Reduced transaction costs
* Simultaneous automated checks on multiple market conditions
* Reduced risk of manual errors when placing trades
* Algo-trading can be backtested using available historical and real-time data to see if it is a viable trading strategy
* Reduced possibility of mistakes by human traders based on emotional and psychological factors

---

## Finance World Algorithmic Trading 

1) **Quantitaitve Finance (ATQs)**
    * Determine prices, manage risk, and identify profitable opportunities by creating financial models and model validation
    * Futures, options, numerical methods, Monte Carlo methods, finite difference methods, binomial model and delta-hedging 
    * Signal processing, game theory, gambling Kelly criterion, market microstructure, econometrics, and time series analysis)
    * **Stocastic Calculus** to model systems that behave randomly
      * Modeling Brownian Motion, diffusion processes of particles subject to random forces in physics
      * Asset prices assumed to follow stochastic differential equations. Ex. [Black-Scholes Model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_equation) where assett prices follow geometric Brownian motion


2) **Trading / Sales Floor** 
    * Buy side uses machine learning
    * Option Volatility & Pricing Models
    * Volatility Considerations
    * Basic and advanced trading strategies
    * Portfolio risk management techniques (Value at risk, Financial stress test, or Economic capital analysis)


---

## Algorithmic Trading

* **Executing orders using automated trading instructions**: 
   * Parameters so traders do not need to manually watch a stock and can execute large orders
      * Time, Price, and Volume
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

## Basic Numerical Methods 

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

## Algorithmic Trading and High Frequency Trading (HFT)

**Define High Frequency Trading (HFT)**
   * High turnover and High order-to-trade ratios
   * Resulted in dramatic change of finance market microstructure (the way liquidity is provided)

**Two Types of HFT:**
  1) Market makers, set price
  2) **Statistical arbitrage, guess price** 
   * Three examples below


---

## HFT Examples
Strategies to use computers to make decisions electronically, before human traders are capable of processing the information. Capitalize on placing a large number of orders at rapid speeds across multiple markets and multiple decision parameters based on preprogrammed instructions. 

Algorithms essentially work as middlemen between buyers and sellers, with HFT and Ultra HFT being a way for traders to capitalize on infinitesimal price discrepancies that might exist only for a minuscule period.

HFT algorithms typically involve two-sided order placements (buy-low and sell-high) in an attempt to benefit from bid-ask spreads. HFT algorithms also try to “sense” any pending large-size orders by sending multiple small-sized orders and analyzing the patterns and time taken in trade execution. If they sense an opportunity, HFT algorithms then try to capitalize on large pending orders by adjusting prices to fill them and make profits.


Also, Ultra HFT is a further specialized stream of HFT. By paying an additional exchange fee, trading firms get access to see pending orders a split-second before the rest of the market does.

HFT is complex algorithmic trading in which large numbers of orders are executed within seconds.
It adds liquidity to the markets and eliminates small bid-ask spreads.
There are two primary criticisms of HFT. The first one is that it allows institutional players to gain an upper hand in trading because they are able to trade in large blocks through the use of algorithms. The second criticism against HFT is that the liquidity produced by this type of trading is momentary. It disappears within seconds, making it impossible for traders to take advantage of it.



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
    * Put noise out there to confuse the other algorithms

* **Ultra-Fast Black Swan**
  * When you have species that all look the same in the ecosystem, the system crashes

---

## Technical Tools

**Python Data Science Toolkit**

* **Python Programming Language**
  * Beautiful
  * Statistical computing language
  
* **Pandas Library**
  * Data analysis library 
  * Created by Wes McKinney working at hedgefund AQR Capital Management
  
* **Financial Data**
  * Spreedsheet datasets with rows and columns
  * Finance problems more about manipulating data than math
  * Quality data is far more important than fancy analysis
  
* Other Tools: Numpy, Scipy, and Matplotlib
* Alternatives: Excel or R

---

## Technical Requirements for Algorithmic Trading

* Acessing market data
  * Bloomberg data terminal
  * Technical and quantitative analysis tools for trading (like Bollinger bands, charts, etc.)
 
* Programming languages
  * Perl, C++, Java, Python are the common ones among the trader community
* Historical and/or real-time data availability, to backtest their identified strategies
* Automated access to brokerage/trading accounts usually through Direct Market Access
  * 

High-speed computers, which need regular and costly hardware upgrades;

Real-time data feeds, which are required to avoid even a microsecond's delay that may impact profits; and


Implementing the algorithm using a computer program is the final component of algorithmic trading, accompanied by backtesting (trying out the algorithm on historical periods of past stock-market performance to see if using it would have been profitable). The challenge is to transform the identified strategy into an integrated computerized process that has access to a trading account for placing orders. The following are the requirements for algorithmic trading:

 Computer-programming knowledge to program the required trading strategy, hired programmers, or pre-made trading software.
 Network connectivity and access to trading platforms to place orders.
 Access to market data feeds that will be monitored by the algorithm for opportunities to place orders.
 The ability and infrastructure to backtest the system once it is built before it goes live on real markets.
 Available historical data for backtesting depending on the complexity of rules implemented in the algorithm.



---

## **Udacity course "Machine Learning for Trading” from Georgia Tech (CS 7646)**

* Implementing **ML based trading strateies** from info gathering to market orders
* Focus on how to apply probalistic approaches to trading decisions
* Statistical approaches - **linear regression, KNN, and regression tree** 
* **Data structures used for algorithmc trading** 
* Build stock trading software system that uses current daily data
* Build software to access **live equity data**, assess it, and **make trading decisions**

* Optomizing portfios
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

