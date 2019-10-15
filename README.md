# Martha Stewart
**Algorithmic Trading** also known as Automated Trading, Black-Box Trading, or Algo-Trading

---

## Motivation 
* Algorithms are important in trading
   * Computer program follows **algorithm of pre-defined set of instructions to place trades**
   * Generates profits at a speed and frequency that is impossible for a human trader
* Relates to **physic background** where in the finance world, **time scales of nanoseconds/speed of light are important**: [Ex. Goldman Sachs is spending $100 million to shave milliseconds off stock trades](https://www.cnbc.com/2019/08/01/goldman-spending-100-million-to-shave-milliseconds-off-stock-trades.html)



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

## Numerical Methods 

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

## Algorithmic Trading and  High Frequency Trading (HFT)

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


---

## References 
* Nassim Nichlas Taleb - AntiFragile Things That Gain From Disorder
* Sean Gourley - High frequency trading and the new algorithmic ecosystem
* Udacity course -  "Machine Learning for Trading” https://www.udacity.com/course/machine-learning-for-trading--ud501
* Wes McKinney - Python for Data Analysis
* Books by Sheldon Natenberg or Lawrence G. McMillan

