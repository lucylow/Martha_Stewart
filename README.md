# Martha Stewart
Artifical Intelligence Algorithmic Trading in Finance

---

## Motivation 
* Algorithms are important in trading
* Relates to **physic background** where in the finance world, **time scales of nanoseconds/speed of light are important**.  
[Ex. Goldman Sachs is spending $100 million to shave milliseconds off stock trades](https://www.cnbc.com/2019/08/01/goldman-spending-100-million-to-shave-milliseconds-off-stock-trades.html)

---

## Finance World Algorithmic Trading 

1) **Quantitaitve Finance (ATQs)**
    * Determine prices, manage risk, and identify profitable opportunities by creating financial models and model validation
    * Futures, options, numerical methods, Monte Carlo methods, finite difference methods, binomial model and delta-hedging 
    * Signal processing, game theory, gambling Kelly criterion, market microstructure, econometrics, and time series analysis)
    * **Stocastic calculus**
      * Model systems that behave randomly
      * Modeling Brownian Motion, diffusion processes of particles subject to random forces in physics
      * Asset prices assumed to follow stochastic differential equations.Ex. [Black-Scholes](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_equation) model, prices follow geometric Brownian motion


2) **Trading / Sales FLoor** 
    * **Buy side uses Machine Learning**
    * Option Volatility & Pricing Models
    * Volatility Considerations
    * Basic and advanced trading strategies
    * Portfolio risk management techniques (Value at risk, Financial stress test, or Economic capital analysis)


---

## Algorithmic Trading

* **Executing orders using automated trading instructions**: 
   * Parameters so traders do not need to manually watch a stock and traders need to execute large orders in certain markets
      * Time, price, and volume
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
   * Hedge funds
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
Strategies to use computers to make decisions electronically, before human traders are capable of processing the information

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

## Technical

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

