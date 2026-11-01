# Quantitative Algorithmic Trading Arbitrage Bot


**Quantitative Algorithmic Trading Software for High Frequency Trading (HFT)** aka Automated Trading, Black-Box Trading, or Algo-Trading. Martha Stewart's goal is to execute trades at the best prices by capitalizing on infinitesimal price discrepancies by forecasting stock market closing prices.

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
* Markov Models applied to financial time series data to mimic dynamics of price trends in the stock markets using four latent states to forecast stock market closing price https://github.com/luisdamiano/gsoc17-hhmm
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



---

## Martha Plus research and paper-trading layer

This repository now includes an additive `martha_plus/` package for validated OHLCV data, strategy research, portfolio accounting, risk controls, simulated paper execution, SQLite/audit persistence, backtesting, and an optional operations API. It does not connect to a live broker and does not support live trading. Review [`docs/martha_plus/INTEGRATION.md`](docs/martha_plus/INTEGRATION.md), [`docs/martha_plus/RISK_AND_OPERATIONS.md`](docs/martha_plus/RISK_AND_OPERATIONS.md), and [`docs/martha_plus/BACKTESTING.md`](docs/martha_plus/BACKTESTING.md) before use.

### Research package quick start

The add-on requires Python 3.10 or newer and uses only the Python standard library for its core. From the repository root, run:

```bash
python -m martha_plus config-check --json
python -m martha_plus demo --bars 500 --seed 7 --strategy momentum
python -m martha_plus backtest --csv examples/sample_bars.csv --symbol DEMO --strategy momentum
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

The bundled CSV is seeded synthetic data, not real market data. The legacy application container remains unchanged and currently specifies Python 3.6; it cannot run this add-on without a separately reviewed runtime upgrade. The research package can be run with a Python 3.10+ interpreter from the project root. Existing `config.py`, `requirements.txt`, Docker/Gunicorn, migrations, and Flask routes were intentionally preserved.


### Advanced research and operations extension (v0.2.0)

The `martha_plus.advanced` namespace adds point-in-time data contracts and causal feature pipelines; local feature storage and provenance; portfolio optimization, constraints, risk budgets, attribution, and rebalancing diagnostics; purged walk-forward and statistical validation; corporate-action/session/universe tools; market-data failover and quality monitoring; liquidity/capacity estimates; paper order-slicing schedules and execution-quality analysis; event replay, manual scheduling, alerts, incidents, readiness probes, experiment tracking, snapshots, and escaped report exports. It does not submit orders or enable live trading.

This layer contributes **5,165 Python implementation lines** (104 page-equivalents at the 50-lines-per-page convention). It was updated in v0.2.1 with app compatibility fixes; the current combined v0.3.0 totals, including the new enterprise layer, are shown below. See [`docs/martha_plus/ADVANCED_EXTENSION.md`](docs/martha_plus/ADVANCED_EXTENSION.md) and [`docs/martha_plus/advanced/README.md`](docs/martha_plus/advanced/README.md) for module boundaries, examples, and limitations.


### Legacy app setup fixes

The legacy test helpers now import from `testing.test_base`, avoiding a name collision with the new research suite in `tests/`. The app factory uses a thread-safe `before_request` database initializer compatible with current Flask versions. For the legacy Python 3.6 SQLite container, `requirements-sqlite.txt` supplies pinned dependencies; the Dockerfile no longer copies secrets into the image. Before using Compose, copy `.env.example` to `.env` and replace `SECRET_KEY` with a unique random value. The `martha_plus` add-on itself still requires Python 3.10+ and is not enabled in the legacy Python 3.6 image.


### Enterprise controls and portfolio operations (v0.3.0)

The `martha_plus.enterprise` namespace adds 5,121 lines of framework-neutral research and paper-operations utilities: authorization and signed-request verification, order lifecycle/idempotency/outbox primitives, event-time streaming and point-in-time joins, governance, cash/tax-lot/FX analysis, risk-limit evaluation, scenario and benchmark diagnostics, portfolio replay, reconciliation, surveillance, and safe exports. These controls are opt-in and do not connect to a live broker or submit live orders. Their in-memory components are not distributed guarantees; see [`docs/martha_plus/ENTERPRISE_EXTENSION.md`](docs/martha_plus/ENTERPRISE_EXTENSION.md) and [`docs/martha_plus/enterprise/README.md`](docs/martha_plus/enterprise/README.md) for the boundaries.

The full package and tests now contain **16,538 Python lines** (331 page-equivalents at 50 lines/page), including 5,121 new enterprise implementation lines. Install `requirements-dev.txt` and run `sh scripts/martha_plus/check.sh` to exercise unittest and pytest-style regression tests together.
