from __future__ import annotations

import math
import unittest
from datetime import date

from martha_plus.technical_finance.core import (
    FinanceInputError, DayCount, bisect, cholesky, normal_ppf, solve_linear, year_fraction,
)
from martha_plus.technical_finance.statistics import moments, zscore, autocorrelation
from martha_plus.technical_finance.returns import (
    simple_returns, log_returns, compound, drawdown_series, maximum_drawdown,
    drawdown_episodes, return_decomposition,
)
from martha_plus.technical_finance.performance import (
    sharpe_ratio, sortino_ratio, omega_ratio, performance_report,
)
from martha_plus.technical_finance.regression import ols, newey_west_covariance
from martha_plus.technical_finance.factors import (
    composite_score, rank_information_coefficient, factor_spread_returns,
    neutralize_by_groups,
)
from martha_plus.technical_finance.portfolio import (
    covariance_matrix, portfolio_variance, minimum_variance_weights,
    risk_parity_weights, component_risk_contribution,
)
from martha_plus.technical_finance.optimization import (
    black_litterman_posterior, mean_variance_weights, efficient_frontier,
)
from martha_plus.technical_finance.risk import (
    historical_var, historical_expected_shortfall, gaussian_var,
    gaussian_expected_shortfall, monte_carlo_var,
)
from martha_plus.technical_finance.options.black_scholes import (
    price as bs_price, greeks, implied_volatility, put_call_parity_residual,
)
from martha_plus.technical_finance.options.binomial import crr_price
from martha_plus.technical_finance.options.surface import ImpliedVolSurface, VolPoint
from martha_plus.technical_finance.options.portfolio import aggregate_greeks, risk_reversal
from martha_plus.technical_finance.fixed_income.bonds import (
    bond_price, yield_to_maturity, bond_risk, duration_convexity_pnl,
)
from martha_plus.technical_finance.fixed_income.curves import (
    CurvePoint, DiscountCurve, bootstrap_deposit_curve, par_swap_rate,
)
from martha_plus.technical_finance.fixed_income.rates_models import (
    vasicek_bond_price, cir_bond_price, cir_feller_condition,
)
from martha_plus.technical_finance.fixed_income.term_structure import (
    nelson_siegel_yield, fit_nelson_siegel,
)
from martha_plus.technical_finance.derivatives import (
    fx_forward, equity_forward, interest_rate_swap_value,
)
from martha_plus.technical_finance.credit import (
    hazard_from_spread, default_probability, expected_credit_loss,
    fair_cds_spread, transition_matrix_check,
)
from martha_plus.technical_finance.econometrics import (
    estimate_ou, half_life_from_ar1, kalman_hedge_ratio, adf_regression,
)
from martha_plus.technical_finance.valuation import dcf_valuation, dividend_discount_model
from martha_plus.technical_finance.attribution import brinson_fachler, active_return_reconciliation
from martha_plus.technical_finance.capital import weighted_average_cost_of_capital, kelly_fraction
from martha_plus.technical_finance.scenarios import factor_scenario, reverse_stress_required_move
from martha_plus.technical_finance.liquidity import (
    amihud_illiquidity, linear_transaction_cost, days_to_liquidate,
)
from martha_plus.technical_finance.volatility import (
    realized_volatility, bipower_variation, ewma_variance, garch11_variance,
    garch11_unconditional_variance, parkinson_volatility,
)
from martha_plus.technical_finance.stochastic import (
    simulate_gbm, simulate_ou, monte_carlo_european_option,
)
from martha_plus.technical_finance.ledger import PortfolioLedger, time_weighted_return
from martha_plus.technical_finance.fundamentals import dupont_analysis, altman_z_score, piotroski_f_score
from martha_plus.technical_finance.hierarchical import hierarchical_risk_parity, diversification_ratio
from martha_plus.technical_finance.tailrisk import gpd_moments, gpd_quantile, gpd_survival, peaks_over_threshold_var, tail_dependence_count


class CoreMathTests(unittest.TestCase):
    def test_normal_quantile_and_solver(self):
        self.assertAlmostEqual(normal_ppf(0.975), 1.95996398, places=6)
        self.assertAlmostEqual(bisect(lambda x: x*x, 4, 0, 3), 2, places=7)

    def test_linear_solve_and_cholesky(self):
        answer = solve_linear(((4, 1), (1, 3)), (1, 2))
        self.assertAlmostEqual(answer[0], 1/11)
        lower = cholesky(((4, 1), (1, 3)))
        self.assertAlmostEqual(lower[0][0], 2)

    def test_day_count(self):
        self.assertAlmostEqual(year_fraction(date(2024, 1, 1), date(2025, 1, 1), DayCount.ACT_ACT), 1.0)

    def test_distribution(self):
        stats = moments((-2, -1, 0, 1, 2, 3))
        self.assertGreater(stats["volatility"], 0)
        self.assertAlmostEqual(sum(zscore((1,2,3))), 0, places=10)
        self.assertAlmostEqual(autocorrelation((1,2,3,4,5), 0), 1.0)


class ReturnPerformanceTests(unittest.TestCase):
    def test_returns_and_drawdowns(self):
        r = simple_returns((100, 110, 99, 108.9))
        self.assertAlmostEqual(compound(r), 0.089)
        self.assertEqual(len(log_returns((100, 110, 99))), 2)
        self.assertLess(maximum_drawdown((0.1, -0.2, 0.05)), 0)
        self.assertTrue(drawdown_episodes((0.1, -0.2, 0.3)))
        self.assertEqual(len(drawdown_series((0.1, -0.2))), 2)

    def test_return_decomposition(self):
        result = return_decomposition(0.1, 0.02, 0.03)
        self.assertAlmostEqual(result["total"], 1.1*1.02*1.03-1)

    def test_performance_metrics(self):
        sample = (0.01, -0.005, 0.02, -0.01, 0.015, 0.003)
        self.assertTrue(math.isfinite(sharpe_ratio(sample)))
        self.assertTrue(math.isfinite(sortino_ratio(sample)))
        self.assertGreater(omega_ratio(sample), 0)
        report = performance_report(sample)
        self.assertIn("max_drawdown", report)


class RegressionFactorTests(unittest.TestCase):
    def test_ols(self):
        x = (1,2,3,4,5,6)
        y = tuple(2+3*v for v in x)
        fit = ols(y, (x,))
        self.assertAlmostEqual(fit.coefficients[0], 2)
        self.assertAlmostEqual(fit.coefficients[1], 3)
        self.assertAlmostEqual(fit.r_squared, 1)
        hac = newey_west_covariance((x,), fit.residuals)
        self.assertEqual(len(hac), 2)

    def test_factor_tools(self):
        scores = composite_score({"value":{"A":1,"B":2,"C":3},
                                  "quality":{"A":3,"B":2,"C":1}},
                                 {"value":0.5,"quality":0.5})
        self.assertAlmostEqual(scores["A"], scores["C"])
        self.assertAlmostEqual(rank_information_coefficient((1,2,3),(1,2,3)), 1)
        spread = factor_spread_returns({"A":3,"B":2,"C":1}, {"A":.1,"B":0,"C":-.1}, 3)
        self.assertGreater(spread["long_short"], 0)
        residual = neutralize_by_groups({"A":1,"B":3,"C":10,"D":14},
                                       {"A":"x","B":"x","C":"y","D":"y"})
        self.assertAlmostEqual(residual["A"], -1)


class PortfolioRiskTests(unittest.TestCase):
    def setUp(self):
        self.returns = ((0.01,0.02,-0.01,0.00,0.015,0.012),
                        (0.005,0.01,-0.005,0.003,0.008,0.006))
        self.cov = covariance_matrix(self.returns)

    def test_portfolio_weights(self):
        weights = minimum_variance_weights(self.cov)
        self.assertAlmostEqual(sum(weights), 1.0, places=6)
        risk = risk_parity_weights(self.cov)
        self.assertAlmostEqual(sum(risk), 1.0, places=6)
        self.assertGreater(portfolio_variance(weights, self.cov), 0)
        self.assertEqual(len(component_risk_contribution(weights, self.cov)), 2)

    def test_optimizer_and_black_litterman(self):
        mu = (0.08, 0.05)
        weights = mean_variance_weights(mu, self.cov, risk_aversion=3)
        self.assertAlmostEqual(sum(weights), 1, places=6)
        frontier = efficient_frontier(mu, self.cov, points=3)
        self.assertEqual(len(frontier), 3)
        posterior = black_litterman_posterior(
            self.cov, (0.6,0.4), 2.5, ((1,-1),), (0.02,), ((0.001,),))
        self.assertEqual(len(posterior["posterior_returns"]), 2)

    def test_var_and_es(self):
        sample = (-0.05,-0.03,-0.02,-0.01,0.0,0.01,0.02,0.03,0.04,0.05)
        var = historical_var(sample, .9)
        es = historical_expected_shortfall(sample, .9)
        self.assertGreaterEqual(es, var)
        self.assertGreater(gaussian_var(sample, .95), 0)
        self.assertGreater(gaussian_expected_shortfall(sample, .95), 0)
        sim = monte_carlo_var((0.5,0.5), self.cov, (0.001,0.0005), .95, simulations=2000)
        self.assertGreaterEqual(sim["var"], 0)


class DerivativesTests(unittest.TestCase):
    def test_black_scholes_and_iv(self):
        call = bs_price(100, 100, 1, .05, .2)
        put = bs_price(100, 100, 1, .05, .2, option_type="put")
        self.assertAlmostEqual(call-put, 100-100*math.exp(-.05), places=7)
        iv = implied_volatility(call, 100, 100, 1, .05)
        self.assertAlmostEqual(iv, .2, places=6)
        self.assertGreater(greeks(100,100,1,.05,.2).gamma, 0)
        self.assertAlmostEqual(put_call_parity_residual(call, put, 100,100,1,.05), 0, places=7)

    def test_binomial_and_surface(self):
        tree = crr_price(100, 100, 1, .05, .2, steps=300)
        analytical = bs_price(100,100,1,.05,.2)
        self.assertAlmostEqual(tree, analytical, delta=.15)
        surface = ImpliedVolSurface([VolPoint(1,90,.22), VolPoint(1,110,.20),
                                     VolPoint(2,90,.24), VolPoint(2,110,.22)])
        self.assertAlmostEqual(surface.volatility(1.5,100), .22)
        self.assertTrue(surface.check_calendar_monotonicity().passed)

    def test_option_book_greeks(self):
        pos = [{"spot":100,"strike":100,"time":1,"rate":.03,"volatility":.2,
                "quantity":2,"multiplier":100,"option_type":"call"}]
        self.assertGreater(aggregate_greeks(pos)["delta"], 0)
        self.assertAlmostEqual(risk_reversal(.2,.25), -.05)

    def test_forwards(self):
        fx = fx_forward(1.3,.04,.02,1)
        self.assertGreater(fx.forward, 1.3)
        self.assertGreater(equity_forward(100,.05,.02,1).forward, 100)


class FixedIncomeCreditTests(unittest.TestCase):
    def test_bond_price_yield_duration(self):
        px = bond_price(100, .05, .04, 5, 2)
        ytm = yield_to_maturity(px, 100, .05, 5, 2)
        self.assertAlmostEqual(ytm, .04, places=7)
        risk = bond_risk(100,.05,.04,5,2)
        self.assertGreater(risk.modified_duration, 0)
        self.assertGreater(risk.convexity, 0)
        self.assertLess(duration_convexity_pnl(100,risk.modified_duration,risk.convexity,.01), 0)

    def test_curve_and_swap(self):
        curve = bootstrap_deposit_curve(((.5,.04),(1,.042),(2,.043),(3,.044)))
        self.assertGreater(curve.discount_factor(1), curve.discount_factor(2))
        self.assertGreater(par_swap_rate(curve, 2, 2), 0)

    def test_short_rate_models_and_ns(self):
        self.assertGreater(vasicek_bond_price(.03,.5,.04,.01,5), 0)
        self.assertGreater(cir_bond_price(.03,.5,.04,.1,5), 0)
        self.assertEqual(cir_feller_condition(.5,.04,.3)["satisfied"], 0.0)
        rate = nelson_siegel_yield(5,.04,-.01,.02,2)
        self.assertTrue(math.isfinite(rate))
        fit = fit_nelson_siegel((.5,1,2,5,10),(.03,.032,.034,.038,.04),(.5,1,2))
        self.assertIn("beta0", fit)

    def test_credit(self):
        hazard = hazard_from_spread(.01,.4)
        self.assertGreater(default_probability(hazard,5),0)
        self.assertAlmostEqual(expected_credit_loss(.02,.6,1000),12)
        curve = DiscountCurve([CurvePoint(.5,.98),CurvePoint(1,.95),CurvePoint(2,.90),CurvePoint(5,.80)])
        spread = fair_cds_spread(curve, 2, hazard, .4)
        self.assertGreater(spread, 0)
        transition_matrix_check(((.9,.1),(.2,.8)))


class EconometricsValuationTests(unittest.TestCase):
    def test_ou_and_kalman(self):
        sample = (1.0,.90,.94,.85,.88,.80,.83,.76,.79,.72,.75,.70,.73,.68,.70)
        ou = estimate_ou(sample)
        self.assertGreater(ou.mean_reversion_speed, 0)
        self.assertAlmostEqual(half_life_from_ar1(.5), 1.0)
        kalman = kalman_hedge_ratio((2,2.2,2.1,2.3),(1,1.1,1.05,1.15))
        self.assertEqual(len(kalman),4)
        self.assertIn("test_statistic", adf_regression(sample, lags=1))

    def test_dcf(self):
        result = dcf_valuation((10,12,14), .1, .03, net_debt=20, shares_outstanding=10)
        self.assertGreater(result["enterprise_value"], 0)
        self.assertGreater(dividend_discount_model(2,.08,.03),0)


class AttributionCapitalTests(unittest.TestCase):
    def test_brinson_reconciles(self):
        data = brinson_fachler({"a":.6,"b":.4},{"a":.5,"b":.5},
                               {"a":.1,"b":.03},{"a":.08,"b":.02})
        self.assertAlmostEqual(data["active_return_explained"], data["portfolio_return"]-data["benchmark_return"])
        check = active_return_reconciliation(.08,.05,{"a":.03})
        self.assertEqual(check["reconciled"],1.0)

    def test_capital_and_scenario(self):
        wacc = weighted_average_cost_of_capital(.1,.05,60,40,.25)
        self.assertGreater(wacc["wacc"],0)
        self.assertGreater(kelly_fraction(.55,1.5),0)
        self.assertEqual(factor_scenario({"equity":100},{"equity":-.1})["total_pnl"],-10)
        self.assertAlmostEqual(reverse_stress_required_move(100,10),.1)


class VolatilityStochasticTests(unittest.TestCase):
    def test_volatility_estimators(self):
        returns = (.01,-.02,.005,.01,-.01,.003,.004)
        self.assertGreater(realized_volatility(returns),0)
        self.assertGreater(bipower_variation(returns),0)
        self.assertGreater(ewma_variance(returns)[-1],0)
        self.assertGreater(garch11_variance(returns,.000001,.05,.9)[-1],0)
        self.assertGreater(garch11_unconditional_variance(.000001,.05,.9),0)
        self.assertGreater(parkinson_volatility((102,103,105),(99,100,101)),0)

    def test_simulation_reproducible_and_option_mc(self):
        a = simulate_gbm(100,.05,.2,1,10,paths=2,seed=11)
        b = simulate_gbm(100,.05,.2,1,10,paths=2,seed=11)
        self.assertEqual(a,b)
        ou = simulate_ou(1,0,1,.2,1,10,seed=11)
        self.assertEqual(len(ou[0]),11)
        mc = monte_carlo_european_option(100,100,.05,.2,1,simulations=10000)
        analytical = bs_price(100,100,1,.05,.2)
        self.assertLess(abs(mc["price"]-analytical),.5)


class LedgerLiquidityFundamentalTests(unittest.TestCase):
    def test_ledger_trade_split_dividend_nav(self):
        ledger = PortfolioLedger("USD", 10000)
        ledger.trade("ABC","buy",10,100,"USD",fee=1)
        ledger.apply_split("ABC",2)
        self.assertAlmostEqual(ledger.position("ABC"),20)
        ledger.mark({"ABC":51})
        dividend = ledger.dividend("ABC",.5)
        self.assertEqual(dividend,10)
        self.assertGreater(ledger.net_asset_value(),0)
        result = ledger.trade("ABC","sell",5,52,"USD")
        self.assertIsNotNone(result)
        self.assertGreater(ledger.realized_pnl(),0)

    def test_time_weighted_return(self):
        self.assertAlmostEqual(time_weighted_return((100,120,110),(10,-10)),
                               (120/110)*(110/110)-1)

    def test_liquidity(self):
        self.assertGreater(amihud_illiquidity((.01,-.02),(1e6,2e6)),0)
        self.assertGreater(linear_transaction_cost(100000,5),0)
        self.assertAlmostEqual(days_to_liquidate(100,100,.5),2)

    def test_fundamentals(self):
        dupont = dupont_analysis(10,100,50,25)
        self.assertAlmostEqual(dupont["roe"],.4)
        z = altman_z_score(10,20,5,40,30,100,100)
        self.assertGreater(z,0)
        score = piotroski_f_score(
            {"net_income":10,"operating_cash_flow":15,"roa":.1,"long_term_debt":20,
             "current_ratio":2,"shares_outstanding":10,"gross_margin":.4,"asset_turnover":1.2},
            {"net_income":8,"operating_cash_flow":10,"roa":.08,"long_term_debt":25,
             "current_ratio":1.8,"shares_outstanding":10,"gross_margin":.35,"asset_turnover":1.1})
        self.assertEqual(score["score"],9)


class AdvancedRiskTests(unittest.TestCase):
    def test_hierarchical_risk_parity(self):
        returns = ((.01,.02,-.01,.005,.012,-.004,.008),
                   (.009,.018,-.009,.004,.011,-.003,.007),
                   (-.01,.005,.012,-.004,.008,.002,-.003))
        weights = hierarchical_risk_parity(returns)
        self.assertAlmostEqual(sum(weights), 1.0, places=8)
        self.assertTrue(all(value >= 0 for value in weights))
        cov = covariance_matrix(returns)
        self.assertGreater(diversification_ratio(weights, cov), 1.0)

    def test_tailrisk(self):
        params = gpd_moments((.1,.2,.3,.4,.5,.6))
        q = gpd_quantile(.95, params["shape"], params["scale"])
        self.assertGreater(q, 0)
        self.assertLessEqual(gpd_survival(q, params["shape"], params["scale"]), .1)
        losses = tuple(i/1000 for i in range(1,101))
        result = peaks_over_threshold_var(losses, confidence=.95, threshold_quantile=.8)
        self.assertGreater(result["expected_shortfall"], result["var"])
        dependence = tail_dependence_count((1,2,3,4,5,6,7,8),
                                           (2,1,4,3,6,5,8,7), .75)
        self.assertGreaterEqual(dependence["joint_exceedance_rate"], 0)


if __name__ == "__main__":
    unittest.main()
