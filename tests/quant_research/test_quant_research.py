"""Regression tests for the dependency-free quantitative research extension."""
from __future__ import annotations
import math
import random
import unittest

from martha_plus.quant_research.core import TimeSeries, normal_ppf, normal_cdf
from martha_plus.quant_research.linalg import inverse, matmul, eigen_symmetric, cholesky
from martha_plus.quant_research import timeseries as ts
from martha_plus.quant_research import signals
from martha_plus.quant_research import validation
from martha_plus.quant_research import risk
from martha_plus.quant_research import volatility
from martha_plus.quant_research import portfolio
from martha_plus.quant_research import fixed_income
from martha_plus.quant_research import derivatives
from martha_plus.quant_research import econometrics
from martha_plus.quant_research import bayesian
from martha_plus.quant_research import performance
from martha_plus.quant_research.numerical import bisect_root,golden_section_minimize,coordinate_descent


def fixture_returns(n=160, seed=19):
    rng=random.Random(seed)
    return [rng.gauss(0.0003,0.012) for _ in range(n)]


def fixture_panel(n=160, seed=21):
    rng=random.Random(seed)
    market=[rng.gauss(0,0.01) for _ in range(n)]
    a=[0.0002+1.1*x+rng.gauss(0,0.006) for x in market]
    b=[-0.0001-0.3*x+rng.gauss(0,0.008) for x in market]
    return market,a,b


class TestCoreAndLinearAlgebra(unittest.TestCase):
    def test_timeseries_index_validation(self):
        self.assertEqual(len(TimeSeries([1,2,3],['a','b','c'])),3)
        with self.assertRaises(ValueError): TimeSeries([1,2],['a'])

    def test_normal_distribution_round_trip(self):
        for p in (0.01,0.1,0.5,0.9,0.99):
            self.assertAlmostEqual(normal_cdf(normal_ppf(p)),p,places=7)

    def test_matrix_inverse(self):
        a=[[4.0,1.0],[1.0,3.0]]
        product=matmul(a,inverse(a))
        self.assertAlmostEqual(product[0][0],1.0,places=8)
        self.assertAlmostEqual(product[1][1],1.0,places=8)

    def test_eigenvalues_and_cholesky(self):
        vals,vecs=eigen_symmetric([[2,1],[1,2]])
        self.assertAlmostEqual(vals[0],3.0,places=8)
        self.assertAlmostEqual(vals[1],1.0,places=8)
        self.assertEqual(len(cholesky([[2,1],[1,2]])),2)

    def test_singular_inverse_rejected(self):
        with self.assertRaises(ValueError): inverse([[1,2],[2,4]])


class TestTimeSeries(unittest.TestCase):
    def test_returns_and_differences(self):
        self.assertAlmostEqual(ts.simple_returns([100,110])[0],0.1)
        self.assertAlmostEqual(ts.log_returns([100,math.e*100])[0],1.0)
        self.assertEqual(ts.difference([1,3,6],1),[2,3])

    def test_fractional_difference(self):
        out=ts.fractional_difference([1,2,3,4,5],0.5,1e-8)
        self.assertEqual(len(out),1)
        self.assertAlmostEqual(out[0],2.4609375)

    def test_rolling_transforms(self):
        self.assertEqual(ts.rolling_mean([1,2,3,4],2),[None,1.5,2.5,3.5])
        self.assertEqual(ts.lag([1,2,3],1),[None,1.0,2.0])

    def test_ewm_and_winsorize(self):
        self.assertEqual(len(ts.ewm([1,2,3],2)),3)
        out=ts.winsorize([1,2,3,100],0.25,0.75)
        self.assertLess(out[-1],100)

    def test_stationarity_outputs(self):
        rng=random.Random(9)
        stationary=[rng.gauss(0,1) for _ in range(200)]
        self.assertIn('p_value_approx',ts.adf_test(stationary,max_lags=1).__dict__)
        self.assertGreaterEqual(ts.kpss_test(stationary,bandwidth=2).statistic,0)
        self.assertIn('variance_ratio',ts.variance_ratio(stationary,2))

    def test_cointegration_hedge_ratio(self):
        rng=random.Random(2); x=[100+rng.gauss(0,1) for _ in range(120)]
        noise=[rng.gauss(0,0.2) for _ in x]; y=[4+1.7*a+e for a,e in zip(x,noise)]
        intercept,beta,resid=ts.hedge_ratio(y,x)
        self.assertAlmostEqual(beta,1.7,delta=0.1)
        self.assertEqual(len(resid),len(x))
        self.assertGreater(ts.mean_reversion_half_life([math.sin(i/3) for i in range(80)]),0)

    def test_kalman_dynamic_regression(self):
        x=[math.sin(i/5) for i in range(60)]; y=[1+2*v for v in x]
        fit=ts.dynamic_regression(y,x,observation_variance=1e-3)
        self.assertEqual(len(fit['beta']),60)
        self.assertTrue(math.isfinite(fit['beta'][-1]))

    def test_hmm_probabilities(self):
        r=fixture_returns(90)
        fit=ts.fit_gaussian_hmm(r,states=2,max_iter=15)
        self.assertEqual(len(fit.states),len(r))
        for row in fit.filtered_probabilities:
            self.assertAlmostEqual(sum(row),1.0,places=7)

    def test_ar_forecast_and_diagnostics(self):
        r=fixture_returns(100)
        model=ts.fit_ar(r,2)
        self.assertEqual(len(ts.forecast_ar(model,r,3)),3)
        self.assertIn('p_value_approx',ts.ljung_box(model['residuals'],5))

    def test_spectral_and_autocorrelation(self):
        wave=[math.sin(2*math.pi*i/10) for i in range(100)]
        self.assertEqual(len(ts.periodogram(wave)),50)
        self.assertGreater(ts.dominant_frequencies(wave,1)[0]['power'],0)
        self.assertEqual(len(ts.partial_autocorrelation(wave,4)),5)

    def test_change_detection(self):
        x=[0.0]*60+[2.0]*60
        self.assertIn('signals',ts.cusum(x,allowance=0.1,threshold=4,scale=1.0))
        self.assertIn('change_points',ts.binary_segmentation(x,min_segment=10,penalty=0.1,max_changes=2))

    def test_bootstrap_and_long_memory(self):
        r=fixture_returns(100)
        self.assertEqual(len(ts.stationary_bootstrap(r,10,5)),10)
        self.assertGreaterEqual(ts.hurst_rescaled_range(r)['hurst'],-1)


class TestSignalsAndValidation(unittest.TestCase):
    def test_ic_and_summary(self):
        x=list(range(20)); y=[v+0.01*(i%2) for i,v in enumerate(x)]
        self.assertGreater(signals.information_coefficient(x,y,rank_based=True),0.99)
        self.assertIn('mean_ic',signals.ic_summary([0.1,0.2,-0.1,0.15]))

    def test_quantile_portfolios_and_turnover(self):
        panel=signals.quantile_portfolio_returns([list(range(10))],[[v/100 for v in range(10)]],5)
        self.assertEqual(len(panel[0]['quantile_returns']),5)
        self.assertAlmostEqual(signals.factor_turnover([[0.5,0.5],[0.4,0.6]])[0],0.1)

    def test_triple_barrier_and_uniqueness(self):
        labels=signals.triple_barrier_labels([100,103,97,101],[0.01]*4,max_holding=2)
        self.assertEqual(len(labels['labels']),3)
        uniqueness=signals.average_uniqueness([0,1],[2,3])
        self.assertAlmostEqual(uniqueness[0],2/3)
        self.assertEqual(signals.meta_labels([1,-1],[0.1,0.1]),(1,0))

    def test_cusum_event_filter(self):
        prices=[100*math.exp(0.02*i) for i in range(30)]
        events=signals.cusum_event_filter(prices,0.2)
        self.assertTrue(len(events)>0)

    def test_purged_splits_no_overlap(self):
        folds=validation.purged_kfold(30,3,label_end=[min(29,i+2) for i in range(30)],embargo=1)
        self.assertEqual(len(folds),3)
        for fold in folds: validation.assert_no_overlap(fold,[min(29,i+2) for i in range(30)])

    def test_walk_forward(self):
        folds=validation.walk_forward(40,initial_train=20,test_size=5,embargo=1)
        self.assertEqual(folds[0].test,(20,21,22,23,24))
        self.assertTrue(all(max(f.train)<min(f.test) for f in folds))

    def test_deflated_sharpe_and_multiple_testing(self):
        r=fixture_returns()
        self.assertGreaterEqual(validation.probabilistic_sharpe(r),0)
        self.assertLessEqual(validation.deflated_sharpe(r,20)['deflated_sharpe_probability'],1)
        bh=validation.benjamini_hochberg([0.001,0.01,0.2,0.5],0.05)
        self.assertTrue(bh['rejected'][0])

    def test_pbo(self):
        matrix=[[0.2,0.1,0.3,0.1],[0.1,0.2,0.1,0.3],[0.0,0.1,0.2,0.1]]
        self.assertGreaterEqual(validation.probability_of_backtest_overfitting(matrix)['pbo'],0)

    def test_reality_check(self):
        base=fixture_returns(60,2)
        out=validation.reality_check(base,[[v+0.0001 for v in base]],bootstrap_samples=20,block_size=3)
        self.assertGreaterEqual(out['p_value'],0)
        self.assertLessEqual(out['p_value'],1)

    def test_permutation_importance(self):
        x=[list(range(30)),[math.sin(i) for i in range(30)]]; y=[float(i) for i in range(30)]
        def score(cols,target):
            return -sum((a-b)**2 for a,b in zip(cols[0],target))/len(target)
        imp=validation.permutation_importance(x,y,score,repeats=4,block_size=3)
        self.assertEqual(len(imp['features']),2)


class TestRiskAndPortfolio(unittest.TestCase):
    def setUp(self):
        m,a,b=fixture_panel()
        self.panel=list(zip(m,a,b))
        self.cov=risk.sample_covariance(self.panel)

    def test_covariance_estimators(self):
        self.assertEqual(len(risk.ewma_covariance(self.panel)),3)
        self.assertEqual(len(risk.shrink_covariance(self.panel)['covariance']),3)
        self.assertEqual(len(risk.pca_risk(self.cov,2)['eigenvalues']),2)

    def test_eigen_cleaning(self):
        clean=risk.eigenvalue_clean([[1,1.1],[1.1,1]])
        self.assertGreater(clean['cleaned_eigenvalues'][-1],0)

    def test_risk_contributions(self):
        out=risk.euler_risk_contribution([0.3,0.3,0.4],self.cov)
        self.assertAlmostEqual(out['contribution_sum'],out['volatility'],places=9)

    def test_copula_scenarios(self):
        cols=[list(x) for x in zip(*self.panel)]
        corr=risk.correlation_matrix(self.cov)
        scenarios=risk.gaussian_copula_scenarios(cols,corr,25)
        self.assertEqual(len(scenarios),25)

    def test_stress_and_drawdown(self):
        out=risk.scenario_summary([0.5,0.5],[[0.01,-0.02],[-0.1,-0.1],[0.02,0.03]])
        self.assertGreater(out['expected_shortfall_loss'],0)
        self.assertIn('cdar',risk.drawdown_summary([0.02,-0.1,0.04,-0.02]))

    def test_liquidity(self):
        out=risk.liquidation_cost([1e6],[1e7],[0.02],spread_bps=[3])
        self.assertGreater(out['total_cost'],0)
        self.assertGreater(risk.capacity_estimate(1e6,0.1),0)

    def test_regime_risk_mix(self):
        regs=[0 if i<80 else 1 for i in range(160)]
        out=risk.regime_covariances(self.panel,regs,min_observations=5)
        self.assertEqual(len(out['covariances']),2)

    def test_black_litterman(self):
        prior=[0.01,0.02,0.03]; p=[[1,-1,0]]; q=[0.015]
        out=portfolio.posterior(self.cov,prior,p,q)
        self.assertEqual(len(out['posterior_returns']),3)

    def test_constrained_optimizer(self):
        out=portfolio.optimize_mean_variance([0.01,0.02,0.015],self.cov,risk_aversion=4,upper=0.8)
        self.assertAlmostEqual(sum(out['weights']),1.0,places=7)
        self.assertTrue(all(0<=w<=0.8 for w in out['weights']))

    def test_risk_budget(self):
        out=portfolio.equal_risk_contribution(self.cov,max_iter=10000,tolerance=1e-5)
        self.assertAlmostEqual(sum(out['weights']),1.0,places=6)

    def test_rebalance_cost(self):
        drift=portfolio.drifted_weights([0.5,0.5],[0.1,0.0])
        self.assertAlmostEqual(sum(drift),1.0)
        costs=portfolio.proportional_transaction_cost([100,-50],[10,20],[0.001,0.002])
        self.assertGreater(costs['total_cost'],0)

    def test_robust_estimators(self):
        self.assertGreater(portfolio.huber_location([1,1,1,1,100])['location'],0)
        self.assertEqual(len(portfolio.robust_expected_returns(self.panel)),3)

    def test_factor_model(self):
        m,a,b=fixture_panel()
        result=portfolio.factor_risk_model([a,b],[m])
        self.assertEqual(len(result['asset_covariance']),2)


class TestVolatilityAndDerivatives(unittest.TestCase):
    def test_realized_measures(self):
        r=[0.001,-0.002,0.003,-0.001]
        self.assertGreater(volatility.realized_variance(r),0)
        self.assertGreaterEqual(volatility.jump_variation(r)['jump_variation'],0)
        self.assertGreaterEqual(volatility.realized_kernel(r,2),0)

    def test_garch_models(self):
        r=fixture_returns(100)
        fit=volatility.fit_garch11(r,max_iter=15)
        self.assertLess(fit.persistence,1)
        self.assertEqual(len(volatility.forecast_variance(fit,3)),3)
        gjr=volatility.fit_gjr_garch(r,max_iter=10)
        self.assertLess(gjr.persistence,1)

    def test_har_model(self):
        rv=[0.0001*(1+0.2*math.sin(i/6)) for i in range(150)]
        fit=volatility.fit_har(rv)
        self.assertEqual(len(volatility.forecast_har(fit,rv,2)),2)

    def test_var_backtest(self):
        losses=[0.01]*100; var=[0.02]*100
        pof=volatility.kupiec_pof_test(losses,var)
        self.assertGreaterEqual(pof['p_value_approx'],0)
        self.assertIn('p_value_approx',volatility.conditional_coverage_test(losses,var))

    def test_futures_and_swap(self):
        self.assertGreater(derivatives.fair_futures_price(100,0.03,1),100)
        rate=derivatives.par_swap_rate([1,2],[0.97,0.93])
        self.assertGreater(rate,0)
        self.assertGreater(derivatives.swap_pv(rate,[1,2],[0.97,0.93],[0.03,0.04])['annuity'],0)

    def test_bond_measures(self):
        cfs=fixed_income.bond_cashflows(100,0.04,5)
        price=fixed_income.present_value(cfs,[0.03]*len(cfs))
        self.assertGreater(price,0)
        self.assertGreater(fixed_income.modified_duration(cfs,0.04),0)
        self.assertGreater(fixed_income.convexity(cfs,0.04),0)

    def test_bootstrap_curve(self):
        curve=fixed_income.bootstrap_discount_curve([(0.5,0.02),(1.0,0.022)],[(2.0,0.025),(3.0,0.027)])
        self.assertTrue(all(a>b for a,b in zip(curve['discount_factors'],curve['discount_factors'][1:])))

    def test_credit_curve(self):
        times=[1,2,3]; hazards=[0.01,0.015,0.02]
        survival=fixed_income.survival_from_hazards(times,hazards)
        recovered=fixed_income.hazard_from_survival(times,survival)
        self.assertAlmostEqual(recovered[0],hazards[0])
        spread=fixed_income.cds_par_spread(times,[0.98,0.95,0.91],survival)
        self.assertGreater(spread['par_spread'],0)


class TestBayesianAndEconometrics(unittest.TestCase):
    def test_bayesian_updates(self):
        post=bayesian.normal_normal_posterior([0.01,0.02,0.03],0,1,0.1)
        self.assertGreater(post['mean'],0)
        self.assertGreater(bayesian.normal_inverse_gamma_posterior([1,2,3])['variance_shape'],2)

    def test_bayesian_ensemble(self):
        out=bayesian.bayesian_model_average([[1,2],[2,3]],[1.1,2.1],[[0.1,0.1],[0.2,0.2]])
        self.assertAlmostEqual(sum(out['posterior_model_probabilities']),1.0)

    def test_ols_and_robust_regression(self):
        x=list(range(20)); y=[2+3*v for v in x]; y[-1]+=100
        fit=econometrics.huber_regression([x],y)
        self.assertAlmostEqual(fit['coefficients'][1],3.0,delta=0.3)
        q=econometrics.quantile_regression([x],y,0.5,max_iter=30)
        self.assertTrue(math.isfinite(q['check_loss']))

    def test_var_and_granger(self):
        rng=random.Random(8); x=[rng.gauss(0,1) for _ in range(100)]; y=[0.0]
        for i in range(1,100): y.append(0.5*y[-1]+0.4*x[i-1]+rng.gauss(0,0.5))
        fit=econometrics.fit_var([x,y],1)
        self.assertEqual(len(econometrics.forecast_var(fit,[x,y],2)),2)
        self.assertIn('p_value_approx',econometrics.granger_causality([x,y],0,1,1))

    def test_panel_fixed_effects(self):
        entities=['a','a','a','b','b','b']; times=['1','2','3']*2
        x=[1,2,3,2,3,4]; y=[2,4,6,5,7,9]
        fit=econometrics.fixed_effects_regression([x],y,entities,times)
        self.assertAlmostEqual(fit['coefficients'][0],2.0,places=6)

    def test_regression_diagnostics(self):
        x=list(range(30)); y=[2+0.5*v+math.sin(v) for v in x]
        fit=econometrics.weighted_least_squares([x],y,[1]*len(x))
        self.assertEqual(len(fit['coefficients']),2)
        self.assertEqual(len(econometrics.variance_inflation_factors([x,[v*v for v in x]])),2)


class TestPerformanceAndNumerics(unittest.TestCase):
    def test_performance_summary(self):
        r=[0.01,-0.02,0.015,0.01,-0.005]
        out=performance.summary(r)
        self.assertIn('sharpe',out)
        self.assertLessEqual(out['max_drawdown'],0)

    def test_benchmarking(self):
        p=[0.01,0.02,-0.01,0.03]; b=[0.005,0.01,-0.02,0.01]
        self.assertIn('up_capture',performance.up_down_capture(p,b))
        self.assertIn('information_ratio',performance.benchmark_summary(p,b))

    def test_numerical_optimizers(self):
        root=bisect_root(lambda x:x*x-4,0,3)
        self.assertAlmostEqual(root['root'],2.0,places=6)
        minimum=golden_section_minimize(lambda x:(x-3)**2,-5,8)
        self.assertAlmostEqual(minimum['x'],3.0,places=5)
        result=coordinate_descent(lambda x:(x[0]-1)**2,[4.0],step=1,max_iter=100)
        self.assertAlmostEqual(result['point'][0],1.0,delta=0.01)


if __name__=='__main__':
    unittest.main(verbosity=2)

class TestAdvancedMultivariateAndMonteCarlo(unittest.TestCase):
    def test_dcc_fit_and_one_step_forecast(self):
        rng = random.Random(1234)
        rows = []
        for _ in range(150):
            common = rng.gauss(0, 0.01)
            rows.append([common + rng.gauss(0, 0.005),
                         0.6*common + rng.gauss(0, 0.007),
                         -0.2*common + rng.gauss(0, 0.006)])
        fit = volatility.fit_dcc(rows, marginal="sample", a_values=(0.03, 0.08),
                                 b_values=(0.80, 0.90))
        self.assertEqual(fit.assets, 3)
        self.assertLess(fit.a+fit.b, 0.995)
        forecast = volatility.forecast_dcc(fit)
        self.assertEqual(len(forecast["correlation"]), 3)
        self.assertAlmostEqual(forecast["correlation"][0][0], 1.0, places=7)
        self.assertGreater(volatility.dcc_portfolio_volatility([1/3]*3, forecast), 0)

    def test_dcc_garch_marginals(self):
        rng = random.Random(222)
        rows = [[rng.gauss(0, 0.01), 0.5*rng.gauss(0, 0.01)+rng.gauss(0, 0.008)]
                for _ in range(90)]
        fit = volatility.fit_dcc(rows, marginal="garch", a_values=(0.03,), b_values=(0.85,))
        self.assertEqual(fit.marginal_method, "garch")
        self.assertEqual(len(fit.conditional_volatility), 90)

    def test_dcc_rolling_forecasts(self):
        rng = random.Random(4321)
        rows = [[rng.gauss(0, 0.01), rng.gauss(0, 0.015)] for _ in range(100)]
        forecasts = volatility.rolling_dcc_forecasts(rows, window=60, step=20, marginal="sample")
        self.assertEqual(len(forecasts), 2)
        self.assertEqual(forecasts[0]["forecast_index"], 60)

    def test_gbm_and_european_option_reproducibility(self):
        paths = derivatives.simulate_gbm_paths(100, 0.03, 0.01, 0.2, 1,
                                               paths=200, steps=20, seed=11)
        self.assertEqual(paths, derivatives.simulate_gbm_paths(100, 0.03, 0.01, 0.2, 1,
                                                               paths=200, steps=20, seed=11))
        estimate = derivatives.monte_carlo_european(100, 100, 0.03, 0.01, 0.2, 1,
                                                    paths=5000, steps=1, seed=12)
        self.assertGreater(estimate.price, 0)
        self.assertGreater(estimate.standard_error, 0)
        self.assertLess(estimate.confidence_interval[0], estimate.price)
        self.assertGreater(estimate.confidence_interval[1], estimate.price)

    def test_asian_and_discrete_barrier_options(self):
        arithmetic = derivatives.monte_carlo_asian(100, 100, 0.02, 0.0, 0.2, 1,
                                                    paths=3000, steps=24, seed=2)
        geometric = derivatives.monte_carlo_asian(100, 100, 0.02, 0.0, 0.2, 1,
                                                   paths=3000, steps=24, seed=2,
                                                   geometric_average=True)
        self.assertGreaterEqual(arithmetic.price, geometric.price)
        vanilla = derivatives.monte_carlo_european(100, 100, 0.02, 0.0, 0.2, 1,
                                                    paths=2000, steps=24, seed=3)
        no_hit = derivatives.monte_carlo_barrier(100, 100, 1e6, 0.02, 0.0, 0.2, 1,
                                                 paths=2000, steps=24, seed=3,
                                                 barrier_type="up_out")
        self.assertAlmostEqual(vanilla.price, no_hit.price, places=10)

    def test_heston_monte_carlo_and_common_random_numbers(self):
        estimate = derivatives.monte_carlo_heston(100, 100, 0.02, 0.0, 0.04,
                                                  2.0, 0.04, 0.3, -0.6, 1.0,
                                                  paths=1000, steps=40, seed=9)
        self.assertGreater(estimate.price, 0)
        self.assertIn("feller_condition", estimate.diagnostics)
        greeks = derivatives.common_random_number_greeks(
            lambda s, v: derivatives.monte_carlo_european(s, 100, 0.02, 0.0, v, 1,
                paths=1000, steps=1, seed=5).price, 100, 0.2)
        self.assertTrue(math.isfinite(greeks["delta"]))
        self.assertTrue(math.isfinite(greeks["gamma"]))


class TestVarianceSwapAnalytics(unittest.TestCase):
    def test_realized_and_forward_variance(self):
        rv = derivatives.annualized_realized_variance([0.01, -0.02, 0.015, 0.005])
        self.assertGreater(rv, 0)
        path = derivatives.realized_variance_path([0.01, -0.02, 0.015, 0.005], 2)
        self.assertIsNone(path[0])
        self.assertGreater(path[-1], 0)
        fwd = derivatives.forward_variance(1.0, 2.0, 0.04, 0.10)
        self.assertAlmostEqual(fwd, 0.16)

    def test_variance_risk_premium_and_settlement(self):
        vrp = derivatives.variance_risk_premium(0.05, 0.04)
        self.assertAlmostEqual(vrp["variance_risk_premium"], 0.01)
        pnl = derivatives.variance_swap_pnl(0.06, 0.05, 10000)
        self.assertAlmostEqual(pnl["payoff"], 100.0)
        self.assertGreater(derivatives.volatility_swap_approximation(0.05, 0.04)["approximate_volatility_strike"], 0)

    def test_variance_term_structure_and_option_strip(self):
        term = derivatives.variance_term_structure([0.5, 1.0, 2.0], [0.04, 0.05, 0.06])
        self.assertTrue(term["nondecreasing_total_variance"])
        strip = derivatives.variance_swap_strike_from_quotes(100, 1.0,
            [80, 90, 100, 110, 120], [1.0, 3.0, 5.0, 3.2, 1.2],
            ["put", "put", "put", "call", "call"], 0.02)
        self.assertGreaterEqual(strip["variance_strike"], 0)
        self.assertGreater(strip["volatility_strike"], 0)
