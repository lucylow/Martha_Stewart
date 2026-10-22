import unittest
from datetime import datetime, timedelta, timezone
from martha_plus.advanced.reporting import ResearchReport, ReportSection, render_html, render_markdown, write_json
from martha_plus.advanced.operations import HealthRegistry, HealthStatus, error_budget_burn, latency_percentiles, dependency_readiness
from martha_plus.advanced.scheduler import ManualScheduler, JobSpec, JobStatus
from martha_plus.advanced.feature_diagnostics import population_stability_index, correlation_matrix, alignment_diagnostics
from martha_plus.advanced.config_diff import compare_configurations, safe_configuration
from martha_plus.advanced.portfolio_diagnostics import turnover, sector_exposure

class ReportingOperationsTests(unittest.TestCase):
    def test_html_is_escaped(self):
        report = ResearchReport.create("<script>alert(1)</script>", "run1", [ReportSection("<img>", {"x":"<b>"}, ("<evil>",))])
        page = render_html(report)
        self.assertNotIn("<script>alert(1)</script>", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertIn("&lt;evil&gt;", page)
        self.assertIn("# <script>alert(1)</script>", render_markdown(report))

    def test_health_aggregation(self):
        registry = HealthRegistry()
        registry.register("db", lambda: True)
        registry.register("broker", lambda: False, critical=False)
        snapshot = registry.run()
        self.assertEqual(snapshot.status, HealthStatus.DEGRADED)
        self.assertFalse(snapshot.ready)
        self.assertTrue(error_budget_burn(errors=1, requests=100, target_availability=.99)["budget_exhausted"])
        self.assertEqual(latency_percentiles([.1,.2,.3])["count"], 3)
        self.assertFalse(dependency_readiness({"db":True}, ["db"])["ready"] is False)

    def test_scheduler_idempotency_and_retry(self):
        count = {"n":0}
        scheduler = ManualScheduler()
        scheduler.register(JobSpec("refresh", lambda: count.update(n=count["n"]+1) or "ok", interval=timedelta(minutes=1)), now=datetime(2025,1,1,tzinfo=timezone.utc))
        first = scheduler.run_once("refresh", idempotency_key="daily", now=datetime(2025,1,1,tzinfo=timezone.utc))
        second = scheduler.run_once("refresh", idempotency_key="daily", now=datetime(2025,1,1,tzinfo=timezone.utc))
        self.assertEqual(first.status, JobStatus.SUCCEEDED)
        self.assertEqual(second.status, JobStatus.SKIPPED)
        self.assertEqual(count["n"], 1)

    def test_feature_drift_and_alignment(self):
        drift = population_stability_index([1,2,3,4,5,6], [100,101,102], buckets=3)
        self.assertGreater(drift["psi"], 0)
        self.assertEqual(correlation_matrix({"x":[1,2,3], "y":[2,4,6]})["x"]["y"], 1)
        self.assertTrue(alignment_diagnostics([3,5],[4,4])["potential_leakage"])

    def test_config_diff_redacts_secrets(self):
        report = compare_configurations({"api_token":"secretA","mode":"paper"}, {"api_token":"secretB","mode":"live"})
        self.assertFalse(report["matches"])
        text = str(report)
        self.assertNotIn("secretA", text)
        self.assertNotIn("secretB", text)
        self.assertEqual(safe_configuration({"password":"pw"})["password"], "[configured]")

    def test_turnover_and_sector_exposure(self):
        self.assertAlmostEqual(turnover({"A":.5},{"A":.25,"B":.25})["gross_turnover"], .5)
        self.assertEqual(sector_exposure({"A":.5,"B":.2},{"A":"tech","B":"tech"})["tech"], .7)

if __name__ == "__main__": unittest.main()
