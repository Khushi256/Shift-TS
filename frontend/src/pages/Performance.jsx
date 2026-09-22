import { useState, useEffect } from 'react';
import MetricsTable from '../components/performance/MetricsTable';
import RiskCoverageChart from '../components/performance/RiskCoverageChart';
import DistributionDiagram from '../components/performance/DistributionDiagram';
import Disclosure from '../components/shared/Disclosure';
import SkeletonBar, { SkeletonBlock } from '../components/shared/SkeletonBar';
import { fetchPerformanceMetrics } from '../api/inference';
import './Performance.css';

export default function Performance() {
  const [metrics, setMetrics]   = useState(null);
  const [loading, setLoading]   = useState(true);

  useEffect(() => {
    fetchPerformanceMetrics()
      .then(setMetrics)
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="perf-page">
      <div className="perf-page__header">
        <h1 className="perf-page__title">Performance</h1>
        <p className="perf-page__subtitle">
          Benchmark results on the target cohort — engines the model never saw during training.
        </p>
      </div>

      {loading ? (
        <SkeletonBlock>
          <SkeletonBar height={28} width="260px" />
          <SkeletonBar height={180} />
          <SkeletonBar height={220} />
        </SkeletonBlock>
      ) : metrics ? (
        <>
          <MetricsTable />

          <RiskCoverageChart
            data={metrics.riskCoverage}
            baselineMae={metrics.mae}
          />

          <DistributionDiagram />

          <div style={{ marginTop: 'var(--space-12)' }}>
            <Disclosure summary="How uncertainty estimation works — MC Dropout">
              <p style={{ margin: 0 }}>
                Monte Carlo Dropout performs T=20 stochastic forward passes through the model
                with dropout active at inference time. The standard deviation across these passes
                approximates the predictive uncertainty of a deep Gaussian Process posterior —
                without the computational cost of full Bayesian inference or MCMC sampling.
                The error–uncertainty Spearman rank correlation (ρ = {metrics.rho}) confirms
                that higher uncertainty correctly identifies higher-error predictions.
              </p>
            </Disclosure>
          </div>
        </>
      ) : (
        <p style={{ color: 'var(--color-text-muted)' }}>Could not load performance data.</p>
      )}
    </div>
  );
}
