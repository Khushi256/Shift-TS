import { useState, useEffect } from 'react';
import MetricsTable from '../components/performance/MetricsTable';
import FewShotTable from '../components/performance/FewShotTable';
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
          Benchmark results on the target cohort — engines evaluated under zero-shot transfer and few-shot adaptation ablation.
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
          <MetricsTable
            rmse={metrics.rmse}
            mae={metrics.mae}
            coverage95={metrics.coverage95}
            rho={metrics.rho}
          />

          <FewShotTable
            fewShotData={metrics.fewShotExperiment}
            zeroShotMae={metrics.mae}
          />

          <RiskCoverageChart
            data={metrics.riskCoverage}
            baselineMae={metrics.mae}
          />

          <DistributionDiagram />

          <div style={{ marginTop: 'var(--space-12)' }}>
            <Disclosure summary="How uncertainty estimation works — MC Dropout">
              <p style={{ margin: 0 }}>
                Monte Carlo Dropout performs T={metrics.mcPasses ?? 20} stochastic forward passes through the model
                with dropout active at inference time. The mean across passes is the point estimate;
                the standard deviation is an uncertainty-aware heuristic proxy — not a calibrated Bayesian
                posterior. The error–uncertainty Spearman rank correlation (ρ = {metrics.rho}) reflects that
                lower uncertainty → lower error at lower coverage under selective prediction.
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
