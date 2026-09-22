import { useState } from 'react';
import PerturbationExplainer from '../components/robustness/PerturbationExplainer';
import RobustnessChart from '../components/robustness/RobustnessChart';
import RobustnessTable from '../components/robustness/RobustnessTable';
import EngineSelector from '../components/predict/EngineSelector';
import SkeletonBar, { SkeletonBlock } from '../components/shared/SkeletonBar';
import { useAppState } from '../state/AppContext';
import { runRobustness } from '../api/inference';
import './Robustness.css';

export default function Robustness() {
  const { state, dispatch } = useAppState();
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const robustness = state.robustness;

  async function handleRun({ split, engineId }) {
    setLoading(true);
    setError(null);
    try {
      const result = await runRobustness({ split, engineId });
      dispatch({ type: 'SET_ROBUSTNESS', payload: result });
    } catch (e) {
      setError(e.message || 'Stress test failed.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="rob-page">
      <div className="rob-page__header">
        <h1 className="rob-page__title">Robustness</h1>
        <p className="rob-page__subtitle">
          Sensors fail. This page tests what happens to the prediction when they do.
        </p>
      </div>

      <PerturbationExplainer />

      <EngineSelector onRun={handleRun} loading={loading} />

      {error && (
        <div className="rob-page__error" role="alert">{error}</div>
      )}

      {loading && (
        <div style={{ marginTop: 'var(--space-10)' }} aria-busy="true" aria-label="Running stress test">
          <SkeletonBlock>
            <SkeletonBar height={200} />
            <SkeletonBar height={180} />
          </SkeletonBlock>
        </div>
      )}

      {!loading && robustness && (
        <>
          <div className="rob-page__result-meta">
            Testing Engine <span className="rob-page__engine-id">#{robustness.engId}</span>
          </div>
          <RobustnessChart results={robustness.results} />
          <RobustnessTable results={robustness.results} />
        </>
      )}
    </div>
  );
}
