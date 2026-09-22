import EngineSelector from '../components/predict/EngineSelector';
import PredictEmpty from '../components/predict/PredictEmpty';
import PredictResult from '../components/predict/PredictResult';
import SkeletonBar, { SkeletonBlock } from '../components/shared/SkeletonBar';
import { usePrediction } from '../hooks/usePrediction';
import { useAppState } from '../state/AppContext';
import './Predict.css';

export default function Predict() {
  const { state } = useAppState();
  const { predict, clear, loading, error } = usePrediction();
  const prediction = state.prediction;

  return (
    <div className="predict-page">
      <div className="predict-page__header">
        <h1 className="predict-page__title">Predict Remaining Useful Life</h1>
        <p className="predict-page__subtitle">
          Run calibrated RUL prediction with uncertainty estimation on NASA C-MAPSS FD002 engine data.
        </p>
      </div>

      <EngineSelector onRun={predict} loading={loading} />

      {error && (
        <div className="predict-page__error" role="alert">
          {error}
        </div>
      )}

      {loading && (
        <div style={{ marginTop: 'var(--space-10)' }} aria-busy="true" aria-label="Running prediction">
          <SkeletonBlock>
            <SkeletonBar height={48} width="220px" />
            <SkeletonBar height={4} />
            <SkeletonBar height={56} />
            <SkeletonBar height={240} />
          </SkeletonBlock>
        </div>
      )}

      {!loading && !prediction && <PredictEmpty />}

      {!loading && prediction && (
        <PredictResult data={prediction} onReset={clear} />
      )}
    </div>
  );
}
