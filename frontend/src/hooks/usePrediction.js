import { useState, useCallback } from 'react';
import { runPrediction } from '../api/inference';
import { useAppState } from '../state/AppContext';

export function usePrediction() {
  const { dispatch } = useAppState();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const predict = useCallback(async ({ split, engineId, file }) => {
    setLoading(true);
    setError(null);
    try {
      const result = await runPrediction({ split, engineId, file });
      dispatch({ type: 'SET_PREDICTION', payload: result });
    } catch (err) {
      setError(err.message || 'Prediction failed. Check the console for details.');
    } finally {
      setLoading(false);
    }
  }, [dispatch]);

  const clear = useCallback(() => {
    dispatch({ type: 'CLEAR_PREDICTION' });
  }, [dispatch]);

  return { predict, clear, loading, error };
}
