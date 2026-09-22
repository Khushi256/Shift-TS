import { createContext, useContext, useReducer } from 'react';

const AppContext = createContext(null);

const initialState = {
  prediction: null,      // { rul, lower, upper, std, cycles, means, stds, ruls, sourceLabel, timestamp }
  robustness: null,      // { engId, results: { [name]: { mae, rmse } } }
  modelStatus: 'dev',    // 'trained' | 'dev'
};

function reducer(state, action) {
  switch (action.type) {
    case 'SET_PREDICTION':
      return { ...state, prediction: action.payload };
    case 'CLEAR_PREDICTION':
      return { ...state, prediction: null };
    case 'SET_ROBUSTNESS':
      return { ...state, robustness: action.payload };
    case 'CLEAR_ROBUSTNESS':
      return { ...state, robustness: null };
    case 'SET_MODEL_STATUS':
      return { ...state, modelStatus: action.payload };
    default:
      return state;
  }
}

export function AppProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
}

export function useAppState() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useAppState must be inside AppProvider');
  return ctx;
}
