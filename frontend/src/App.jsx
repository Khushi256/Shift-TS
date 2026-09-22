import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AppProvider } from './state/AppContext';
import Shell from './components/layout/Shell';
import Landing     from './pages/Landing';
import Predict     from './pages/Predict';
import Performance from './pages/Performance';
import Robustness  from './pages/Robustness';
import About       from './pages/About';

export default function App() {
  return (
    <AppProvider>
      <BrowserRouter>
        <Shell>
          <Routes>
            <Route path="/"            element={<Landing />} />
            <Route path="/predict"     element={<Predict />} />
            <Route path="/performance" element={<Performance />} />
            <Route path="/robustness"  element={<Robustness />} />
            <Route path="/about"       element={<About />} />
          </Routes>
        </Shell>
      </BrowserRouter>
    </AppProvider>
  );
}
