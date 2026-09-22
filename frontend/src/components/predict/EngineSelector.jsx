import { useState, useEffect, useRef } from 'react';
import { Upload } from 'lucide-react';
import Button from '../shared/Button';
import { fetchEngineList } from '../../api/inference';
import './EngineSelector.css';

const SPLITS = [
  { value: 'target', label: 'Target cohort (unseen engines)' },
  { value: 'val',    label: 'Validation cohort' },
  { value: 'train',  label: 'Training cohort' },
];

export default function EngineSelector({ onRun, loading }) {
  const [split, setSplit]       = useState('target');
  const [engineId, setEngineId] = useState(null);
  const [engines, setEngines]   = useState([]);
  const [file, setFile]         = useState(null);
  const [mode, setMode]         = useState('sample'); // 'sample' | 'upload'
  const fileRef = useRef(null);

  useEffect(() => {
    fetchEngineList(split).then(list => {
      setEngines(list);
      const first = list[0];
      const firstId = typeof first === 'object' && first !== null ? first.id : first;
      setEngineId(firstId ?? null);
    });
  }, [split]);

  function handleSubmit(e) {
    e.preventDefault();
    if (mode === 'upload' && !file) return;
    onRun({ split, engineId, file: mode === 'upload' ? file : null });
  }

  return (
    <form className="engine-selector" onSubmit={handleSubmit} aria-label="Engine selection">
      {/* Mode toggle */}
      <div className="engine-selector__tabs" role="tablist">
        <button
          role="tab"
          type="button"
          aria-selected={mode === 'sample'}
          className={`engine-selector__tab ${mode === 'sample' ? 'engine-selector__tab--active' : ''}`}
          onClick={() => setMode('sample')}
        >
          Sample engine
        </button>
        <button
          role="tab"
          type="button"
          aria-selected={mode === 'upload'}
          className={`engine-selector__tab ${mode === 'upload' ? 'engine-selector__tab--active' : ''}`}
          onClick={() => setMode('upload')}
        >
          Upload telemetry
        </button>
      </div>

      <div className="engine-selector__controls">
        {mode === 'sample' ? (
          <>
            <div className="engine-selector__field">
              <label htmlFor="split-select" className="engine-selector__label">
                Cohort
              </label>
              <select
                id="split-select"
                className="engine-selector__select"
                value={split}
                onChange={e => setSplit(e.target.value)}
              >
                {SPLITS.map(s => (
                  <option key={s.value} value={s.value}>{s.label}</option>
                ))}
              </select>
            </div>

            <div className="engine-selector__field engine-selector__field--engine">
              <label htmlFor="engine-select" className="engine-selector__label">
                Engine Unit
              </label>
              <select
                id="engine-select"
                className="engine-selector__select"
                value={engineId ?? ''}
                onChange={e => setEngineId(Number(e.target.value))}
              >
                {engines.map(item => {
                  const id = typeof item === 'object' && item !== null ? item.id : item;
                  return (
                    <option key={id} value={id}>
                      Engine #{id}
                    </option>
                  );
                })}
              </select>
            </div>
          </>
        ) : (
          <div className="engine-selector__field engine-selector__field--upload">
            <label htmlFor="file-upload" className="engine-selector__label">
              Telemetry file (.txt or .csv, 26 columns — NASA C-MAPSS format)
            </label>
            <div
              className={`engine-selector__dropzone ${file ? 'engine-selector__dropzone--active' : ''}`}
              onClick={() => fileRef.current?.click()}
              onKeyDown={e => e.key === 'Enter' && fileRef.current?.click()}
              role="button"
              tabIndex={0}
              aria-label="Click or press Enter to choose a file"
            >
              <Upload size={16} className="engine-selector__upload-icon" aria-hidden="true" />
              <span>{file ? file.name : 'Choose file'}</span>
            </div>
            <input
              ref={fileRef}
              id="file-upload"
              type="file"
              accept=".txt,.csv"
              className="engine-selector__file-input"
              onChange={e => setFile(e.target.files?.[0] ?? null)}
              aria-label="Upload telemetry file"
            />
          </div>
        )}

        <Button
          type="submit"
          variant="primary"
          size="md"
          loading={loading}
          disabled={loading || (mode === 'upload' && !file)}
          className="engine-selector__run"
        >
          {loading ? 'Running…' : 'Predict RUL'}
        </Button>
      </div>
    </form>
  );
}
