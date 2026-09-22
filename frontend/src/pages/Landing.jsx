import { Link } from 'react-router-dom';
import {
  ArrowRight,
  Gauge,
  BarChart2,
  Shield,
  BookOpen,
  Cpu,
  Layers,
  Radio,
  Database,
} from 'lucide-react';
import './Landing.css';

const PIPELINE_STAGES = [
  {
    step: '01',
    title: 'Multivariate Ingestion',
    tech: '21 Sensors · 3 Flight Settings',
    summary: 'Processes 30-cycle sliding telemetry windows normalized across 6 distinct NASA C-MAPSS operating regimes.',
    icon: Database,
  },
  {
    step: '02',
    title: 'Self-Supervised Encoding',
    tech: 'InfoNCE + Masked Recovery',
    summary: 'Pre-trains recurrent representations on unlabelled degradation trends to build robust fleet-invariant features.',
    icon: Cpu,
  },
  {
    step: '03',
    title: 'Few-Shot Adaptation',
    tech: 'ProtoNet · 1% Target Labels',
    summary: 'Fine-tunes the latent boundary for novel operating environments using minimal labelled flight trajectories.',
    icon: Layers,
  },
  {
    step: '04',
    title: 'Calibrated Inference',
    tech: '2-Layer GRU · MC Dropout',
    summary: 'Generates point RUL estimates accompanied by 95% credible intervals (±1.96σ) across T=20 stochastic forward passes.',
    icon: Radio,
  },
];

const MODULES = [
  {
    to: '/predict',
    icon: Gauge,
    tag: 'Operational Workstation',
    title: 'RUL Prediction & Telemetry',
    desc: 'Select engines across test and validation cohorts or upload raw CSV telemetry to evaluate remaining cycles alongside credible intervals.',
    action: 'Open Predictor',
  },
  {
    to: '/performance',
    icon: BarChart2,
    tag: 'Validation Suite',
    title: 'Model Benchmarks & Coverage',
    desc: 'Examine zero-shot target RMSE, error-uncertainty correlation (ρ ≈ 0.224), and selective prediction risk-coverage trade-offs.',
    action: 'View Performance',
  },
  {
    to: '/robustness',
    icon: Shield,
    tag: 'Stress Diagnostics',
    title: 'Sensor Fault Resilience',
    desc: 'Perturb sensor channels with synthetic Gaussian noise, channel dropouts, and calibration drift to evaluate model failure modes.',
    action: 'Run Stress Tests',
  },
];

const SPECS = [
  { label: 'Architecture', value: '2-Layer GRU (64 hidden units), MC Dropout (p=0.2)' },
  { label: 'Input Specification', value: '30-cycle sliding window, 24 features (21 sensors + 3 operational settings)' },
  { label: 'Uncertainty Estimation', value: 'Monte Carlo Dropout with T=20 stochastic inference passes (95% CI)' },
  { label: 'Target Fleet Benchmark', value: 'NASA C-MAPSS FD002 (6 operating regimes, 260 train / 259 test engines)' },
  { label: 'Zero-Shot Target RMSE', value: '20.35 cycles on unseen target cohort under distribution shift' },
  { label: 'Calibration Fidelity', value: 'Spearman rank correlation ρ ≈ 0.224 between predictive variance and true error' },
];

export default function Landing() {
  return (
    <div className="landing-page">
      {/* Hero Section */}
      <section className="landing-hero" aria-label="System Introduction">
        <div className="landing-hero__status">
          <span className="landing-hero__status-dot" aria-hidden="true" />
          <span className="landing-hero__status-text">SYSTEM READY · NASA C-MAPSS FD002 PROGNOSTICS</span>
        </div>

        <h1 className="landing-hero__title">
          Turbofan Degradation Prognostics Under Operational Condition Shift
        </h1>

        <p className="landing-hero__subtitle">
          A deep learning prognostics framework for aircraft engine Remaining Useful Life (RUL) estimation,
          coupling self-supervised representation learning with calibrated epistemic uncertainty bounds.
        </p>

        <div className="landing-hero__actions">
          <Link to="/predict" className="landing-btn landing-btn--primary">
            <span>Launch Predictor</span>
            <ArrowRight size={15} aria-hidden="true" />
          </Link>
          <Link to="/performance" className="landing-btn landing-btn--secondary">
            <span>View Benchmarks</span>
          </Link>
          <Link to="/about" className="landing-btn landing-btn--ghost">
            <BookOpen size={15} aria-hidden="true" />
            <span>Architecture Dossier</span>
          </Link>
        </div>
      </section>

      {/* Pipeline Flow Section */}
      <section className="landing-section" aria-label="Prognostics Pipeline">
        <div className="landing-section__header">
          <span className="landing-section__kicker">End-to-End Architecture</span>
          <h2 className="landing-section__heading">Prognostics Pipeline</h2>
          <p className="landing-section__subheading">
            From raw multivariate sensor telemetry to calibrated decision support.
          </p>
        </div>

        <div className="landing-pipeline">
          {PIPELINE_STAGES.map(stage => {
            const Icon = stage.icon;
            return (
              <div key={stage.step} className="landing-pipeline__card">
                <div className="landing-pipeline__top">
                  <span className="landing-pipeline__step">{stage.step}</span>
                  <div className="landing-pipeline__icon-box">
                    <Icon size={16} aria-hidden="true" />
                  </div>
                </div>
                <h3 className="landing-pipeline__title">{stage.title}</h3>
                <span className="landing-pipeline__tech">{stage.tech}</span>
                <p className="landing-pipeline__summary">{stage.summary}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* Core Workstations / Modules */}
      <section className="landing-section" aria-label="System Modules">
        <div className="landing-section__header">
          <span className="landing-section__kicker">Interactive Modules</span>
          <h2 className="landing-section__heading">System Workstations</h2>
          <p className="landing-section__subheading">
            Direct access to operational inference, benchmark validation, and sensor perturbation tests.
          </p>
        </div>

        <div className="landing-modules">
          {MODULES.map(mod => {
            const Icon = mod.icon;
            return (
              <Link key={mod.to} to={mod.to} className="landing-module">
                <div className="landing-module__top">
                  <div className="landing-module__icon-box">
                    <Icon size={18} aria-hidden="true" />
                  </div>
                  <span className="landing-module__tag">{mod.tag}</span>
                </div>

                <h3 className="landing-module__title">{mod.title}</h3>
                <p className="landing-module__desc">{mod.desc}</p>

                <div className="landing-module__footer">
                  <span className="landing-module__action">{mod.action}</span>
                  <ArrowRight size={14} className="landing-module__arrow" aria-hidden="true" />
                </div>
              </Link>
            );
          })}
        </div>
      </section>

      {/* Technical Specifications Matrix */}
      <section className="landing-section" aria-label="Technical Specifications">
        <div className="landing-section__header">
          <span className="landing-section__kicker">Reference Parameters</span>
          <h2 className="landing-section__heading">System Specifications</h2>
          <p className="landing-section__subheading">
            Verified model configuration, training parameters, and benchmark results.
          </p>
        </div>

        <div className="landing-specs">
          <div className="landing-specs__grid">
            {SPECS.map(spec => (
              <div key={spec.label} className="landing-specs__item">
                <span className="landing-specs__label">{spec.label}</span>
                <span className="landing-specs__value">{spec.value}</span>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
