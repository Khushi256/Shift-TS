import Disclosure from '../shared/Disclosure';
import './PillarSection.css';

export default function PillarSection({ plainTitle, technicalTitle, layExplainer, technicalNote, index }) {
  return (
    <div className="pillar">
      <div className="pillar__index" aria-hidden="true">{String(index).padStart(2, '0')}</div>
      <div className="pillar__body">
        <h3 className="pillar__plain-title">{plainTitle}</h3>
        <p className="pillar__technical-title">{technicalTitle}</p>
        <p className="pillar__lay">{layExplainer}</p>
        <Disclosure summary="Technical note">
          <p style={{ margin: 0 }}>{technicalNote}</p>
        </Disclosure>
      </div>
    </div>
  );
}
