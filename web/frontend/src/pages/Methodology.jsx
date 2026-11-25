import React from 'react'

export default function Methodology() {
  const stages = [
    {
      title: 'Stage 1: RetinaFace Face Detection & Alignment',
      summary: 'Deep multi-task face detection architecture with ResNet50 backbone.',
      badge: 'RetinaFace',
      badgeColor: 'bg-blue-100 text-blue-800 border-blue-200',
      points: [
        'Raw image evaluated with detection confidence cutoff (score ≥ 0.8 for acceptance).',
        '10% spatial context margin expanded around each bounding box to retain facial boundary context.',
        'Low-confidence detections (score < 0.8) isolated into separate inspection buffer without dropping.',
        'Extracts 5 facial landmarks (eyes, nose, mouth corners) relative to individual crop boundaries.',
      ],
    },
    {
      title: 'Stage 2: SHQS Facial Quality Assessment Engine',
      summary: 'Deterministic fuzzy membership mathematical formulation (S × R × F × N × C × L).',
      badge: 'SHQS v5',
      badgeColor: 'bg-teal-100 text-teal-800 border-teal-200',
      points: [
        'S (Sharpness): High-frequency bandpass filter response compensated for noise via Immerkaer Laplacian variance.',
        'R (Resolution): Continuous ramp evaluated over native crop pixel bounding dimension (bad: 10px, good: 48px).',
        'F (Fidelity): Validation via OpenCV Haar cascaded frontal and profile detector sweeps.',
        'N (Noise): Fast Immerkaer 3×3 convolution operator estimating standard deviation of noise.',
        'C (Contrast): 2nd to 98th percentile pixel luminosity range with pixel clipping penalization.',
        'L (Illumination): Absolute deviation of mean facial luminance from ideal mid-grey (128).',
        'Calculated score scale: [0, 100]. Classifies into NO (≥ 75.0), SOFT (35.0 – 74.9), and STRONG (< 35.0).',
      ],
    },
    {
      title: 'Stage 3: Quality-Aware Selective Enhancement',
      summary: 'Strict zero-duplication selective routing preventing over-enhancement artifacts.',
      badge: 'Selective Enhancement',
      badgeColor: 'bg-amber-100 text-amber-800 border-amber-200',
      points: [
        'NO Class (SHQS ≥ 75.0): Passed directly through without modification. Original high-frequency details preserved.',
        'SOFT Class (35.0 ≤ SHQS < 75.0): Enhanced via adaptive CLAHE on the L-channel of CIELAB color space.',
        'STRONG Class (SHQS < 35.0): Passed to CodeFormer transformer restoration (fidelity weight w=0.8, padded context).',
        'Strict Verification: Exactly one output crop generated per detected accepted candidate face (zero duplication invariant).',
      ],
    },
    {
      title: 'Stage 4: ArcFace Deep Embedding & Cosine Similarity',
      summary: 'InsightFace Buffalo_L recognition backbone yielding 512-dimensional representations.',
      badge: 'ArcFace 512-D',
      badgeColor: 'bg-emerald-100 text-emerald-800 border-emerald-200',
      points: [
        'Input face crop resized to standard 112×112 resolution for ArcFace deep feature extraction.',
        'Feature embeddings subjected to L2-normalization ensuring vector length ||x|| = 1.',
        'Verification: Dot product computes direct Cosine Similarity against the reference image embedding.',
        'Decision boundary: Default matching threshold set at 0.50 (configurable in research runs).',
      ],
    },
    {
      title: 'Stage 5: Dual-Attribute Priority Ranking',
      summary: 'Hierarchical sorting combining biometric similarity and image quality confidence.',
      badge: 'Ranking Engine',
      badgeColor: 'bg-purple-100 text-purple-800 border-purple-200',
      points: [
        'Primary sort key: ArcFace cosine similarity score (descending).',
        'Secondary sort key: SHQS facial quality score (descending) to break similarity ties.',
        'Full JSON structured report, CSV tabular results, and bounding-box visual annotations generated.',
      ],
    },
  ]

  return (
    <div className="py-8 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto space-y-12">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-extrabold text-navy-800 tracking-tight">
          System Architecture & Methodology
        </h1>
        <p className="mt-2 text-slate-600 text-base max-w-3xl">
          Detailed mathematical formulation and operational flow of the Quality-Aware Selective Enhancement for Low-Quality Face Recognition pipeline.
        </p>
      </div>

      {/* Visual Pipeline Flowchart */}
      <div className="bg-white rounded-2xl p-6 sm:p-8 border border-slate-200 shadow-sm">
        <h2 className="text-lg font-bold text-navy-800 mb-6">
          High-Level Pipeline Flowchart
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-7 gap-2 items-center text-center">
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-xs font-semibold text-slate-700">
            Input Image<br/>(Crowded Scene)
          </div>
          <div className="text-slate-400 font-bold hidden md:block">→</div>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-xs font-semibold text-blue-900">
            RetinaFace<br/>(Detection & Crop)
          </div>
          <div className="text-slate-400 font-bold hidden md:block">→</div>

          <div className="bg-teal-50 border border-teal-200 rounded-lg p-3 text-xs font-semibold text-teal-900">
            SHQS Engine<br/>(0–100 Scoring)
          </div>
          <div className="text-slate-400 font-bold hidden md:block">→</div>

          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-xs font-semibold text-amber-900">
            Selective Route<br/>(NO / CLAHE / CodeFormer)
          </div>
        </div>

        <div className="my-4 border-t border-slate-100"></div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-2 items-center text-center">
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-xs font-semibold text-amber-900">
            Merged Single Crop<br/>(Zero Duplicates)
          </div>
          <div className="text-slate-400 font-bold hidden md:block">→</div>

          <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-3 text-xs font-semibold text-emerald-900">
            ArcFace 512-D<br/>(Normalized Embeddings)
          </div>
          <div className="text-slate-400 font-bold hidden md:block">→</div>

          <div className="bg-purple-50 border border-purple-200 rounded-lg p-3 text-xs font-semibold text-purple-900">
            Cosine Sim & Ranking<br/>(Threshold ≥ 0.5)
          </div>
        </div>
      </div>

      {/* Detailed Stages */}
      <div className="space-y-6">
        {stages.map((stage, idx) => (
          <div key={idx} className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
              <h2 className="text-lg font-bold text-navy-800">{stage.title}</h2>
              <span className={`text-xs font-semibold px-2.5 py-1 rounded-full border ${stage.badgeColor}`}>
                {stage.badge}
              </span>
            </div>
            <p className="text-sm font-medium text-slate-600 mb-4">{stage.summary}</p>
            <ul className="space-y-2">
              {stage.points.map((pt, pIdx) => (
                <li key={pIdx} className="text-xs text-slate-700 flex items-start gap-2">
                  <span className="text-teal-600 font-bold">•</span>
                  <span>{pt}</span>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      {/* SHQS Formula Deep Dive */}
      <div className="bg-navy-900 text-white rounded-2xl p-6 sm:p-8 shadow-sm">
        <h2 className="text-xl font-bold mb-3 text-white">Mathematical Formulation of SHQS</h2>
        <div className="bg-navy-800/80 p-4 rounded-lg font-mono text-sm sm:text-base text-teal-300 border border-navy-700 my-4 text-center">
          SHQS = 100 × S(x) × R(x) × F(x) × N(x) × C(x) × L(x)
        </div>
        <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
          Where each dimension is normalized into a fuzzy continuous membership interval [0.0, 1.0] using empirical logistic ramps. The product formulation guarantees that if any single essential criterion (e.g. extreme blur or non-facial region) drops near zero, the composite score reflects heavy degradation without arbitrary hand-tuned additive weights.
        </p>
      </div>
    </div>
  )
}
