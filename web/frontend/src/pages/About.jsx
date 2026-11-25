import React from 'react'

export default function About() {
  const techStack = [
    { category: 'Face Detection', name: 'RetinaFace', detail: 'ResNet50 backbone, landmark localization & confidence filtering' },
    { category: 'Quality Assessment', name: 'SHQS Engine (v5)', detail: 'Deterministic 6-factor fuzzy membership quality scoring' },
    { category: 'Moderate Enhancement', name: 'CLAHE (LAB-L)', detail: 'Contrast-limited adaptive histogram equalization on luminance' },
    { category: 'Severe Restoration', name: 'CodeFormer', detail: 'Transformer-based discrete codebook prior restoration (w=0.8)' },
    { category: 'Biometric Recognition', name: 'ArcFace (InsightFace)', detail: 'Buffalo_L model producing 512-dimensional normalized embeddings' },
    { category: 'Application Backend', name: 'FastAPI & Python 3.13', detail: 'Asynchronous event streaming (SSE), OpenCV, and REST services' },
    { category: 'User Interface', name: 'React 18 & Tailwind CSS', detail: 'Responsive academic dashboard with real-time pipeline monitoring' },
  ]

  return (
    <div className="py-8 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto space-y-12">
      {/* Title Card */}
      <div className="bg-white rounded-2xl p-8 border border-slate-200 shadow-sm">
        <span className="text-xs font-semibold uppercase tracking-wider text-teal-600 block mb-2">
          Final-Year Undergraduate Research Project
        </span>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-navy-800 tracking-tight">
          Quality-Aware Selective Enhancement for Low-Quality Face Recognition
        </h1>
        <p className="mt-3 text-slate-500 text-sm font-medium">
          Department of Computer Science &bull; Final-Year Capstone Project &bull; Academic Year 2026
        </p>
      </div>

      {/* Abstract */}
      <div className="bg-white rounded-2xl p-8 border border-slate-200 shadow-sm space-y-4">
        <h2 className="text-xl font-bold text-navy-800">Abstract</h2>
        <p className="text-slate-700 text-sm leading-relaxed">
          Unconstrained face recognition in crowded surveillance, drone imagery, and low-resolution video feeds presents profound challenges due to extreme blur, sub-optimal illumination, compression artifacts, and off-angle poses. Traditional enhancement pipelines indiscriminately subject every detected face to heavy generative restoration models, introducing hallucinated micro-features that alter biometric identity vectors and waste significant computation.
        </p>
        <p className="text-slate-700 text-sm leading-relaxed">
          In this research, we propose a modular, quality-aware framework that couples a fast, deterministic Selective Heuristic Quality Scoring (SHQS) engine with an adaptive enhancement router. Faces are categorized into high-quality (NO enhancement), moderately degraded (CLAHE contrast equalization), and severely degraded (CodeFormer codebook restoration) tiers. Feature embeddings generated via ArcFace demonstrate improved verification margins while drastically reducing overall inference latency compared to unconditional deep restoration pipelines.
        </p>
      </div>

      {/* Tech Stack */}
      <div className="bg-white rounded-2xl p-8 border border-slate-200 shadow-sm">
        <h2 className="text-xl font-bold text-navy-800 mb-6">Technical Architecture Stack</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {techStack.map((tech, i) => (
            <div key={i} className="p-4 rounded-xl bg-slate-50 border border-slate-200">
              <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block mb-1">
                {tech.category}
              </span>
              <h3 className="font-bold text-navy-800 text-base">{tech.name}</h3>
              <p className="text-xs text-slate-600 mt-1">{tech.detail}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Privacy, Ethics & Research Governance */}
      <div className="bg-slate-100 rounded-2xl p-8 border border-slate-200 space-y-3">
        <h2 className="text-lg font-bold text-navy-800 flex items-center gap-2">
          <svg className="w-5 h-5 text-teal-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
          </svg>
          Research Governance & Biometric Privacy Policy
        </h2>
        <p className="text-xs text-slate-700 leading-relaxed">
          This software application is engineered strictly for authorized academic evaluation, benchmark reproduction, and controlled consent-based experimentation. No persistent biometric templates or extracted feature vectors are stored without user permission.
        </p>
        <p className="text-xs text-slate-700 leading-relaxed">
          The application session memory may be purged completely at any time by utilizing the <strong>Clear Session</strong> control in the recognition interface, which securely unlinks all temporary image crops, embeddings, and intermediate cache structures from server storage.
        </p>
      </div>
    </div>
  )
}
