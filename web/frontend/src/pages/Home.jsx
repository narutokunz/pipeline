import React from 'react'
import { Link } from 'react-router-dom'

export default function Home() {
  const steps = [
    {
      num: '01',
      title: 'Face Detection',
      desc: 'RetinaFace identifies and isolates individual candidate faces from unconstrained crowded scenes.',
      tag: 'RetinaFace',
      color: 'border-blue-500 text-blue-700 bg-blue-50',
    },
    {
      num: '02',
      title: 'Quality Assessment',
      desc: 'SHQS scores each face (0–100) across 6 quality dimensions without heavy deep learning overhead.',
      tag: 'SHQS Engine',
      color: 'border-teal-500 text-teal-700 bg-teal-50',
    },
    {
      num: '03',
      title: 'Selective Routing',
      desc: 'Three-tier classification determines whether a face needs no enhancement, CLAHE, or CodeFormer.',
      tag: 'Adaptive Logic',
      color: 'border-indigo-500 text-indigo-700 bg-indigo-50',
    },
    {
      num: '04',
      title: 'Targeted Enhancement',
      desc: 'Only degraded faces are processed, preventing artifact generation and preserving natural high-quality features.',
      tag: 'CLAHE / CodeFormer',
      color: 'border-amber-500 text-amber-700 bg-amber-50',
    },
    {
      num: '05',
      title: 'ArcFace Recognition',
      desc: '512-dimensional normalized embeddings and cosine similarity establish identity verification against reference.',
      tag: 'InsightFace ArcFace',
      color: 'border-emerald-500 text-emerald-700 bg-emerald-50',
    },
  ]

  const stats = [
    { label: 'Quality Factors', value: '6 Dimensions', detail: 'Sharpness, Resolution, Fidelity, Noise, Contrast, Illumination' },
    { label: 'Decision Thresholds', value: 'NO ≥ 75 | SOFT ≥ 35', detail: 'Selective enhancement logic' },
    { label: 'Embedding Space', value: '512-Dimensional', detail: 'ArcFace L2-normalized feature vectors' },
    { label: 'Zero Duplication', value: '1:1 Mapping', detail: 'Single final enhanced crop per detected candidate' },
  ]

  return (
    <div className="space-y-16 py-8 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Hero Section */}
      <section className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden p-8 sm:p-12 lg:p-16 text-center sm:text-left">
        <div className="max-w-4xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-navy-50 border border-navy-200 text-navy-700 text-xs font-semibold mb-6">
            <span className="w-2 h-2 rounded-full bg-teal-600 animate-pulse"></span>
            Computer Science Final-Year Research Project
          </div>
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold text-navy-800 tracking-tight leading-tight">
            Quality-Aware Selective Enhancement for Low-Quality Face Recognition
          </h1>
          <p className="mt-6 text-lg sm:text-xl text-slate-600 leading-relaxed max-w-3xl">
            An intelligent, multi-stage pipeline designed to mitigate recognition failure in unconstrained and low-resolution environments. By assessing facial image quality upfront using SHQS, our system selectively applies adaptive CLAHE or CodeFormer restoration only where needed — avoiding unnecessary computation and artifact degradation.
          </p>
          <div className="mt-8 flex flex-wrap gap-4 justify-center sm:justify-start">
            <Link
              to="/recognize"
              className="px-6 py-3.5 rounded-lg bg-navy-700 hover:bg-navy-800 text-white font-medium shadow-sm transition-all flex items-center gap-2"
            >
              <span>Launch Recognition Workspace</span>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
              </svg>
            </Link>
            <Link
              to="/methodology"
              className="px-6 py-3.5 rounded-lg bg-white hover:bg-slate-50 text-slate-700 font-medium border border-slate-300 shadow-sm transition-all"
            >
              Explore Methodology
            </Link>
            <Link
              to="/results"
              className="px-6 py-3.5 rounded-lg bg-white hover:bg-slate-50 text-slate-700 font-medium border border-slate-300 shadow-sm transition-all"
            >
              Benchmark Evaluation
            </Link>
          </div>
        </div>
      </section>

      {/* Key Architectural Principles */}
      <section>
        <div className="text-center sm:text-left mb-8">
          <h2 className="text-2xl font-bold text-navy-800 tracking-tight">
            Key Architectural Principles
          </h2>
          <p className="text-slate-500 text-sm mt-1">
            Systematic quality assessment paired with targeted restoration.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {stats.map((stat, i) => (
            <div key={i} className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
              <span className="text-xs font-semibold uppercase tracking-wider text-teal-600 block mb-1">
                {stat.label}
              </span>
              <div className="text-2xl font-bold text-navy-800">{stat.value}</div>
              <p className="text-xs text-slate-500 mt-2">{stat.detail}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Pipeline Stages */}
      <section>
        <div className="text-center sm:text-left mb-8">
          <h2 className="text-2xl font-bold text-navy-800 tracking-tight">
            End-to-End Pipeline Workflow
          </h2>
          <p className="text-slate-500 text-sm mt-1">
            Sequential stages executed synchronously per input photograph.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
          {steps.map((s, idx) => (
            <div key={idx} className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-bold text-slate-400">{s.num}</span>
                  <span className={`text-[11px] font-semibold px-2 py-0.5 rounded border ${s.color}`}>
                    {s.tag}
                  </span>
                </div>
                <h3 className="font-semibold text-slate-800 text-base mb-2">{s.title}</h3>
                <p className="text-xs text-slate-600 leading-relaxed">{s.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Academic Problem Statement Callout */}
      <section className="bg-slate-100 rounded-2xl p-8 border border-slate-200">
        <h3 className="text-lg font-bold text-navy-800 mb-3">
          Why Unconditional Enhancement Fails
        </h3>
        <p className="text-sm text-slate-700 leading-relaxed mb-4">
          Conventional face restoration approaches apply deep generative models (such as CodeFormer or GFPGAN) uniformly to every detected crop in an image. While this reconstructs severely degraded faces, applying deep hallucination networks to already clear or moderately clear faces often alters micro-textures and subtle identity features, leading to <em>identity drift</em> and reduced ArcFace cosine similarity.
        </p>
        <p className="text-sm text-slate-700 leading-relaxed">
          Our proposed pipeline employs the mathematical <strong>SHQS (Selective Heuristic Quality Scoring)</strong> framework to classify faces into three distinct operational tiers (NO, SOFT, STRONG), ensuring optimal trade-offs between enhancement fidelity, processing speed, and biometric recognition accuracy.
        </p>
      </section>
    </div>
  )
}
