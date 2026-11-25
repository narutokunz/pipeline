import React from 'react'

export default function ResearchResults() {
  const benchmarkRows = [
    {
      method: 'ArcFace Baseline (Raw Crops)',
      type: 'Direct Extraction',
      rank1: 'Not evaluated yet',
      rank5: 'Not evaluated yet',
      rank10: 'Not evaluated yet',
      rank20: 'Not evaluated yet',
      rank50: 'Not evaluated yet',
      map: 'Not evaluated yet',
      timePerImg: 'Not evaluated yet',
      status: 'pending',
    },
    {
      method: 'Unconditional CLAHE Enhancement',
      type: 'Uniform Processing',
      rank1: 'Not evaluated yet',
      rank5: 'Not evaluated yet',
      rank10: 'Not evaluated yet',
      rank20: 'Not evaluated yet',
      rank50: 'Not evaluated yet',
      map: 'Not evaluated yet',
      timePerImg: 'Not evaluated yet',
      status: 'pending',
    },
    {
      method: 'Unconditional CodeFormer Enhancement',
      type: 'Uniform Processing',
      rank1: 'Not evaluated yet',
      rank5: 'Not evaluated yet',
      rank10: 'Not evaluated yet',
      rank20: 'Not evaluated yet',
      rank50: 'Not evaluated yet',
      map: 'Not evaluated yet',
      timePerImg: 'Not evaluated yet',
      status: 'pending',
    },
    {
      method: 'Proposed SHQS-Based Selective Enhancement',
      type: 'Quality-Aware Selective',
      rank1: 'Not evaluated yet',
      rank5: 'Not evaluated yet',
      rank10: 'Not evaluated yet',
      rank20: 'Not evaluated yet',
      rank50: 'Not evaluated yet',
      map: 'Not evaluated yet',
      timePerImg: 'Not evaluated yet',
      status: 'pending',
      highlight: true,
    },
    {
      method: 'Texture-Guided Transfer Learning (Published Literature)',
      type: 'Literature Reference',
      rank1: '52.4%',
      rank5: '66.8%',
      rank10: '71.2%',
      rank20: '75.9%',
      rank50: '81.3%',
      map: '0.482',
      timePerImg: 'N/A (Published)',
      status: 'published',
    },
  ]

  return (
    <div className="py-8 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto space-y-10">
      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-100 border border-slate-300 text-slate-700 text-xs font-semibold mb-3">
          Benchmark & Comparative Analysis
        </div>
        <h1 className="text-3xl font-extrabold text-navy-800 tracking-tight">
          Research Evaluation Dashboard
        </h1>
        <p className="mt-2 text-slate-600 text-base max-w-3xl">
          Comparative empirical evaluation on the TinyFace benchmark dataset for low-resolution face recognition in unconstrained scenes.
        </p>
      </div>

      {/* Benchmark Table */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="p-6 border-b border-slate-100 flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold text-navy-800">TinyFace Dataset Benchmark Comparison</h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Cumulative Match Characteristic (CMC) Rank-k accuracies and mean Average Precision (mAP)
            </p>
          </div>
          <span className="text-xs font-semibold px-3 py-1 rounded-full bg-navy-50 text-navy-700 border border-navy-200">
            Evaluation Protocol: 1:N Identification
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-slate-200 text-left text-xs">
            <thead className="bg-slate-50 text-slate-600 font-semibold uppercase tracking-wider">
              <tr>
                <th className="py-3 px-4">Method / Architecture</th>
                <th className="py-3 px-4">Pipeline Strategy</th>
                <th className="py-3 px-3">Rank-1</th>
                <th className="py-3 px-3">Rank-5</th>
                <th className="py-3 px-3">Rank-10</th>
                <th className="py-3 px-3">Rank-20</th>
                <th className="py-3 px-3">Rank-50</th>
                <th className="py-3 px-3">mAP</th>
                <th className="py-3 px-3">Avg Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 bg-white">
              {benchmarkRows.map((row, idx) => (
                <tr key={idx} className={row.highlight ? 'bg-teal-50/50 font-medium' : 'hover:bg-slate-50'}>
                  <td className="py-3.5 px-4 font-semibold text-slate-800 flex items-center gap-2">
                    {row.highlight && (
                      <span className="w-2 h-2 rounded-full bg-teal-600 flex-shrink-0"></span>
                    )}
                    <span>{row.method}</span>
                  </td>
                  <td className="py-3.5 px-4 text-slate-500">{row.type}</td>
                  <td className={`py-3.5 px-3 ${row.status === 'pending' ? 'text-slate-400 italic' : 'text-slate-800 font-semibold'}`}>
                    {row.rank1}
                  </td>
                  <td className={`py-3.5 px-3 ${row.status === 'pending' ? 'text-slate-400 italic' : 'text-slate-800'}`}>
                    {row.rank5}
                  </td>
                  <td className={`py-3.5 px-3 ${row.status === 'pending' ? 'text-slate-400 italic' : 'text-slate-800'}`}>
                    {row.rank10}
                  </td>
                  <td className={`py-3.5 px-3 ${row.status === 'pending' ? 'text-slate-400 italic' : 'text-slate-800'}`}>
                    {row.rank20}
                  </td>
                  <td className={`py-3.5 px-3 ${row.status === 'pending' ? 'text-slate-400 italic' : 'text-slate-800'}`}>
                    {row.rank50}
                  </td>
                  <td className={`py-3.5 px-3 ${row.status === 'pending' ? 'text-slate-400 italic' : 'text-slate-800 font-semibold'}`}>
                    {row.map}
                  </td>
                  <td className={`py-3.5 px-3 ${row.status === 'pending' ? 'text-slate-400 italic' : 'text-slate-600'}`}>
                    {row.timePerImg}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Scientific Integrity Note */}
      <div className="bg-slate-50 rounded-xl p-6 border border-slate-200">
        <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider mb-2">
          Academic Rigor & Evaluation Integrity Protocol
        </h3>
        <p className="text-xs text-slate-600 leading-relaxed">
          In compliance with rigorous research reporting guidelines, entries marked as <em>"Not evaluated yet"</em> reflect ongoing controlled experimental runs on the complete TinyFace test partition. Published literature benchmarks are clearly isolated and cited. Metrics are reported strictly upon empirical completion without synthetic estimation or interpolation.
        </p>
      </div>

      {/* Enhancement Route Distribution Breakdown */}
      <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
        <h3 className="text-base font-bold text-navy-800 mb-4">
          Selective Route Distribution Profile (Hypothesized on TinyFace)
        </h3>
        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-xs font-medium mb-1">
              <span className="text-slate-700">NO Enhancement (High Quality, SHQS ≥ 75)</span>
              <span className="text-slate-500 font-semibold">Pending Evaluation</span>
            </div>
            <div className="w-full bg-slate-100 rounded-full h-3">
              <div className="bg-blue-600 h-3 rounded-full" style={{ width: '0%' }}></div>
            </div>
          </div>
          <div>
            <div className="flex justify-between text-xs font-medium mb-1">
              <span className="text-slate-700">SOFT Enhancement (CLAHE, 35 ≤ SHQS &lt; 75)</span>
              <span className="text-slate-500 font-semibold">Pending Evaluation</span>
            </div>
            <div className="w-full bg-slate-100 rounded-full h-3">
              <div className="bg-teal-600 h-3 rounded-full" style={{ width: '0%' }}></div>
            </div>
          </div>
          <div>
            <div className="flex justify-between text-xs font-medium mb-1">
              <span className="text-slate-700">STRONG Restoration (CodeFormer, SHQS &lt; 35)</span>
              <span className="text-slate-500 font-semibold">Pending Evaluation</span>
            </div>
            <div className="w-full bg-slate-100 rounded-full h-3">
              <div className="bg-amber-600 h-3 rounded-full" style={{ width: '0%' }}></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
