import React, { useState, useEffect, useRef } from 'react'

export default function FaceRecognition() {
  // Upload and file states
  const [refPreview, setRefPreview] = useState(null)
  const [refFilename, setRefFilename] = useState(null)
  const [crowdPreview, setCrowdPreview] = useState(null)
  const [crowdFilename, setCrowdFilename] = useState(null)

  const [uploading, setUploading] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [errorMsg, setErrorMsg] = useState(null)

  // Pipeline execution & live progress
  const [taskId, setTaskId] = useState(null)
  const [pipelineEvents, setPipelineEvents] = useState([])
  const [elapsedTime, setElapsedTime] = useState(0)

  // Results
  const [results, setResults] = useState(null)
  const [filterType, setFilterType] = useState('ALL') // ALL, MATCH, NO_MATCH, NO_ENH, CLAHE, CODEFORMER
  const [expandedFaceIds, setExpandedFaceIds] = useState({})

  const timerRef = useRef(null)
  const eventSourceRef = useRef(null)
  const faceCardsRef = useRef({})

  const STORAGE_KEY = 'qaselr_face_recognition_state'

  // Pipeline stages definition
  const STAGES = [
    { key: 'detecting_faces', label: 'Detecting faces (RetinaFace)' },
    { key: 'evaluating_quality', label: 'Evaluating face quality (SHQS Engine)' },
    { key: 'enhancing_faces', label: 'Selective Enhancement (CLAHE / CodeFormer)' },
    { key: 'matching_faces', label: 'Facial embeddings & similarity matching (ArcFace)' },
    { key: 'ranking_results', label: 'Ranking candidate faces by similarity score' },
    { key: 'generating_output', label: 'Generating report & annotated scene' },
  ]

  // Load persistent state on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY)
      if (saved) {
        const parsed = JSON.parse(saved)
        if (parsed.refFilename) setRefFilename(parsed.refFilename)
        if (parsed.refPreview) setRefPreview(parsed.refPreview)
        if (parsed.crowdFilename) setCrowdFilename(parsed.crowdFilename)
        if (parsed.crowdPreview) setCrowdPreview(parsed.crowdPreview)
        if (parsed.taskId) setTaskId(parsed.taskId)
        if (parsed.results) setResults(parsed.results)
      }
    } catch (e) {
      console.error('Failed to load saved session state', e)
    }

    return () => {
      if (eventSourceRef.current) eventSourceRef.current.close()
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [])

  // Save persistent state whenever key items change
  const saveSessionState = (updates = {}) => {
    try {
      const currentState = {
        refFilename: updates.refFilename !== undefined ? updates.refFilename : refFilename,
        refPreview: updates.refPreview !== undefined ? updates.refPreview : refPreview,
        crowdFilename: updates.crowdFilename !== undefined ? updates.crowdFilename : crowdFilename,
        crowdPreview: updates.crowdPreview !== undefined ? updates.crowdPreview : crowdPreview,
        taskId: updates.taskId !== undefined ? updates.taskId : taskId,
        results: updates.results !== undefined ? updates.results : results,
      }
      localStorage.setItem(STORAGE_KEY, JSON.stringify(currentState))
    } catch (e) {
      console.error('Failed to persist session state', e)
    }
  }

  // Handle Reference File Selection
  const handleRefSelect = async (file) => {
    if (!file) return
    setErrorMsg(null)
    const localUrl = URL.createObjectURL(file)
    setRefPreview(localUrl)

    const formData = new FormData()
    formData.append('file', file)
    try {
      setUploading(true)
      const res = await fetch('/api/upload/reference', {
        method: 'POST',
        body: formData,
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Failed to upload reference image')
      }
      const data = await res.json()
      setRefFilename(data.filename)
      saveSessionState({ refFilename: data.filename, refPreview: `/api/images/${data.filename}` })
    } catch (err) {
      setErrorMsg(`Reference upload error: ${err.message}`)
      setRefPreview(null)
      setRefFilename(null)
    } finally {
      setUploading(false)
    }
  }

  // Handle Crowd File Selection
  const handleCrowdSelect = async (file) => {
    if (!file) return
    setErrorMsg(null)
    const localUrl = URL.createObjectURL(file)
    setCrowdPreview(localUrl)

    const formData = new FormData()
    formData.append('file', file)
    try {
      setUploading(true)
      const res = await fetch('/api/upload/crowd', {
        method: 'POST',
        body: formData,
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Failed to upload crowd image')
      }
      const data = await res.json()
      setCrowdFilename(data.filename)
      saveSessionState({ crowdFilename: data.filename, crowdPreview: `/api/images/${data.filename}` })
    } catch (err) {
      setErrorMsg(`Crowd image upload error: ${err.message}`)
      setCrowdPreview(null)
      setCrowdFilename(null)
    } finally {
      setUploading(false)
    }
  }

  // Clear / Reset Session
  const handleClearSession = async () => {
    if (eventSourceRef.current) eventSourceRef.current.close()
    if (timerRef.current) clearInterval(timerRef.current)

    try {
      await fetch('/api/clear', { method: 'POST' })
    } catch (e) {
      // ignore
    }

    localStorage.removeItem(STORAGE_KEY)
    setRefPreview(null)
    setRefFilename(null)
    setCrowdPreview(null)
    setCrowdFilename(null)
    setAnalyzing(false)
    setTaskId(null)
    setPipelineEvents([])
    setResults(null)
    setErrorMsg(null)
    setElapsedTime(0)
  }

  // Start Pipeline Analysis
  const handleStartAnalysis = async () => {
    if (!refFilename || !crowdFilename) {
      setErrorMsg('Please upload both the reference photograph and the crowd image.')
      return
    }

    setErrorMsg(null)
    setAnalyzing(true)
    setPipelineEvents([])
    setElapsedTime(0)

    const startTime = Date.now()
    timerRef.current = setInterval(() => {
      setElapsedTime(((Date.now() - startTime) / 1000).toFixed(1))
    }, 100)

    try {
      const res = await fetch(`/api/analyze?reference_filename=${encodeURIComponent(refFilename)}&crowd_filename=${encodeURIComponent(crowdFilename)}`, {
        method: 'POST',
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Failed to start pipeline analysis')
      }
      const data = await res.json()
      const newTaskId = data.task_id
      setTaskId(newTaskId)

      // Connect SSE
      const evtSource = new EventSource(`/api/status/${newTaskId}`)
      eventSourceRef.current = evtSource

      evtSource.onmessage = async (e) => {
        try {
          const payload = JSON.parse(e.data)
          setPipelineEvents((prev) => [...prev, payload])

          if (payload.stage === 'complete' && payload.results_ready) {
            evtSource.close()
            clearInterval(timerRef.current)
            setAnalyzing(false)

            // Fetch final results
            const resResults = await fetch(`/api/results/${newTaskId}`)
            if (resResults.ok) {
              const resData = await resResults.json()
              setResults(resData)
              saveSessionState({ taskId: newTaskId, results: resData })
            }
          } else if (payload.stage === 'error') {
            evtSource.close()
            clearInterval(timerRef.current)
            setAnalyzing(false)
            setErrorMsg(payload.message || 'Pipeline encountered an unexpected error.')
          }
        } catch (err) {
          console.error('SSE Error', err)
        }
      }

      evtSource.onerror = () => {
        evtSource.close()
      }
    } catch (err) {
      clearInterval(timerRef.current)
      setAnalyzing(false)
      setErrorMsg(err.message)
    }
  }

  // Status helper for progress stage
  const getStageStatus = (stageKey) => {
    const hasComplete = pipelineEvents.some((ev) => ev.stage === stageKey && ev.status === 'complete')
    if (hasComplete) return 'complete'
    const hasActive = pipelineEvents.some((ev) => ev.stage === stageKey && ev.status === 'active')
    if (hasActive) return 'active'
    return 'pending'
  }

  const getStageElapsed = (stageKey) => {
    const ev = pipelineEvents.find((e) => e.stage === stageKey && e.status === 'complete')
    return ev ? `${ev.elapsed}s` : null
  }

  // RANKING: Strict descending sort by Cosine Similarity!
  // Assign explicit Rank 1 to the highest similarity candidate.
  const getRankedFaces = () => {
    if (!results || !results.faces) return []

    // Sort strictly by similarity descending
    const sorted = [...results.faces].sort((a, b) => {
      const simA = a.similarity_score !== null && a.similarity_score !== undefined ? a.similarity_score : -999
      const simB = b.similarity_score !== null && b.similarity_score !== undefined ? b.similarity_score : -999
      return simB - simA
    })

    // Assign sequential Rank 1, 2, 3...
    return sorted.map((face, idx) => ({
      ...face,
      rank: idx + 1,
    }))
  }

  const rankedFaces = getRankedFaces()

  // Best Match (Rank #1)
  const topCandidate = rankedFaces.length > 0 ? rankedFaces[0] : null
  const isConfirmedMatch = topCandidate && (topCandidate.similarity_score >= (results?.thresholds?.MATCH || 0.5))

  // Filtered faces for the grid
  const filteredFaces = rankedFaces.filter((f) => {
    if (filterType === 'MATCH') return f.match_status === 'MATCH'
    if (filterType === 'NO_MATCH') return f.match_status === 'NO MATCH'
    if (filterType === 'NO_ENH') return f.enhancement_method === 'none'
    if (filterType === 'CLAHE') return f.enhancement_method === 'CLAHE'
    if (filterType === 'CODEFORMER') return f.enhancement_method === 'CodeFormer'
    return true
  })

  const toggleExpand = (faceId) => {
    setExpandedFaceIds((prev) => ({ ...prev, [faceId]: !prev[faceId] }))
  }

  const formatDecision = (method) => {
    if (method === 'none') return 'skip'
    if (method === 'CLAHE') return 'enhance (CLAHE)'
    if (method === 'CodeFormer') return 'enhance (CodeFormer)'
    return method
  }

  return (
    <div className="py-8 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto space-y-10 print-page text-[#1a1c20]">
      {/* Top Controls Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#d7dadc] pb-4 no-print">
        <div>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-[#1a1c20]">
            Face Recognition Analysis
          </h2>
          <p className="text-base text-[#545c65] mt-1">
            Compare candidate faces detected in crowd scenes against reference portraits with automated selective quality restoration.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={handleClearSession}
            type="button"
            className="px-5 py-2.5 rounded-md text-base font-bold text-[#424850] bg-white border border-[#b4bac0] hover:bg-[#ebeeef] shadow-sm transition-all"
          >
            Clear / Reset
          </button>
        </div>
      </div>

      {/* Error Banner */}
      {errorMsg && (
        <div className="bg-[#fdf2f2] border-l-4 border-[#b91c1c] p-4 rounded text-[#991b1b] text-base flex items-center justify-between no-print">
          <div>
            <span className="font-bold">Notice:</span> {errorMsg}
          </div>
          <button onClick={() => setErrorMsg(null)} className="text-[#991b1b] hover:text-[#7f1d1d] font-bold text-lg">
            &times;
          </button>
        </div>
      )}

      {/* Upload Cards Section */}
      <section className="space-y-6 no-print">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Reference Image Card */}
          <div className="bg-white rounded-xl p-6 border border-[#d7dadc] shadow-sm flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-bold uppercase tracking-wider text-[#215c5a] bg-[#daf0ee] px-3 py-1 rounded">
                  Reference Photograph
                </span>
                {refPreview && (
                  <button
                    onClick={() => {
                      setRefPreview(null)
                      setRefFilename(null)
                      saveSessionState({ refPreview: null, refFilename: null })
                    }}
                    className="text-sm text-[#b91c1c] hover:underline font-bold"
                  >
                    Remove
                  </button>
                )}
              </div>
              <h3 className="text-xl font-bold text-[#1a1c20] mt-1">Target Reference Image</h3>
              <p className="text-base text-[#545c65] mt-0.5 mb-4">
                Upload a portrait photograph of the target individual to identify.
              </p>

              {refPreview ? (
                <div className="rounded-lg overflow-hidden border border-[#d7dadc] bg-[#f6f7f8] flex items-center justify-center min-h-60 max-h-72">
                  <img
                    src={refPreview}
                    alt="Reference Preview"
                    className="object-contain max-h-72 w-full"
                  />
                </div>
              ) : (
                <label className="flex flex-col items-center justify-center p-10 bg-[#fbfcfd] hover:bg-[#f4f6f8] rounded-lg cursor-pointer border-2 border-dashed border-[#b4bac0] transition-colors">
                  <span className="text-base font-bold text-[#1a1c20]">Click or drag &amp; drop to upload</span>
                  <span className="text-sm text-[#545c65] mt-1">Supports JPG, JPEG, and PNG formats</span>
                  <input
                    type="file"
                    accept="image/jpeg,image/png,image/jpg"
                    className="hidden"
                    onChange={(e) => handleRefSelect(e.target.files[0])}
                  />
                </label>
              )}
            </div>
          </div>

          {/* Crowd Image Card */}
          <div className="bg-white rounded-xl p-6 border border-[#d7dadc] shadow-sm flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-bold uppercase tracking-wider text-[#2b2f35] bg-[#ebeeef] px-3 py-1 rounded">
                  Crowd / Target Scene
                </span>
                {crowdPreview && (
                  <button
                    onClick={() => {
                      setCrowdPreview(null)
                      setCrowdFilename(null)
                      saveSessionState({ crowdPreview: null, crowdFilename: null })
                    }}
                    className="text-sm text-[#b91c1c] hover:underline font-bold"
                  >
                    Remove
                  </button>
                )}
              </div>
              <h3 className="text-xl font-bold text-[#1a1c20] mt-1">Crowd / Group Photograph</h3>
              <p className="text-base text-[#545c65] mt-0.5 mb-4">
                Upload a crowded, group, or low-resolution image containing candidate faces.
              </p>

              {crowdPreview ? (
                <div className="rounded-lg overflow-hidden border border-[#d7dadc] bg-[#f6f7f8] flex items-center justify-center min-h-60 max-h-72">
                  <img
                    src={crowdPreview}
                    alt="Crowd Preview"
                    className="object-contain max-h-72 w-full"
                  />
                </div>
              ) : (
                <label className="flex flex-col items-center justify-center p-10 bg-[#fbfcfd] hover:bg-[#f4f6f8] rounded-lg cursor-pointer border-2 border-dashed border-[#b4bac0] transition-colors">
                  <span className="text-base font-bold text-[#1a1c20]">Click or drag &amp; drop to upload</span>
                  <span className="text-sm text-[#545c65] mt-1">Supports JPG, JPEG, and PNG formats</span>
                  <input
                    type="file"
                    accept="image/jpeg,image/png,image/jpg"
                    className="hidden"
                    onChange={(e) => handleCrowdSelect(e.target.files[0])}
                  />
                </label>
              )}
            </div>
          </div>
        </div>

        {/* Action Button */}
        <div className="text-center pt-2">
          <button
            onClick={handleStartAnalysis}
            disabled={!refFilename || !crowdFilename || analyzing || uploading}
            type="button"
            className={`w-full sm:w-auto px-10 py-4 rounded-lg font-bold text-lg shadow-sm transition-all inline-flex items-center justify-center gap-3 ${
              !refFilename || !crowdFilename || analyzing || uploading
                ? 'bg-[#d7dadc] text-[#8b949d] cursor-not-allowed'
                : 'bg-[#2b2f35] hover:bg-[#1a1c20] text-white'
            }`}
          >
            {analyzing ? (
              <span>Executing Automated Pipeline ({elapsedTime}s)...</span>
            ) : (
              <span>Analyze &amp; Recognize Faces</span>
            )}
          </button>
        </div>
      </section>

      {/* Live Processing Substeps */}
      {(analyzing || (pipelineEvents.length > 0 && !results)) && (
        <section className="bg-white rounded-xl p-6 border border-[#d7dadc] shadow-sm space-y-4 no-print">
          <div className="flex items-center justify-between border-b border-[#ebeeef] pb-3">
            <h3 className="text-xl font-bold text-[#1a1c20]">Live Pipeline Stage Progress</h3>
            <span className="text-base font-bold text-[#215c5a] bg-[#daf0ee] px-3 py-1 rounded">
              Elapsed: {elapsedTime}s
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {STAGES.map((stg) => {
              const status = getStageStatus(stg.key)
              const stageTime = getStageElapsed(stg.key)

              return (
                <div
                  key={stg.key}
                  className={`p-4 rounded-lg border flex items-center justify-between text-base ${
                    status === 'complete'
                      ? 'bg-[#f0f9f8] border-[#86cac6] text-[#183e3c] font-bold'
                      : status === 'active'
                      ? 'bg-[#fbfcfd] border-[#2b2f35] text-[#1a1c20] font-bold shadow-sm'
                      : 'bg-[#f6f7f8] border-[#d7dadc] text-[#8b949d]'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    {status === 'complete' && <span className="text-[#215c5a] text-lg font-bold">&#10003;</span>}
                    {status === 'active' && <span className="text-[#358f8b] font-bold text-lg animate-pulse">&bull;</span>}
                    {status === 'pending' && <span className="text-[#b4bac0] text-lg">&bull;</span>}
                    <span>{stg.label}</span>
                  </div>
                  {stageTime && <span className="text-sm font-normal text-[#545c65]">{stageTime}</span>}
                </div>
              )
            })}
          </div>
        </section>
      )}

      {/* Recognition Results */}
      {results && (
        <section className="space-y-10">
          {/* Executive Summary Dashboard */}
          <div className="bg-white rounded-xl p-6 border border-[#d7dadc] shadow-sm">
            <div className="border-b border-[#ebeeef] pb-3 mb-5">
              <h3 className="text-2xl font-bold text-[#1a1c20]">Recognition Summary</h3>
              <p className="text-base text-[#545c65] mt-0.5">
                Overview of detected candidate faces, enhancement routing, and identification results.
              </p>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 text-center">
              <div className="p-4 bg-[#fbfcfd] rounded-lg border border-[#ebeeef]">
                <span className="text-sm font-bold text-[#545c65] uppercase block">Total Detected</span>
                <span className="text-3xl font-bold text-[#1a1c20] mt-1 block">{results.total_detected}</span>
              </div>
              <div className="p-4 bg-[#fbfcfd] rounded-lg border border-[#ebeeef]">
                <span className="text-sm font-bold text-[#545c65] uppercase block">Accepted Faces</span>
                <span className="text-3xl font-bold text-[#215c5a] mt-1 block">{results.accepted}</span>
              </div>
              <div className="p-4 bg-[#fbfcfd] rounded-lg border border-[#ebeeef]">
                <span className="text-sm font-bold text-[#545c65] uppercase block">No Enhancement</span>
                <span className="text-3xl font-bold text-[#424850] mt-1 block">{results.quality_distribution?.NO || 0}</span>
              </div>
              <div className="p-4 bg-[#fbfcfd] rounded-lg border border-[#ebeeef]">
                <span className="text-sm font-bold text-[#545c65] uppercase block">CLAHE Enhanced</span>
                <span className="text-3xl font-bold text-[#277370] mt-1 block">{results.quality_distribution?.SOFT || 0}</span>
              </div>
              <div className="p-4 bg-[#fbfcfd] rounded-lg border border-[#ebeeef]">
                <span className="text-sm font-bold text-[#545c65] uppercase block">CodeFormer Restored</span>
                <span className="text-3xl font-bold text-[#6c757f] mt-1 block">{results.quality_distribution?.STRONG || 0}</span>
              </div>
              <div className="p-4 bg-[#fbfcfd] rounded-lg border border-[#ebeeef]">
                <span className="text-sm font-bold text-[#545c65] uppercase block">Processing Time</span>
                <span className="text-3xl font-bold text-[#1a1c20] mt-1 block">{results.total_time}s</span>
              </div>
            </div>
          </div>

          {/* Top Match Section (Rank 1 Face) */}
          {topCandidate && (
            <div className="bg-white rounded-xl p-6 sm:p-8 border border-[#d7dadc] shadow-sm space-y-6">
              <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#ebeeef] pb-4">
                <div>
                  <span className="text-sm font-bold uppercase tracking-wider text-[#545c65]">
                    Rank 1 &bull; Highest Similarity Match
                  </span>
                  <h3 className="text-2xl font-bold text-[#1a1c20] mt-1">
                    {isConfirmedMatch ? 'Identity Verified: MATCH' : 'Highest Ranked Candidate (No Match)'}
                  </h3>
                </div>

                <div className={`px-5 py-2 rounded-lg text-lg font-bold border ${
                  isConfirmedMatch
                    ? 'bg-[#daf0ee] text-[#183e3c] border-[#86cac6]'
                    : 'bg-[#fdf2f2] text-[#991b1b] border-[#fca5a5]'
                }`}>
                  {isConfirmedMatch ? 'MATCH CONFIRMED' : 'NO MATCH'}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
                {/* Reference Portrait */}
                <div className="bg-[#fbfcfd] p-4 rounded-lg border border-[#ebeeef] text-center">
                  <span className="text-base font-bold text-[#545c65] block mb-2">Reference Image</span>
                  <div className="h-48 flex items-center justify-center bg-white rounded border border-[#d7dadc] overflow-hidden">
                    <img
                      src={refPreview || `/api/images/${refFilename}`}
                      alt="Reference"
                      className="object-contain max-h-48 max-w-full"
                    />
                  </div>
                </div>

                {/* Similarity Score Card */}
                <div className="text-center space-y-2 px-4">
                  <span className="text-sm font-bold text-[#545c65] uppercase tracking-wider block">
                    Cosine Similarity Score
                  </span>
                  <div className="text-5xl font-bold text-[#1a1c20]">
                    {topCandidate.similarity_score !== null ? topCandidate.similarity_score.toFixed(4) : 'N/A'}
                  </div>
                  <div className="text-base text-[#545c65] pt-2">
                    <span className="font-bold text-[#1a1c20]">Candidate Face ID: #{topCandidate.face_id}</span>
                  </div>
                  <div className="text-sm text-[#6c757f]">
                    SHQS Score: {topCandidate.shqs_score} &bull; Enhancement: {topCandidate.enhancement_method}
                  </div>
                </div>

                {/* Candidate Face Crops (Original and Enhanced) */}
                <div className="bg-[#fbfcfd] p-4 rounded-lg border border-[#ebeeef] text-center">
                  <span className="text-base font-bold text-[#545c65] block mb-2">
                    Rank 1 Face #{topCandidate.face_id}
                  </span>
                  <div className="flex gap-3 justify-center">
                    <div className="h-48 flex-1 flex flex-col items-center justify-center bg-white rounded border border-[#d7dadc] overflow-hidden p-1">
                      <img
                        src={`/api/images/${results.task_id}/${topCandidate.original_crop}`}
                        alt="Original"
                        className="object-contain max-h-40 max-w-full"
                      />
                      <span className="text-xs text-[#545c65] mt-1 font-bold">Original Crop</span>
                    </div>
                    {topCandidate.enhanced_crop && (
                      <div className="h-48 flex-1 flex flex-col items-center justify-center bg-white rounded border border-[#d7dadc] overflow-hidden p-1">
                        <img
                          src={`/api/images/${results.task_id}/${topCandidate.enhanced_crop}`}
                          alt="Enhanced"
                          className="object-contain max-h-40 max-w-full"
                        />
                        <span className="text-xs text-[#215c5a] mt-1 font-bold">
                          Enhanced ({topCandidate.enhancement_method})
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}


          {/* Annotated Crowd Photograph */}
          <div className="bg-white rounded-xl p-6 border border-[#d7dadc] shadow-sm space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#ebeeef] pb-3">
              <div>
                <h3 className="text-2xl font-bold text-[#1a1c20]">Annotated Crowd Photograph</h3>
                <p className="text-base text-[#545c65] mt-0.5">
                  Visual localization of detected faces in the original search scene.
                </p>
              </div>
              <div className="flex items-center gap-6 text-sm font-bold">
                <span className="flex items-center gap-2">
                  <span className="w-3.5 h-3.5 rounded-sm bg-[#16a34a] inline-block"></span>
                  <span>Match (Green)</span>
                </span>
                <span className="flex items-center gap-2">
                  <span className="w-3.5 h-3.5 rounded-sm bg-[#d97706] inline-block"></span>
                  <span>No Match (Orange)</span>
                </span>
              </div>
            </div>

            <div className="rounded-lg overflow-hidden border border-[#d7dadc] bg-[#111215] flex justify-center">
              <img
                src={`/api/results/${results.task_id}/annotated`}
                alt="Annotated Crowd Result"
                className="max-h-[600px] w-auto object-contain"
              />
            </div>
          </div>

          {/* Detailed Ranked Face Results (Card Grid) */}
          <div className="space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#d7dadc] pb-3">
              <div>
                <h3 className="text-2xl font-bold text-[#1a1c20]">Detailed Candidate Face Analysis</h3>
                <p className="text-base text-[#545c65] mt-0.5">
                  Individual crops displayed in descending order of ArcFace similarity scores.
                </p>
              </div>

              {/* Filter Tabs */}
              <div className="flex flex-wrap gap-2 text-sm font-bold no-print">
                {[
                  { key: 'ALL', label: 'All Candidates' },
                  { key: 'MATCH', label: 'Matches Only' },
                  { key: 'NO_MATCH', label: 'Non-Matches' },
                  { key: 'NO_ENH', label: 'No Enhancement' },
                  { key: 'CLAHE', label: 'CLAHE' },
                  { key: 'CODEFORMER', label: 'CodeFormer' },
                ].map((f) => (
                  <button
                    key={f.key}
                    onClick={() => setFilterType(f.key)}
                    className={`px-3.5 py-1.5 rounded border transition-all ${
                      filterType === f.key
                        ? 'bg-[#2b2f35] text-white border-[#2b2f35]'
                        : 'bg-white text-[#424850] border-[#d7dadc] hover:bg-[#ebeeef]'
                    }`}
                  >
                    {f.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredFaces.map((face) => {
                const isExpanded = !!expandedFaceIds[face.face_id]
                const qf = face.quality_factors || {}

                return (
                  <div
                    key={face.face_id}
                    ref={(el) => (faceCardsRef.current[face.face_id] = el)}
                    className="bg-white rounded-xl border border-[#d7dadc] shadow-sm p-6 space-y-4"
                  >
                    {/* Header */}
                    <div className="flex items-center justify-between border-b border-[#ebeeef] pb-2">
                      <span className="text-base font-bold text-[#1a1c20]">
                        Rank #{face.rank} &bull; Face ID: #{face.face_id}
                      </span>
                      <span
                        className={`text-sm font-bold px-2.5 py-0.5 rounded border ${
                          face.match_status === 'MATCH'
                            ? 'bg-[#daf0ee] text-[#183e3c] border-[#86cac6]'
                            : 'bg-[#f6f7f8] text-[#545c65] border-[#d7dadc]'
                        }`}
                      >
                        {face.match_status === 'MATCH' ? 'Match' : 'No Match'}
                      </span>
                    </div>

                    {/* Previews: Original and Enhanced */}
                    <div className="flex gap-3 justify-center items-center bg-[#fbfcfd] p-3 rounded-lg border border-[#ebeeef]">
                      <div className="flex-1 text-center">
                        <div className="h-32 flex items-center justify-center overflow-hidden rounded bg-white border border-[#d7dadc]">
                          <img
                            src={`/api/images/${results.task_id}/${face.original_crop}`}
                            alt={`Face ${face.face_id} Original`}
                            className="max-h-32 max-w-full object-contain"
                          />
                        </div>
                        <span className="text-xs text-[#545c65] font-bold mt-1 block">Original</span>
                      </div>

                      {face.enhanced_crop && (
                        <div className="flex-1 text-center">
                          <div className="h-32 flex items-center justify-center overflow-hidden rounded bg-white border border-[#d7dadc]">
                            <img
                              src={`/api/images/${results.task_id}/${face.enhanced_crop}`}
                              alt={`Face ${face.face_id} Enhanced`}
                              className="max-h-32 max-w-full object-contain"
                            />
                          </div>
                          <span className="text-xs text-[#215c5a] font-bold mt-1 block">{face.enhancement_method}</span>
                        </div>
                      )}
                    </div>

                    {/* Key Metrics */}
                    <div className="space-y-2 text-base">
                      <div className="flex justify-between py-1 border-b border-[#ebeeef]">
                        <span className="text-[#545c65]">Cosine Similarity:</span>
                        <span className="font-bold text-[#1a1c20]">
                          {face.similarity_score !== null ? face.similarity_score.toFixed(4) : 'N/A'}
                        </span>
                      </div>
                      <div className="flex justify-between py-1 border-b border-[#ebeeef]">
                        <span className="text-[#545c65]">SHQS Score:</span>
                        <span className="font-bold text-[#1a1c20]">{face.shqs_score}</span>
                      </div>
                      <div className="flex justify-between py-1 border-b border-[#ebeeef]">
                        <span className="text-[#545c65]">Enhancement:</span>
                        <span className="font-bold text-[#215c5a]">
                          {face.enhancement_method === 'none' ? 'Not Enhanced' : `Enhanced (${face.enhancement_method})`}
                        </span>
                      </div>
                    </div>

                    {/* Expand Button */}
                    <button
                      onClick={() => toggleExpand(face.face_id)}
                      type="button"
                      className="w-full py-2 rounded bg-[#f6f7f8] hover:bg-[#ebeeef] text-[#424850] font-bold text-sm border border-[#d7dadc] transition-colors no-print"
                    >
                      {isExpanded ? 'Hide Quality Factors' : 'View SHQS Quality Breakdown'}
                    </button>

                    {/* Expandable Breakdown */}
                    {isExpanded && (
                      <div className="pt-3 border-t border-[#ebeeef] space-y-2 text-sm">
                        {[
                          { key: 'S', label: 'Sharpness', val: qf.S },
                          { key: 'R', label: 'Resolution', val: qf.R },
                          { key: 'F', label: 'Fidelity', val: qf.F },
                          { key: 'N', label: 'Noise (Low)', val: qf.N },
                          { key: 'C', label: 'Contrast', val: qf.C },
                          { key: 'L', label: 'Illumination', val: qf.L },
                        ].map((factor) => {
                          const v = factor.val !== undefined ? factor.val : 0
                          return (
                            <div key={factor.key} className="space-y-1">
                              <div className="flex justify-between text-[#545c65]">
                                <span>{factor.label}</span>
                                <span className="font-bold text-[#1a1c20]">{v.toFixed(3)}</span>
                              </div>
                              <div className="w-full bg-[#ebeeef] rounded h-1.5">
                                <div
                                  className="h-1.5 rounded bg-[#277370]"
                                  style={{ width: `${Math.min(100, v * 100)}%` }}
                                ></div>
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>

          {/* Export Artifacts */}
          <div className="bg-white rounded-xl p-6 border border-[#d7dadc] shadow-sm space-y-3 no-print">
            <h3 className="text-xl font-bold text-[#1a1c20]">Export Analysis Report</h3>
            <p className="text-base text-[#545c65]">
              Save or download the recognition results for documentation and academic review.
            </p>

            <div className="flex flex-wrap gap-4 pt-2">
              <button
                onClick={() => window.print()}
                type="button"
                className="px-6 py-3 rounded-md bg-[#2b2f35] hover:bg-[#1a1c20] text-white font-bold text-base shadow-sm"
              >
                Print / Save as PDF Report
              </button>

              <a
                href={`/api/results/${results.task_id}/download/csv`}
                download
                className="px-6 py-3 rounded-md bg-white hover:bg-[#f6f7f8] text-[#1a1c20] font-bold text-base border border-[#b4bac0] shadow-sm"
              >
                Download Results (CSV)
              </a>

              <a
                href={`/api/results/${results.task_id}/download/json`}
                download
                className="px-6 py-3 rounded-md bg-white hover:bg-[#f6f7f8] text-[#1a1c20] font-bold text-base border border-[#b4bac0] shadow-sm"
              >
                Download Results (JSON)
              </a>
            </div>
          </div>
        </section>
      )}
    </div>
  )
}
