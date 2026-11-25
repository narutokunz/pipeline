import React from 'react'

export default function Navbar() {
  return (
    <header className="sticky top-0 z-50 bg-[#1e2227] text-white shadow-md border-b border-[#2b3036]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between py-3.5 gap-3">
          {/* Prominent Full Project Title */}
          <div>
            <h1 className="text-xl sm:text-2xl font-bold tracking-normal text-[#f4f6f8] leading-snug">
              Quality-Aware Selective Enhancement for Low-Quality Face Recognition
            </h1>
          </div>

          {/* Only One Navigation Tab: Face Recognition */}
          <nav className="flex items-center">
            <span className="px-4 py-1.5 rounded text-base sm:text-lg font-semibold bg-[#2a313a] text-[#52aba7] border-b-2 border-[#52aba7]">
              Face Recognition
            </span>
          </nav>
        </div>
      </div>
    </header>
  )
}
