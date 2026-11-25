import React from 'react'
import Navbar from './components/Navbar.jsx'
import FaceRecognition from './pages/FaceRecognition.jsx'

function App() {
  return (
    <div className="flex flex-col min-h-screen bg-[#f7f8f9] text-[#1a1c20]">
      <Navbar />
      <main className="flex-1 pb-16">
        <FaceRecognition />
      </main>
    </div>
  )
}

export default App
