import React, { useRef, Suspense, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { ScrollControls, useScroll, Scroll, useGLTF, OrbitControls } from '@react-three/drei';
import { motion, AnimatePresence } from 'framer-motion';
import * as THREE from 'three';

// --- ICONS ---
// (Icons are unchanged)
const HeartIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 mr-3 inline-block text-brand-cyan" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M4.318 6.318a4.5 4.5 0 016.364 0L12 7.636l1.318-1.318a4.5 4.5 0 116.364 6.364L12 20.364l-7.682-7.682a4.5 4.5 0 010-6.364z" /></svg>);
const BriefcaseIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 mr-3 inline-block text-brand-cyan" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2m4 6h.01M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" /></svg>);
const CompassIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>;
const ObstacleIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" /></svg>;
const VoiceIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" /></svg>;
const JoystickIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.5L15.232 5.232z" /></svg>;
const PhoneIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" /></svg>;
const ShieldIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.49l-1.955 5.237a3.5 3.5 0 01-4.32 2.625L8.238 14.93a3.5 3.5 0 01-3.56-4.993l3.296-8.24a3.5 3.5 0 014.993-3.56l5.237 1.955a3.5 3.5 0 012.625 4.32z" /></svg>;
const AlertIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>;
const CopyIcon = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 inline-block ml-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>;

// --- Load the actual GLB Model ---
function WheelchairGLTFModel(props) {
  const { scene } = useGLTF('/combat_wheelchair_base_model_spiral_metal.glb');
  const clonedScene = scene.clone();
  return (
    <primitive
      object={clonedScene}
      rotation={[0, 0, 0]} // Start facing directly forward
      position={[0, -0.5, 0]}
      {...props}
    />
  );
}
useGLTF.preload('/combat_wheelchair_base_model_spiral_metal.glb');

// --- MAIN 3D EXPERIENCE ---
function Experience() {
  const scroll = useScroll();
  const wheelchairRef = useRef();
  const groupRef = useRef();

  useFrame((state, delta) => {
    const totalPages = 7;
    const animationEndOffset = 5 / totalPages;

    // --- Opacity / Fade Logic ---
    groupRef.current.visible = scroll.offset < animationEndOffset;
    const groupOpacity = 1 - scroll.range(animationEndOffset - 0.2, 0.2);
    groupRef.current.traverse((child) => {
      if (child.isMesh && child.material) {
        child.material.opacity = groupOpacity;
        child.material.transparent = true;
      }
      // Fade point light intensity
      if (child.isPointLight) {
        // We handle intensity based on scroll separately if needed
        // This generic fade might conflict, let's rely on specific logic
        // child.intensity = child.intensity * groupOpacity; // Optional generic fade
      }
    });

    // --- Path & Rotation Logic ---
    const sectionProgress = Math.min(1, Math.max(0, scroll.offset / animationEndOffset));
    const verticalTravel = 6;
    const numSpirals = 4;
    const radius = 1.8;
    const angleOffset = Math.PI;

    const x = Math.sin(sectionProgress * Math.PI * numSpirals + angleOffset) * radius;
    const z = Math.cos(sectionProgress * Math.PI * numSpirals + angleOffset) * radius;
    const y = -sectionProgress * verticalTravel;

    const page0Vis = scroll.visible(0, 0.8 / totalPages);
    const page1Vis = scroll.visible(0.8 / totalPages, 1 / totalPages);
    const page2Vis = scroll.visible(1.8 / totalPages, 1 / totalPages);
    const page3Vis = scroll.visible(2.8 / totalPages, 1 / totalPages);
    const page4Vis = scroll.visible(3.8 / totalPages, 1 / totalPages);

    let targetRotY;
    const tangentAngle = Math.atan2(
      Math.cos(sectionProgress * Math.PI * numSpirals + angleOffset) * numSpirals * Math.PI,
      -Math.sin(sectionProgress * Math.PI * numSpirals + angleOffset) * numSpirals * Math.PI
    );

    // --- ADJUST VIEW ANGLES HERE ---
    if (page0Vis) {
      targetRotY = tangentAngle; // Follow path at start
    } else if (page1Vis) {
      targetRotY = Math.PI * 0.5; // Turn 90 deg right (towards left card)
    } else if (page2Vis) {
      targetRotY = -Math.PI * 0.5; // Turn 90 deg left (towards right card)
    } else if (page3Vis) {
      targetRotY = Math.PI * 0.5; // Turn 90 deg right again (towards left card)
    } else if (page4Vis) {
      targetRotY = tangentAngle + Math.PI; // Turn around to face backwards along path
    } else {
      // Hold last angle or follow path if between sections
      targetRotY = wheelchairRef.current ? wheelchairRef.current.rotation.y : tangentAngle;
    }


    if (wheelchairRef.current) {
      wheelchairRef.current.position.set(x, y, z);
      // --- ADJUST TURN SPEED HERE (4 is default) ---
      wheelchairRef.current.rotation.y = THREE.MathUtils.damp(
        wheelchairRef.current.rotation.y,
        targetRotY,
        4, // Lower = Slower/Smoother Turn, Higher = Faster/Snappier Turn
        delta
      );
    }
  });


  return (
    <group ref={groupRef}>
      <ambientLight intensity={0.8} />
      <directionalLight position={[10, 10, 5]} intensity={1.5} castShadow />

      {/* Wheelchair Model */}
      <group ref={wheelchairRef} scale={1.0}>
        <WheelchairGLTFModel />
        {/* Basic point light */}
        {/* Note: Intensity animation was removed, set initial intensity if needed */}
        <pointLight color="red" intensity={0} position={[0, 1, 0.5]} distance={10} decay={2} />
      </group>

      {/* <OrbitControls enableZoom={true} enablePan={true} enableRotate={true} /> */}
    </group>
  );
}

// --- REUSABLE COMPONENTS ---
// (Unchanged)
function AnimatedSection({ children }) {
  const ref = useRef(null);
  return (
    <motion.section
      ref={ref}
      className="w-full min-h-screen flex flex-col justify-center p-8 md:p-16"
      initial={{ opacity: 0, y: 50 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.3 }}
      transition={{ duration: 0.8, ease: "easeOut" }}
    >
      {children}
    </motion.section>
  );
}

// (Unchanged)
function FeatureCard({ title, icon, children, alignment = 'left', icons }) {
  const textAlign = alignment === 'right' ? 'text-right' : 'text-left';
  const itemVariants = {
    hidden: { opacity: 0, y: 10 },
    visible: (i) => ({ opacity: 1, y: 0, transition: { delay: i * 0.15, duration: 0.5, }, }),
  };
  return (
    <div className={`w-full h-screen flex items-center p-8 md:p-16 ${alignment === 'right' ? 'justify-end' : 'justify-start'}`}>
      <motion.div
        className={`max-w-md w-full p-8 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-md shadow-lg ${textAlign}`}
        initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.5 }} transition={{ staggerChildren: 0.1 }} >
        <motion.div variants={itemVariants}>{icon}</motion.div>
        <motion.h2 variants={itemVariants} className="text-3xl md:text-5xl font-bold">{title}</motion.h2>
        <motion.p variants={itemVariants} className="text-md md:text-lg mt-4 mb-6">{children}</motion.p>
        {icons && (<motion.div variants={itemVariants} className={`flex gap-4 mt-4 ${alignment === 'right' ? 'justify-end' : 'justify-start'}`}> {icons.map((item, i) => (<motion.div key={i} custom={i} variants={itemVariants} className="bg-white/10 p-3 rounded-lg"> {item} </motion.div>))} </motion.div>)}
      </motion.div>
    </div>
  );
}

// QrCodeModal (Unchanged)
function QrCodeModal({ show, onClose }) {
  return (
    <AnimatePresence>
      {show && (
        <motion.div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose}>
          <motion.div className="bg-white p-6 rounded-lg text-center" initial={{ scale: 0.5 }} animate={{ scale: 1 }} exit={{ scale: 0.5 }} onClick={(e) => e.stopPropagation()}>
            <h2 className="text-2xl font-bold text-brand-dark mb-4">Scan to Donate</h2>
            <img src="/samay-qr.jpg" alt="Donation QR Code" className="w-64 h-64" onError={(e) => { e.target.onerror = null; e.target.src = 'https://placehold.co/256x256/e2e8f0/e2e8f0?text=QR' }} />
            <p className="text-brand-dark mt-4">Thank you for your support!</p>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// SponsorModal (Unchanged)
function SponsorModal({ show, onClose }) {
  const email1 = "nishalpoojary2022@kccemsr.edu.in";
  const email2 = "samaypandey2022@kccemsr.edu.in";
  const [copied1, setCopied1] = useState(false);
  const [copied2, setCopied2] = useState(false);

  const copyToClipboard = (text, setCopied) => {
    const textArea = document.createElement("textarea");
    textArea.value = text; document.body.appendChild(textArea); textArea.select();
    try { document.execCommand('copy'); setCopied(true); setTimeout(() => setCopied(false), 2000); }
    catch (err) { console.error('Failed to copy text: ', err); }
    document.body.removeChild(textArea);
  };

  return (
    <AnimatePresence>
      {show && (
        <motion.div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose}>
          <motion.div className="bg-white p-6 rounded-lg text-left max-w-md w-full" initial={{ scale: 0.5 }} animate={{ scale: 1 }} exit={{ scale: 0.5 }} onClick={(e) => e.stopPropagation()}>
            <h2 className="text-2xl font-bold text-brand-dark mb-4">Contact Sponsorship</h2>
            <p className="text-gray-700 mb-4">Please reach out to our team leads for partnership inquiries:</p>
            <div className="mb-3">
              <p className="font-semibold text-brand-dark break-all">{email1}</p>
              <button onClick={() => copyToClipboard(email1, setCopied1)} className="text-sm text-blue-600 hover:text-blue-800"> {copied1 ? 'Copied!' : <>Copy <CopyIcon /></>} </button>
            </div>
            <div>
              <p className="font-semibold text-brand-dark break-all">{email2}</p>
              <button onClick={() => copyToClipboard(email2, setCopied2)} className="text-sm text-blue-600 hover:text-blue-800"> {copied2 ? 'Copied!' : <>Copy <CopyIcon /></>} </button>
            </div>
            <button onClick={onClose} className="mt-6 w-full bg-gray-200 text-gray-700 font-bold py-2 px-4 rounded hover:bg-gray-300 transition-colors"> Close </button>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// --- HTML OVERLAY ---
function Overlay({ onQrClick, onSponsorClick }) {
  // Re-add useFrame and useScroll for footer visibility
  const scroll = useScroll();
  useFrame(() => {
    const totalPages = 7;
    const footerDiv = document.getElementById('page-footer');
    // Show footer when past the 3D animation section (e.g., after 6 pages out of 7)
    const showFooter = scroll.offset > (totalPages - 1) / totalPages;

    if (footerDiv) {
      footerDiv.style.opacity = showFooter ? '1' : '0';
      footerDiv.style.pointerEvents = showFooter ? 'auto' : 'none'; // Enable links only when visible
    }
  });

  return (
    <Scroll html style={{ width: '100%' }}>
      {/* Container for all scrollable HTML content */}
      <div className="w-full text-[#e6f9ff]">
        {/* Page 0: Intro */}
        <div className="w-full h-screen flex justify-center items-center text-center">
          <div>
            {/* LOGO REMOVED from here */}
            <h1 className="text-4xl md:text-6xl font-bold">Meet SmartNav</h1>
            <p className="text-lg md:text-xl mt-2 text-brand-cyan">The Future of Personal Mobility</p>
          </div>
        </div>

        {/* Page 1: Navigation */}
        <FeatureCard title="Autonomous Navigation" icons={[<CompassIcon />, <ObstacleIcon />]}>
          “Understands your route. Moves safely — by itself.”
        </FeatureCard>

        {/* Page 2: Control */}
        <FeatureCard title="Multi-Mode Control" icons={[<VoiceIcon />, <JoystickIcon />, <PhoneIcon />]} alignment="right">
          “Control your way — voice, joystick, or touch.”
        </FeatureCard>

        {/* Page 3: Safety */}
        <FeatureCard title="Safety & Assistance" icons={[<ShieldIcon />, <AlertIcon />]}>
          “Protects you when it matters most.”
        </FeatureCard>

        {/* Page 4: Outro */}
        <div className="w-full h-screen flex justify-center items-center text-center">
          <div>
            <h1 className="text-4xl md:text-6xl font-bold text-brand-cyan">SmartNav</h1>
            <p className="text-lg md:text-xl mt-2">Intelligent, Adaptive, Safe.</p>
          </div>
        </div>

        {/* PART 2: STATIC CONTENT SECTIONS */}
        <div className="w-full bg-brand-dark relative z-10">
          <AnimatedSection>
            <div className="max-w-5xl mx-auto text-center">
              <h2 className="text-4xl md:text-5xl font-bold text-brand-cyan mb-4">Meet The Team</h2>
              <p className="text-lg md:text-xl mb-12">This project was born from a shared passion for a simple idea: that freedom of movement shouldn't come with a premium price tag. We are a team of final-year engineering students who saw a gap between expensive, high-tech mobility aids and the affordable, everyday solutions that people in our communities need. SmartNav is our answer—a commitment to leveraging technology for accessibility and independence.</p>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-8">{Array.from({ length: 6 }).map((_, i) => (<div key={i} className="text-center"><div className="w-24 h-24 bg-gray-700 rounded-full mx-auto mb-2"></div><h4 className="font-bold">Team Member</h4><p className="text-sm text-gray-400">Role</p></div>))}</div>
            </div>
          </AnimatedSection>

          <AnimatedSection>
            <div className="max-w-4xl mx-auto text-center">
              <h2 className="text-4xl md:text-5xl font-bold text-brand-cyan mb-8">Get Involved</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="bg-gray-800/50 p-8 rounded-lg text-center border border-gray-700 flex flex-col justify-between">
                  <div><h3 className="text-3xl font-bold mb-4 flex items-center justify-center"><BriefcaseIcon /> Partner With Us</h3><p className="mb-6">We are seeking corporate partners who share our vision. For inquiries and sponsorship opportunities, please get in touch with our team.</p></div>
                  <button onClick={onSponsorClick} className="bg-transparent border-2 border-brand-cyan text-brand-cyan font-bold py-3 px-8 rounded-full hover:bg-brand-cyan hover:text-brand-dark transition-colors inline-block mt-4">Contact Sponsorship</button>
                </div>
                <div className="bg-gray-800/50 p-8 rounded-lg text-center border border-gray-700 flex flex-col justify-between">
                  <div><h3 className="text-3xl font-bold mb-4 flex items-center justify-center"><HeartIcon /> Support Our Mission</h3><p className="mb-6">Your individual contribution fuels our research and helps keep the final product affordable for those who need it most.</p></div>
                  <button onClick={onQrClick} className="bg-brand-cyan text-brand-dark font-bold py-3 px-8 rounded-full hover:bg-white transition-colors mt-4">Support with QR Code</button>
                </div>
              </div>
            </div>
          </AnimatedSection>

          {/* FOOTER RE-ADDED HERE */}
          {/* Added id="page-footer" and styles for visibility control */}
          <div id="page-footer" className="text-center p-8 bg-brand-dark relative z-10 opacity-0 transition-opacity duration-500 pointer-events-none">
            <p className="text-gray-400">© 2025 SmartNav Project. All Rights Reserved.</p>
            <p className="text-sm text-gray-300 mt-2">
              3D Model "Combat Wheelchair Base Model - Spiral Metal" by Chakyas K Kiron, licensed under the Unreal Engine License.
            </p>
            <div className="mt-4 space-x-4"><a href="https://github.com/ChampionSamay1644/Smart-Wheelchair" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-brand-cyan">GitHub</a></div>
          </div>

        </div> {/* End of static content section */}
      </div> {/* End of main scroll container */}
    </Scroll>
  );
}

// --- MAIN APP COMPONENT ---
export default function App() {
  const [showQr, setShowQr] = useState(false);
  const [showSponsor, setShowSponsor] = useState(false);

  return (
    <div style={{ width: '100vw', height: '100vh', backgroundColor: '#0a101f', overflow: 'hidden' }}>
      <QrCodeModal show={showQr} onClose={() => setShowQr(false)} />
      <SponsorModal show={showSponsor} onClose={() => setShowSponsor(false)} />

      <Suspense fallback={<div className="text-white fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1.2 z-20">Loading...</div>}>
        <Canvas shadows camera={{ position: [0, 0, 10], fov: 50 }} style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', zIndex: 1 }}>
          <color attach="background" args={['#0a101f']} />
          <ScrollControls pages={7} damping={0.1}>
            <Experience />
            <Overlay
              onQrClick={() => setShowQr(true)}
              onSponsorClick={() => setShowSponsor(true)}
            />
          </ScrollControls>
        </Canvas>
      </Suspense>
    </div>
  );
}

