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
      rotation={[0, -Math.PI / 2, 0]}// Start facing directly forward
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

  // ----- Global tunables (edit these to change feel) -----
  const totalPages = 7;         // must match <ScrollControls pages={7}>
  const animationPages = 5;     // how many pages the spiral occupies (0..totalPages)
  const turns = 3;              // full 360° turns across the animationPages
  const radiusStart = 3.0;      // spiral radius at t=0
  const radiusEnd = 0.4;        // spiral radius at t=1 (tighten)
  const verticalTravel = 6.0;   // how far the model moves down through the sequence
  const angleOffset = Math.PI;  // rotate entire spiral if you want a different starting heading

  const posDamping = 6.0;       // position damping (higher => snappier)
  const rotDamping = 4.0;       // rotation damping (higher => snappier)

  // Composition poses (where the wheelchair should be for intro & feature cards)
  const POSES = {
    intro: { x: 0.0, z: 3.0, y: 0.0 },   // Page 0 center: behind headline, centered
    feature1: { x: -1.6, z: 2.0, y: -1.2 },   // Page 1: left card
    feature2: { x: 1.6, z: 2.0, y: -1.2 },   // Page 2: right card
    feature3: { x: -1.6, z: 2.0, y: -2.6 }    // Page 3: left card further down
  };

  // internal smooth state so we avoid abrupt jumps on initial render
  const displayed = useRef({ x: 0, y: 0, z: 0, rotY: 0, t: 0 });

  useFrame((state, delta) => {
    // ---------- TUNABLES ----------
    const totalPages = 7;
    const animationPages = 5;
    const turns = 3;
    const radiusStart = 3.0;
    const radiusEnd = 0.4;
    const verticalTravel = 6.0;
    const angleOffset = Math.PI;
    const posDamping = 6.0;
    const liftAboveCard = 0.6;
    const nearCenterWindow = 0.09;
    const spinSpeed = 1.0;
    const rotDamping = 4.0;

    // safety
    if (!scroll || !groupRef.current || !wheelchairRef.current) return;

    // ---------- NOTE: Do NOT toggle groupRef.visible here ----------
    // If you want fade, keep opacity changes only (don't hide the whole group).
    const animationEndOffset = animationPages / totalPages;
    const groupOpacity = 1 - scroll.range(Math.max(0, animationEndOffset - 0.22), 0.2);
    groupRef.current.traverse((child) => {
      if (child.isMesh && child.material) {
        child.material.transparent = true;
        child.material.opacity = Math.max(0, Math.min(1, groupOpacity));
      }
    });

    // ---------- normalized progress ----------
    const so = Math.min(1, Math.max(0, scroll.offset || 0));
    const sectionProgress = animationEndOffset > 0 ? Math.min(1, Math.max(0, so / animationEndOffset)) : 0;

    // ---------- spiral base ----------
    const theta = sectionProgress * turns * Math.PI * 2 + angleOffset;
    const radius = THREE.MathUtils.lerp(radiusStart, radiusEnd, sectionProgress);
    const spiralX = Math.cos(theta) * radius;
    const spiralZ = Math.sin(theta) * radius;
    const spiralY = -sectionProgress * verticalTravel;

    // ---------- poses ----------
    const POSES = {
      intro: { x: 0.0, z: 3.0, y: 0.0 },
      feature1: { x: -1.6, z: 2.0, y: -1.2 },
      feature2: { x: 1.6, z: 2.0, y: -1.2 },
      feature3: { x: -1.6, z: 2.0, y: -2.6 }
    };

    // ---------- page visibility (still useful but don't rely on it alone) ----------
    const pageSize = 1 / totalPages;
    const page0Vis = scroll.visible(0 * pageSize, 0.8 * pageSize);
    const page1Vis = scroll.visible(0.8 * pageSize, 1 * pageSize);
    const page2Vis = scroll.visible(1.8 * pageSize, 1 * pageSize);
    const page3Vis = scroll.visible(2.8 * pageSize, 1 * pageSize);

    // helper centers in sectionProgress-space
    const center = (pageIndex) => ((pageIndex / totalPages) / (animationEndOffset === 0 ? 1 : animationEndOffset));
    const center0 = center(0), center1 = center(1), center2 = center(2), center3 = center(3);

    // ---------- target (blend start-center -> spiral) ----------
    const blendX = THREE.MathUtils.lerp(POSES.intro.x, spiralX, sectionProgress);
    const blendZ = THREE.MathUtils.lerp(POSES.intro.z, spiralZ, sectionProgress);
    const blendY = THREE.MathUtils.lerp(POSES.intro.y, spiralY, sectionProgress);

    let targetX = blendX, targetZ = blendZ, targetY = blendY;

    // nudge upward when near centers
    if (Math.abs(sectionProgress - center1) < nearCenterWindow) {
      targetY = THREE.MathUtils.lerp(targetY, POSES.feature1.y + liftAboveCard, 0.9);
    } else if (Math.abs(sectionProgress - center2) < nearCenterWindow) {
      targetY = THREE.MathUtils.lerp(targetY, POSES.feature2.y + liftAboveCard, 0.9);
    } else if (Math.abs(sectionProgress - center3) < nearCenterWindow) {
      targetY = THREE.MathUtils.lerp(targetY, POSES.feature3.y + liftAboveCard, 0.9);
    }

    // snap to composed pose when page visible (for clean framing)
    if (page0Vis) {
      targetX = POSES.intro.x; targetZ = POSES.intro.z; targetY = POSES.intro.y;
    } else if (page1Vis) {
      targetX = POSES.feature1.x; targetZ = POSES.feature1.z; targetY = POSES.feature1.y + liftAboveCard;
    } else if (page2Vis) {
      targetX = POSES.feature2.x; targetZ = POSES.feature2.z; targetY = POSES.feature2.y + liftAboveCard;
    } else if (page3Vis) {
      targetX = POSES.feature3.x; targetZ = POSES.feature3.z; targetY = POSES.feature3.y + liftAboveCard;
    }

    // ---------- apply position damping ----------
    const cur = wheelchairRef.current.position;
    const nextX = THREE.MathUtils.damp(cur.x, targetX, posDamping, delta);
    const nextY = THREE.MathUtils.damp(cur.y, targetY, posDamping, delta);
    const nextZ = THREE.MathUtils.damp(cur.z, targetZ, posDamping, delta);
    wheelchairRef.current.position.set(nextX, nextY, nextZ);

    // ---------- robust spin logic (use sectionProgress proximity to decide) ----------
    const nearCard = Math.abs(sectionProgress - center0) < nearCenterWindow
      || Math.abs(sectionProgress - center1) < nearCenterWindow
      || Math.abs(sectionProgress - center2) < nearCenterWindow
      || Math.abs(sectionProgress - center3) < nearCenterWindow;

    if (wheelchairRef.current.userData.__spinAngle === undefined) {
      wheelchairRef.current.userData.__spinAngle = wheelchairRef.current.rotation.y || 0;
    }

    let desiredYaw;
    if (!nearCard) {
      wheelchairRef.current.userData.__spinAngle += spinSpeed * delta;
      if (wheelchairRef.current.userData.__spinAngle > Math.PI * 2) wheelchairRef.current.userData.__spinAngle -= Math.PI * 2;
      desiredYaw = wheelchairRef.current.userData.__spinAngle;
    } else {
      // face camera when near card
      const cam = state.camera;
      const dx = cam.position.x - targetX;
      const dz = cam.position.z - targetZ;
      desiredYaw = Math.atan2(dx, dz);
      wheelchairRef.current.userData.__spinAngle = desiredYaw;
    }

    const curY = wheelchairRef.current.rotation.y || 0;
    wheelchairRef.current.rotation.y = THREE.MathUtils.damp(curY, desiredYaw, rotDamping, delta);

    // ---------- debug (temporary) ----------
    // open browser console and watch these while scrolling:
    if (Math.random() < 0.02) {
      // eslint-disable-next-line no-console
      console.log('sp-debug', {
        scrollOffset: so.toFixed(3),
        sectionProgress: sectionProgress.toFixed(3),
        targetX: targetX.toFixed(2),
        targetY: targetY.toFixed(2),
        targetZ: targetZ.toFixed(2),
        nearCard
      });
    }
  });






  // small dev-only debug overlay (remove or comment out in production)
  React.useEffect(() => {
    const dbg = document.createElement('div');
    dbg.id = 'debug-spiral';
    Object.assign(dbg.style, {
      position: 'fixed',
      left: 12,
      top: 12,
      zIndex: 9999,
      color: '#e6f9ff',
      background: '#021018cc',
      padding: '8px 10px',
      fontSize: '12px',
      borderRadius: '6px',
      pointerEvents: 'none'
    });
    document.body.appendChild(dbg);
    const id = setInterval(() => {
      const animationEndOffset = animationPages / totalPages;
      const t = Math.min(1, Math.max(0, (animationEndOffset === 0 ? 0 : scroll.offset / animationEndOffset)));
      dbg.innerHTML = `t:${t.toFixed(3)} turns:${turns} radius:${THREE.MathUtils.lerp(radiusStart, radiusEnd, t).toFixed(2)}`;
    }, 120);
    return () => { clearInterval(id); dbg.remove(); };
  }, [scroll]);

  return (
    <group ref={groupRef}>
      <ambientLight intensity={0.8} />
      <directionalLight position={[10, 10, 5]} intensity={1.5} castShadow />
      <group ref={wheelchairRef} scale={0.8}>
        <WheelchairGLTFModel />
        <pointLight color="red" intensity={0} position={[0, 1, 0.5]} distance={10} decay={2} />
      </group>
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
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-8">
                {Array.from({ length: 6 }).map((_, i) => (
                  <div key={i} className="text-center">
                    <div className="w-24 h-24 bg-gray-700 rounded-full mx-auto mb-2"></div>
                    <h4 className="font-bold">Team Member</h4>
                    <p className="text-sm text-gray-400">Role</p>
                  </div>
                ))}
              </div>
            </div>
          </AnimatedSection>

          <AnimatedSection>
            <div className="max-w-4xl mx-auto text-center">
              <h2 className="text-4xl md:text-5xl font-bold text-brand-cyan mb-8">Get Involved</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="bg-gray-800/50 p-8 rounded-lg text-center border border-gray-700 flex flex-col justify-between">
                  <div>
                    <h3 className="text-3xl font-bold mb-4 flex items-center justify-center"><BriefcaseIcon /> Partner With Us</h3>
                    <p className="mb-6">We are seeking corporate partners who share our vision. For inquiries and sponsorship opportunities, please get in touch with our team.</p>
                  </div>
                  <button onClick={onSponsorClick} className="bg-transparent border-2 border-brand-cyan text-brand-cyan font-bold py-3 px-8 rounded-full hover:bg-brand-cyan hover:text-brand-dark transition-colors inline-block mt-4">Contact Sponsorship</button>
                </div>

                <div className="bg-gray-800/50 p-8 rounded-lg text-center border border-gray-700 flex flex-col justify-between">
                  <div>
                    <h3 className="text-3xl font-bold mb-4 flex items-center justify-center"><HeartIcon /> Support Our Mission</h3>
                    <p className="mb-6">Your individual contribution fuels our research and helps keep the final product affordable for those who need it most.</p>
                  </div>
                  <button onClick={onQrClick} className="bg-brand-cyan text-brand-dark font-bold py-3 px-8 rounded-full hover:bg-white transition-colors mt-4">Support with QR Code</button>
                </div>
              </div>
            </div>
          </AnimatedSection>

          {/* ---------- Page 7: Credits / Footer (full screen) ---------- */}
          <div className="w-full h-screen flex flex-col items-center justify-center text-center bg-brand-dark p-8">
            <div className="max-w-3xl">
              <h2 className="text-4xl md:text-5xl font-bold text-brand-cyan mb-4">Credits & License</h2>

              <p className="text-gray-300 mb-4">
                3D Model <span className="font-semibold">"Combat Wheelchair Base Model - Spiral Metal"</span>
                {' '}by{' '}
                <a
                  href="https://www.fab.com/listings/035ad9b4-8f89-4e83-b637-2dc0fc0bcd13"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-brand-cyan hover:underline"
                  aria-label="Open model page on Fab (opens in new tab)"
                >
                  Chakyas K Kiron
                </a>.
              </p>

              <p className="text-sm text-gray-400 mb-6">
                Licensed as described on the model page. Visit the author page above for license details and usage permissions.
              </p>

              <div className="flex gap-3 justify-center">
                <a
                  href="https://github.com/ChampionSamay1644/Smart-Wheelchair"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-6 py-3 rounded-full border border-white/10 bg-white/5 hover:bg-white/10 text-brand-cyan font-semibold"
                  aria-label="Open project GitHub (opens in new tab)"
                >
                  Project GitHub
                </a>
              </div>

              <p className="text-xs text-gray-500 mt-8">
                © 2025 SmartNav Project. All Rights Reserved.
              </p>
            </div>
          </div>

          {/* End of static content section */}
        </div>
        {/* End of main scroll container */}
      </div>
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
          <ScrollControls pages={8} damping={0.1}>
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

