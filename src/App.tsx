// Zero‑Key Voice Similarity MVP (v4.1)
// Status: Smart vocal emphasis (low‑energy + low‑HNR) ACTIVE with Off/Basic/Smart toggle in UI.
// New in v4:
// - White & gold aesthetic + subtle musical‑note accents (SVG, no deps)
// - Recording duration control (defaults to 30s, supports 5–60s)
// - Minor layout polish
// - NEW: Vocal emphasis toggle with Off / Basic (min‑noise subtraction) / Smart (low‑energy + low‑HNR median noise)
//   Smart uses short‑lag autocorrelation to approximate HNR and selects noise frames more robustly.

import React, { useMemo, useRef, useState } from 'react';

// ========== Cosine Similarity ==========
function cosineSim(a: number[], b: number[]) {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
}

// ========== Small helpers ==========
function hannWindow(N: number) {
  const w = new Float32Array(N);
  for (let n = 0; n < N; n++) w[n] = 0.5 * (1 - Math.cos((2 * Math.PI * n) / (N - 1)));
  return w;
}
function hzToMel(f: number) { return 2595 * Math.log10(1 + f / 700); }
function melToHz(m: number) { return 700 * (Math.pow(10, m / 2595) - 1); }

function melFilterbank(fftSize: number, sr: number, nMels: number, fmin: number, fmax: number) {
  const fMinMel = hzToMel(fmin), fMaxMel = hzToMel(fmax);
  const mels = new Array(nMels + 2).fill(0).map((_, i) => fMinMel + (i * (fMaxMel - fMinMel)) / (nMels + 1));
  const hz = mels.map(melToHz);
  const bins = hz.map(f => Math.floor(((fftSize + 1) * f) / sr));
  const fb: number[][] = new Array(nMels).fill(0).map(() => new Array(fftSize / 2 + 1).fill(0));
  for (let m = 1; m <= nMels; m++) {
    const f_m_minus = bins[m - 1], f_m = bins[m], f_m_plus = bins[m + 1];
    for (let k = f_m_minus; k <= f_m; k++) if (k >= 0 && k < fb[m - 1].length) fb[m - 1][k] = (k - f_m_minus) / Math.max(1, (f_m - f_m_minus));
    for (let k = f_m; k <= f_m_plus; k++) if (k >= 0 && k < fb[m - 1].length) fb[m - 1][k] = (f_m_plus - k) / Math.max(1, (f_m_plus - f_m));
  }
  return fb;
}

function dctMatrix(nMfcc: number, nMels: number) {
  const M: number[][] = Array.from({ length: nMfcc }, () => Array(nMels).fill(0));
  const factor = Math.PI / nMels;
  for (let i = 0; i < nMfcc; i++) {
    for (let j = 0; j < nMels; j++) {
      M[i][j] = Math.cos((j + 0.5) * i * factor) * (i === 0 ? Math.sqrt(1 / nMels) : Math.sqrt(2 / nMels));
    }
  }
  return M;
}

function powerSpectrum(fft: { re: Float64Array; im: Float64Array }) {
  const N2 = fft.re.length / 2 + 1;
  const p = new Float64Array(N2);
  for (let k = 0; k < N2; k++) { const r = fft.re[k], im = fft.im[k]; p[k] = r * r + im * im; }
  return p;
}

// Minimal in‑place radix‑2 FFT for real input; zero‑pads to N.
function fftReal(x: Float32Array, N: number) {
  const re = new Float64Array(N).fill(0), im = new Float64Array(N).fill(0);
  for (let n = 0; n < x.length; n++) re[n] = x[n];
  let i = 0, j = 0;
  for (i = 1; i < N - 1; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { const tr = re[i]; re[i] = re[j]; re[j] = tr; const ti = im[i]; im[i] = im[j]; im[j] = ti; }
  }
  for (let len = 2; len <= N; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wlenRe = Math.cos(ang), wlenIm = Math.sin(ang);
    for (let i = 0; i < N; i += len) {
      let wRe = 1, wIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const uRe = re[i + k], uIm = im[i + k];
        const vRe = re[i + k + len / 2] * wRe - im[i + k + len / 2] * wIm;
        const vIm = re[i + k + len / 2] * wIm + im[i + k + len / 2] * wRe;
        re[i + k] = uRe + vRe; im[i + k] = uIm + vIm;
        re[i + k + len / 2] = uRe - vRe; im[i + k + len / 2] = uIm - vIm;
        const nwRe = wRe * wlenRe - wIm * wlenIm;
        const nwIm = wRe * wlenIm + wIm * wlenRe;
        wRe = nwRe; wIm = nwIm;
      }
    }
  }
  return { re, im };
}

// ========== Spectral subtraction (simple) ==========
function spectralSubtractAll(specs: Float64Array[], alpha = 1.0, floor = 0.1): Float64Array[] {
  if (specs.length === 0) return specs;
  const K = specs[0].length;
  const noise = new Float64Array(K).fill(Number.POSITIVE_INFINITY);
  for (const s of specs) for (let k = 0; k < K; k++) noise[k] = Math.min(noise[k], s[k]);
  const out: Float64Array[] = [];
  for (const s of specs) {
    const o = new Float64Array(K);
    for (let k = 0; k < K; k++) o[k] = Math.max(s[k] - alpha * noise[k], floor * noise[k]);
    out.push(o);
  }
  return out;
}

// ========== MFCC (mean) with optional vocal emphasis ==========
function mfccMean(
  x: Float32Array,
  sr = 16000,
  opts = { frameMs: 25, hopMs: 10, fftSize: 512, nMels: 40, nMfcc: 13, fmin: 50, fmax: 7600, vocalEmphasis: 'off' as 'off' | 'basic' | 'smart' }
): number[] {
  const frame = Math.round((sr * opts.frameMs) / 1000);
  const hop = Math.round((sr * opts.hopMs) / 1000);
  const hann = hannWindow(frame);
  const mel = melFilterbank(opts.fftSize, sr, opts.nMels, opts.fmin, opts.fmax);
  const dct = dctMatrix(opts.nMfcc, opts.nMels);

    const specs: Float64Array[] = [];
  const framesRaw: Float32Array[] = [];
  for (let start = 0; start + frame <= x.length; start += hop) {
    const win = new Float32Array(frame);
    for (let i = 0; i < frame; i++) win[i] = x[start + i] * hann[i];
    framesRaw.push(win);
    specs.push(powerSpectrum(fftReal(win, opts.fftSize)));
  }
  let usedSpecs: Float64Array[] = specs;
  if (opts.vocalEmphasis === 'basic') {
    usedSpecs = spectralSubtractAll(specs, 1.0, 0.1);
  } else if (opts.vocalEmphasis === 'smart') {
    const noiseIdx = pickNoiseFrames(framesRaw, sr);
    if (noiseIdx.length) {
      const K = specs[0].length;
      const noiseFrames = noiseIdx.map(i => specs[i]);
      const noise = new Float64Array(K);
      for (let k = 0; k < K; k++) {
        const col = noiseFrames.map(s => s[k]).sort((a,b)=>a-b);
        noise[k] = col[Math.floor(col.length/2)];
      }
      const out: Float64Array[] = [];
      for (const s of specs) {
        const o = new Float64Array(K);
        for (let k = 0; k < K; k++) o[k] = Math.max(s[k] - noise[k], 0.1 * noise[k]);
        out.push(o);
      }
      usedSpecs = out;
    } else {
      usedSpecs = spectralSubtractAll(specs, 1.0, 0.1);
    }
  }

  const frames: number[][] = [];
  for (const spec of usedSpecs) {
    const melE = new Float64Array(mel.length);
    for (let m = 0; m < mel.length; m++) {
      let sum = 0; const filt = mel[m];
      for (let k = 0; k < filt.length; k++) sum += filt[k] * spec[k];
      melE[m] = Math.log(1e-10 + sum);
    }
    const mfcc = new Array(dct.length).fill(0);
    for (let i = 0; i < dct.length; i++) {
      let s = 0, row = dct[i];
      for (let j = 0; j < row.length; j++) s += row[j] * melE[j];
      mfcc[i] = s;
    }
    frames.push(mfcc);
  }

  const mean = new Array(opts.nMfcc).fill(0);
  for (const v of frames) for (let i = 0; i < mean.length; i++) mean[i] += v[i];
  for (let i = 0; i < mean.length; i++) mean[i] /= Math.max(1, frames.length);
  return mean;
}

// ========== Audio decode / resample ==========
async function decodeToMono16k(arrayBuf: ArrayBuffer): Promise<Float32Array> {
  const ctx = new AudioContext();
  const buf = await ctx.decodeAudioData(arrayBuf.slice(0));
  const mono = (() => {
    if (buf.numberOfChannels === 1) return buf.getChannelData(0);
    const L = buf.getChannelData(0), R = buf.getChannelData(1);
    const out = new Float32Array(buf.length);
    for (let i = 0; i < buf.length; i++) out[i] = 0.5 * (L[i] + R[i]);
    return out;
  })();
  if (buf.sampleRate === 16000) return mono;
  const off = new OfflineAudioContext(1, Math.ceil(mono.length * 16000 / buf.sampleRate), 16000);
  const b = off.createBuffer(1, mono.length, buf.sampleRate);
  b.copyToChannel(mono, 0);
  const src = off.createBufferSource();
  src.buffer = b;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return rendered.getChannelData(0);
}

// ========== HNR & noise-frame helpers ==========
function frameEnergy(f: Float32Array){ let s=0; for(let i=0;i<f.length;i++){ const v=f[i]; s+=v*v; } return s/Math.max(1,f.length); }
function zeroCrossRate(f: Float32Array){ let z=0; for(let i=1;i<f.length;i++){ if((f[i-1]>=0)!==(f[i]>=0)) z++; } return z/Math.max(1,(f.length-1)); }
function hnrApprox(f: Float32Array, sr:number){
  // Autocorr in 80–400 Hz range
  const minLag = Math.floor(sr/400), maxLag = Math.ceil(sr/80);
  let r0 = 0; for(let i=0;i<f.length;i++) r0 += f[i]*f[i];
  let best = 0;
  for(let lag=minLag; lag<=maxLag; lag++){
    let r=0; for(let i=lag;i<f.length;i++) r += f[i]*f[i-lag];
    if (r>best) best=r;
  }
  const eps=1e-9; return 10*Math.log10((best+eps)/(r0+eps));
}
function pickNoiseFrames(frames: Float32Array[], sr:number){
  const feats = frames.map(f=>({ e: frameEnergy(f), h: hnrApprox(f, sr), z: zeroCrossRate(f) }));
  const idx = feats.map((_,i)=>i);
  idx.sort((i,j)=> (feats[i].e - feats[j].e) || (feats[i].h - feats[j].h) || (feats[i].z - feats[j].z));
  const n = Math.max(5, Math.floor(frames.length*0.15));
  return idx.slice(0, n);
}

// ========== iTunes Search (no key) ==========
async function searchITunes(artistOrSong: string) {
  const url = `https://itunes.apple.com/search?term=${encodeURIComponent(artistOrSong)}&media=music&limit=8`;
  const res = await fetch(url);
  const data = await res.json();
  const out = (data.results || [])
    .filter((r: any) => r.previewUrl)
    .map((r: any) => ({
      id: String(r.trackId),
      artist: r.artistName,
      title: r.trackName,
      previewUrl: r.previewUrl as string,
      artwork: r.artworkUrl100 as string,
    }));
  return out as { id: string; artist: string; title: string; previewUrl: string; artwork: string }[];
}

// ========== Local storage helpers ==========
function saveArtistVector(rec: { id: string; artist: string; title: string }, vec: number[]) {
  const key = `artistVec:${rec.id}`;
  localStorage.setItem(key, JSON.stringify({ meta: rec, vec }));
}
function loadAllArtistVectors() {
  const out: { name: string; title: string; vec: number[] }[] = [];
  for (let i = 0; i < localStorage.length; i++) {
    const k = localStorage.key(i)!;
    if (!k.startsWith('artistVec:')) continue;
    const { meta, vec } = JSON.parse(localStorage.getItem(k)!);
    out.push({ name: meta.artist, title: meta.title, vec });
  }
  return out;
}
function clearArtistVectors() {
  const keys: string[] = [];
  for (let i = 0; i < localStorage.length; i++) {
    const k = localStorage.key(i)!;
    if (k.startsWith('artistVec:')) keys.push(k);
  }
  for (const k of keys) localStorage.removeItem(k);
}

// ========== Recording (Gia) ==========
async function recordGia(seconds = 30): Promise<Float32Array> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const ctx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 48000 });
  const src = ctx.createMediaStreamSource(stream);
  const dest = ctx.createMediaStreamDestination();
  src.connect(dest);
  const recorder = new MediaRecorder(dest.stream);
  const chunks: BlobPart[] = [];
  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.start();
  await new Promise((r) => setTimeout(r, seconds * 1000));
  recorder.stop();
  await new Promise((r) => (recorder.onstop = r as any));
  stream.getTracks().forEach((t) => t.stop());
  const blob = new Blob(chunks, { type: 'audio/webm' });
  const arr = await blob.arrayBuffer();
  return await decodeToMono16k(arr);
}

// ========== Decorative musical notes (SVG) ==========
function NotesBadge(){
  return (
    <svg viewBox="0 0 200 60" className="w-40 h-12">
      <defs>
        <linearGradient id="g" x1="0" x2="1">
          <stop offset="0%" stopColor="#f7e9a6"/>
          <stop offset="100%" stopColor="#e4b400"/>
        </linearGradient>
      </defs>
      <g fill="url(#g)">
        <path d="M20 40 q10 -20 20 0 q10 -20 20 0" opacity="0.5"/>
        <circle cx="70" cy="35" r="4" />
        <rect x="74" y="20" width="3" height="18" rx="1"/>
        <circle cx="90" cy="30" r="4" />
        <rect x="94" y="15" width="3" height="20" rx="1"/>
        <circle cx="110" cy="38" r="4" />
        <rect x="114" y="22" width="3" height="18" rx="1"/>
      </g>
    </svg>
  );
}

// ========== UI Component ==========
export default function App() {
  const [query, setQuery] = useState('Shania Twain');
  const [songQuery, setSongQuery] = useState('');
  const [results, setResults] = useState<{ id: string; artist: string; title: string; previewUrl: string; artwork: string }[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [giaVec, setGiaVec] = useState<number[] | null>(null);
  const [matches, setMatches] = useState<{ name: string; title: string; score: number }[]>([]);
  const [vocalEmphasis, setVocalEmphasis] = useState<'off' | 'basic' | 'smart'>('smart');
  const [recLen, setRecLen] = useState(30);
  const audioRef = useRef<HTMLAudioElement>(null);

  const artistBank = useMemo(() => loadAllArtistVectors(), [results, giaVec, busy]);

  async function doSearch() {
    setBusy('Searching…');
    try {
      const res = await searchITunes(query);
      setResults(res);
    } finally {
      setBusy(null);
    }
  }

  async function addTrackVector(t: { id: string; artist: string; title: string; previewUrl: string }) {
    setBusy(`Analyzing “${t.title}”`);
    try {
      const arr = await fetch(t.previewUrl).then((r) => r.arrayBuffer());
      const y = await decodeToMono16k(arr);
      const vec = mfccMean(y, 16000, { frameMs:25, hopMs:10, fftSize:512, nMels:40, nMfcc:13, fmin:50, fmax:7600, vocalEmphasis });
      saveArtistVector({ id: t.id, artist: t.artist, title: t.title }, vec);
    } catch (e) {
      console.error(e);
      alert('Failed to analyze preview. Try another track.');
    } finally {
      setBusy(null);
    }
  }

  async function addBySongName() {
    if (!songQuery.trim()) return;
    setBusy('Searching song…');
    try {
      const res = await searchITunes(songQuery.trim());
      if (!res.length) { alert('No preview found for that query.'); return; }
      await addTrackVector(res[0]);
    } finally { setBusy(null); }
  }

  async function handleRecordGia() {
    setBusy(`Recording Gia for ${recLen}s…`);
    try {
      const y = await recordGia(Math.max(5, Math.min(60, recLen)));
      const v = mfccMean(y, 16000, { frameMs:25, hopMs:10, fftSize:512, nMels:40, nMfcc:13, fmin:50, fmax:7600, vocalEmphasis });
      setGiaVec(v);
    } catch (e) {
      console.error(e);
      alert('Mic permission or recording failed.');
    } finally {
      setBusy(null);
    }
  }

  async function handleUploadGia(file: File) {
    setBusy('Processing upload…');
    try {
      const arr = await file.arrayBuffer();
      const y = await decodeToMono16k(arr);
      const v = mfccMean(y, 16000, { frameMs:25, hopMs:10, fftSize:512, nMels:40, nMfcc:13, fmin:50, fmax:7600, vocalEmphasis });
      setGiaVec(v);
    } catch (e) {
      console.error(e);
      alert('Could not read that file. Try a WAV/MP3/M4A.');
    } finally { setBusy(null); }
  }

  function compareNow() {
    if (!giaVec) { alert('Record or upload Gia first.'); return; }
    const bank = loadAllArtistVectors();
    if (!bank.length) { alert('Add at least one reference track.'); return; }
    const scored = bank.map(b => ({ name: b.name, title: b.title, score: cosineSim(giaVec, b.vec) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 5);
    setMatches(scored);
  }

  function clearBank() { clearArtistVectors(); setMatches([]); setGiaVec(null); }

  return (
    <div className="min-h-screen bg-white text-yellow-800 relative overflow-hidden">
      {/* gold gradient ribbon */}
      <div className="absolute inset-x-0 -top-10 h-40 bg-gradient-to-r from-yellow-100 via-yellow-200 to-yellow-100 blur-2xl opacity-70 pointer-events-none"/>
      <div className="absolute right-6 top-4 opacity-80"><NotesBadge/></div>

      <div className="max-w-6xl mx-auto p-6">
        <header className="flex items-center gap-4 mb-6">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-yellow-200 to-yellow-400 border border-yellow-300 flex items-center justify-center shadow-sm">
            <span className="text-xl">♪</span>
          </div>
          <div>
            <h1 className="text-3xl font-semibold">Voice Match</h1>
            <p className="text-sm text-yellow-700/80">White & gold · MFCC cosine · zero‑key iTunes previews · <span className="inline-flex items-center gap-1"><span className="opacity-70">Vocal:</span><span className="font-medium">{vocalEmphasis}</span></span></p>
          </div>
        </header>

        <div className="grid md:grid-cols-3 gap-6">
          {/* Left: Search & Results */}
          <div className="md:col-span-2">
            <div className="flex flex-wrap items-center gap-3 mb-3">
              <input
                className="flex-1 border border-yellow-200 rounded-xl px-3 py-2 outline-none focus:ring-2 focus:ring-yellow-300 bg-white"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search artist or song (e.g., Shania Twain)"
              />
              <button onClick={doSearch} className="px-4 py-2 rounded-xl bg-yellow-500 text-white shadow hover:bg-yellow-600">Search</button>
              <div className="flex items-center gap-2 text-sm px-2 py-1 rounded-lg bg-yellow-50 border border-yellow-200">
                <span>Vocal emphasis:</span>
                <select className="border border-yellow-200 rounded-md px-2 py-1 bg-white" value={vocalEmphasis} onChange={e=>setVocalEmphasis(e.target.value as any)}>
                  <option value="off">Off</option>
                  <option value="basic">Basic</option>
                  <option value="smart">Smart</option>
                </select>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-3">
              {results.map((r) => (
                <div key={r.id} className="rounded-2xl border border-yellow-100 p-3 bg-white shadow-sm hover:shadow-md transition">
                  <div className="flex items-center gap-3">
                    <img src={r.artwork} alt="art" className="w-16 h-16 rounded-xl object-cover border border-yellow-100" />
                    <div className="flex-1">
                      <div className="font-medium text-yellow-900">{r.title}</div>
                      <div className="text-sm text-yellow-700/80">{r.artist}</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 mt-3">
                    <audio ref={audioRef} src={r.previewUrl} controls className="w-full" />
                  </div>
                  <div className="flex gap-2 mt-3">
                    <button onClick={() => addTrackVector(r)} className="px-3 py-2 rounded-xl border border-yellow-200 bg-yellow-50 hover:bg-yellow-100">Add to Reference</button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Right: Actions */}
          <div className="space-y-4">
            <div className="rounded-2xl border border-yellow-100 bg-white p-4 shadow-sm">
              <div className="font-semibold mb-2 text-yellow-900">Gia</div>
              <div className="flex items-center gap-2 mb-2">
                <label className="text-sm">Record length:</label>
                <input type="range" min={5} max={60} value={recLen} onChange={e=>setRecLen(parseInt(e.target.value))} />
                <span className="text-sm tabular-nums w-10 text-right">{recLen}s</span>
              </div>
              <button onClick={handleRecordGia} className="w-full px-4 py-2 rounded-xl bg-yellow-500 text-white shadow hover:bg-yellow-600">Record</button>
              <div className="mt-3 text-sm text-yellow-700/80">or upload a file:</div>
              <div className="mt-1">
                <input className="text-sm" type="file" accept="audio/*" onChange={e=> e.target.files && handleUploadGia(e.target.files[0])} />
              </div>
              {giaVec && (
                <div className="text-xs text-yellow-700/80 mt-2">MFCC ready · dim={giaVec.length}</div>
              )}
            </div>

            <div className="rounded-2xl border border-yellow-100 bg-white p-4 shadow-sm">
              <div className="font-semibold mb-2 text-yellow-900">Add by Song Name</div>
              <div className="flex gap-2">
                <input className="flex-1 border border-yellow-200 rounded-xl px-3 py-2 bg-white" placeholder="e.g., Laufey - From The Start" value={songQuery} onChange={e=>setSongQuery(e.target.value)} />
                <button onClick={addBySongName} className="px-3 py-2 rounded-xl border border-yellow-200 bg-yellow-50 hover:bg-yellow-100">Add</button>
              </div>
              <div className="text-xs text-yellow-700/80 mt-2">Takes the first iTunes preview match and stores its MFCC vector.</div>
            </div>

            <div className="rounded-2xl border border-yellow-100 bg-white p-4 shadow-sm">
              <div className="font-semibold mb-2 text-yellow-900">Reference Bank</div>
              <div className="text-sm text-yellow-700/80">Stored tracks: {artistBank.length}</div>
              <div className="flex gap-2 mt-3">
                <button onClick={compareNow} className="flex-1 px-3 py-2 rounded-xl border border-yellow-200 bg-yellow-50 hover:bg-yellow-100">Compare</button>
                <button onClick={clearBank} className="px-3 py-2 rounded-xl border border-yellow-200">Clear</button>
              </div>
            </div>

            <div className="rounded-2xl border border-yellow-100 bg-white p-4 shadow-sm">
              <div className="font-semibold mb-2 text-yellow-900">Top Matches</div>
              {matches.length === 0 ? (
                <div className="text-sm text-yellow-700/80">No results yet.</div>
              ) : (
                <ul className="space-y-2">
                  {matches.map((m, i) => (
                  <li key={i} className="text-sm">
                    <div className="flex items-center justify-between">
                      <span>{i + 1}. {m.name} — <span className="text-yellow-700/80">{m.title}</span></span>
                      <span className="font-mono">{(m.score * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-2 bg-yellow-100 rounded-full mt-1">
                      <div className="h-2 rounded-full bg-yellow-500" style={{ width: `${Math.max(0, Math.min(100, m.score * 100)).toFixed(0)}%` }} />
                    </div>
                  </li>
                ))}
                </ul>
              )}
            </div>

            {busy && (
              <div className="rounded-2xl border border-yellow-100 bg-white p-4 shadow-sm animate-pulse">
                <div className="text-sm">{busy}</div>
              </div>
            )}

            <div className="rounded-2xl border border-yellow-100 bg-white p-4 shadow-sm">
              <div className="font-semibold mb-2 text-yellow-900">Notes</div>
              <ul className="list-disc pl-5 text-sm text-yellow-700/80 space-y-1">
                <li>Previews are analyzed in memory; only MFCC vectors are stored (localStorage).</li>
                <li>"Vocal emphasis" offers Off / Basic (minimum-noise subtraction) / Smart (low-energy + low-HNR median noise) to better downweight accompaniment.</li>
                <li>For the best "voice-to-voice" comparison, prefer acapella or spoken references.</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
