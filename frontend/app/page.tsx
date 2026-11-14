
'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import styles from './page.module.css';

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

type ConnectionStatus = {
  payments_connected: boolean;
  inventory_connected: boolean;
  payment_transactions?: number;
  inventory_products?: number;
};

type AppPhase = 'idle' | 'listening' | 'thinking' | 'speaking';

export default function Home() {
  const [paymentsConnected, setPaymentsConnected] = useState(false);
  const [inventoryConnected, setInventoryConnected] = useState(false);
  const [paymentsRecords, setPaymentsRecords] = useState(0);
  const [inventoryRecords, setInventoryRecords] = useState(0);

  const [phase, setPhase] = useState<AppPhase>('idle');
  const [transcript, setTranscript] = useState('');
  const [englishQuestion, setEnglishQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [question, setQuestion] = useState('');
  const [error, setError] = useState('');
  const [connecting, setConnecting] = useState<
    'transactions' | 'inventory' | null
  >(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const gainRef = useRef<GainNode | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);
  const recordingRef = useRef(false);
  const startingRef = useRef(false);
  const mountedRef = useRef(true);

  const fetchStatus = useCallback(async () => {
    setError('');

    try {
      const response = await fetch(`${API_BASE}/status`);
      if (!response.ok) throw new Error('Could not fetch backend status');

      const data: ConnectionStatus = await response.json();

      setPaymentsConnected(Boolean(data.payments_connected));
      setInventoryConnected(Boolean(data.inventory_connected));
      setPaymentsRecords(data.payment_transactions ?? 0);
      setInventoryRecords(data.inventory_products ?? 0);
    } catch {
      setError('Cannot connect to the backend. Make sure FastAPI is running on port 8000.');
    }
  }, []);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  useEffect(() => {
    mountedRef.current = true;

    return () => {
      mountedRef.current = false;
      recordingRef.current = false;
      processorRef.current?.disconnect();
      sourceRef.current?.disconnect();
      gainRef.current?.disconnect();
      streamRef.current?.getTracks().forEach((track) => track.stop());

      const audioContext = audioContextRef.current;
      if (audioContext && audioContext.state !== 'closed') {
        void audioContext.close().catch((err) => {
          console.error('Could not close the audio context:', err);
        });
      }
    };
  }, []);

  async function connectSampleData(type: 'transactions' | 'inventory') {
    setError('');
    setConnecting(type);

    try {
      const response = await fetch(`${API_BASE}/connect/${type}/sample`, {
        method: 'POST',
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || result.message || 'Could not connect sample data');
      }

      await fetchStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not connect sample data');
    } finally {
      setConnecting(null);
    }
  }

  function floatToWav(samples: Float32Array[], sampleRate: number): Blob {
    const totalLength = samples.reduce((sum, chunk) => sum + chunk.length, 0);
    const pcm = new Int16Array(totalLength);

    let offset = 0;
    for (const chunk of samples) {
      for (let i = 0; i < chunk.length; i++) {
        const value = Math.max(-1, Math.min(1, chunk[i]));
        pcm[offset++] = value < 0 ? value * 32768 : value * 32767;
      }
    }

    const buffer = new ArrayBuffer(44 + pcm.length * 2);
    const view = new DataView(buffer);

    function writeString(position: number, value: string) {
      for (let i = 0; i < value.length; i++) {
        view.setUint8(position + i, value.charCodeAt(i));
      }
    }

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + pcm.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, pcm.length * 2, true);

    new Uint8Array(buffer, 44).set(
      new Uint8Array(pcm.buffer)
    );

    return new Blob([buffer], { type: 'audio/wav' });
  }

  async function startRecording() {
    if (startingRef.current || recordingRef.current) return;

    startingRef.current = true;
    setError('');
    setTranscript('');
    setEnglishQuestion('');
    setAnswer('');
    chunksRef.current = [];

    if (!navigator.mediaDevices?.getUserMedia) {
      startingRef.current = false;
      setError('Microphone access is not supported in this browser.');
      setPhase('idle');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      if (!mountedRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      streamRef.current = stream;

      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      sourceRef.current = source;

      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      const gain = audioContext.createGain();
      gain.gain.value = 0;
      gainRef.current = gain;

      processor.onaudioprocess = (event) => {
        if (!recordingRef.current) return;
        const input = event.inputBuffer.getChannelData(0);
        chunksRef.current.push(new Float32Array(input));
      };

      source.connect(processor);
      processor.connect(gain);
      gain.connect(audioContext.destination);

      recordingRef.current = true;
      setPhase('listening');
    } catch (err) {
      recordingRef.current = false;
      processorRef.current?.disconnect();
      sourceRef.current?.disconnect();
      gainRef.current?.disconnect();
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;

      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        await audioContextRef.current.close().catch((closeError) => {
          console.error('Could not close the audio context:', closeError);
        });
      }
      audioContextRef.current = null;
      processorRef.current = null;
      sourceRef.current = null;
      gainRef.current = null;

      if (mountedRef.current) {
        setError(
          err instanceof Error
            ? `Microphone error: ${err.message}`
            : 'Please allow microphone access.'
        );
        setPhase('idle');
      }
    } finally {
      startingRef.current = false;
    }
  }

  async function stopRecording() {
    if (!recordingRef.current) return;

    recordingRef.current = false;
    setPhase('thinking');

    const audioContext = audioContextRef.current;
    const stream = streamRef.current;
    const samples = chunksRef.current;

    processorRef.current?.disconnect();
    sourceRef.current?.disconnect();
    gainRef.current?.disconnect();
    stream?.getTracks().forEach((track) => track.stop());

    if (audioContext && audioContext.state !== 'closed') {
      await audioContext.close().catch((err) => {
        console.error('Could not close the audio context:', err);
      });
    }

    const sampleRate = audioContext?.sampleRate || 44100;

    audioContextRef.current = null;
    streamRef.current = null;
    processorRef.current = null;
    sourceRef.current = null;
    gainRef.current = null;
    chunksRef.current = [];

    if (!samples.length) {
      setError('No audio was captured. Please try again.');
      setPhase('idle');
      return;
    }

    const wavBlob = floatToWav(samples, sampleRate);
    await sendVoiceQuestion(wavBlob);
  }

  async function sendVoiceQuestion(blob: Blob) {
    try {
      const formData = new FormData();
      formData.append('file', blob, 'merchant_question.wav');
      formData.append('language', 'unknown');
      formData.append('mode', 'transcribe');

      const response = await fetch(`${API_BASE}/ask-voice`, {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || 'Voice request failed');
      }

      setTranscript(result.transcript || result.question || '');
      setEnglishQuestion(result.translated_question || '');
      setAnswer(result.answer || 'I could not generate an answer.');
      if (result.audio_error) setError(result.audio_error);

      if (result.audio_base64) {
        const binary = atob(result.audio_base64);
        const bytes = new Uint8Array(binary.length);

        for (let i = 0; i < binary.length; i++) {
          bytes[i] = binary.charCodeAt(i);
        }

        const audioBlob = new Blob(
          [bytes],
          { type: result.audio_content_type || 'audio/wav' }
        );

        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);

        setPhase('speaking');

        audio.onended = () => {
          URL.revokeObjectURL(audioUrl);
          setPhase('idle');
        };

        audio.onerror = () => {
          URL.revokeObjectURL(audioUrl);
          setPhase('idle');
          setError('The answer was generated, but audio playback failed.');
        };

        await audio.play();
      } else {
        setPhase('idle');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not process voice question');
      setPhase('idle');
    }
  }

  async function sendTextQuestion(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmed = question.trim();
    if (!trimmed) return;

    setError('');
    setTranscript(trimmed);
    setEnglishQuestion('');
    setAnswer('');
    setPhase('thinking');

    try {
      const response = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: trimmed,
          language: 'unknown',
        }),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || 'Question failed');
      }

      setEnglishQuestion(result.translated_question || '');
      setAnswer(result.answer || 'No answer received.');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not process question');
    } finally {
      setPhase('idle');
    }
  }

  const isRecording = phase === 'listening';

  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <div className={styles.brand}>
          <div className={styles.brandIcon}>D</div>
          <div>
            <h1>Dukaan Saathi</h1>
            <p>Your intelligent shop assistant</p>
          </div>
        </div>
        <div className={styles.status}>
          <span className={styles.statusDot} />
          AI ASSISTANT
        </div>
      </header>

      <section className={styles.content}>
        <div className={styles.intro}>
          <p className={styles.eyebrow}>YOUR BUSINESS, IN YOUR VOICE</p>
          <h2>Good morning, merchant.</h2>
          <p>Ask anything about your sales, customers, and inventory.</p>
        </div>

        <section className={styles.connections}>
          <button
            className={`${styles.connectionCard} ${
              paymentsConnected ? styles.connected : ''
            }`}
            onClick={() => {
              if (!paymentsConnected) connectSampleData('transactions');
            }}
            disabled={connecting !== null}
          >
            <span className={styles.connectionIcon}>₹</span>
            <span className={styles.connectionText}>
              <strong>Connect Transactions</strong>
              <small>
                {connecting === 'transactions'
                  ? 'Connecting...'
                  : paymentsConnected
                  ? `Connected · ${paymentsRecords} records`
                  : 'Use included sample data'}
              </small>
            </span>
            <span className={styles.connectionCheck}>
              {paymentsConnected ? '✓' : '+'}
            </span>
          </button>

          <button
            className={`${styles.connectionCard} ${
              inventoryConnected ? styles.connected : ''
            }`}
            onClick={() => {
              if (!inventoryConnected) connectSampleData('inventory');
            }}
            disabled={connecting !== null}
          >
            <span className={styles.connectionIcon}>▦</span>
            <span className={styles.connectionText}>
              <strong>Connect Inventory</strong>
              <small>
                {connecting === 'inventory'
                  ? 'Connecting...'
                  : inventoryConnected
                  ? `Connected · ${inventoryRecords} products`
                  : 'Use included sample data'}
              </small>
            </span>
            <span className={styles.connectionCheck}>
              {inventoryConnected ? '✓' : '+'}
            </span>
          </button>
        </section>

        <section className={styles.voiceSection}>
          <p className={styles.voiceLabel}>
            {phase === 'listening'
              ? 'LISTENING TO YOU'
              : phase === 'thinking'
              ? 'THINKING'
              : phase === 'speaking'
              ? 'SPEAKING'
              : 'TAP TO SPEAK'}
          </p>

          <button
            className={`${styles.orbButton} ${
              isRecording ? styles.orbListening : ''
            } ${phase === 'thinking' ? styles.orbThinking : ''} ${
              phase === 'speaking' ? styles.orbSpeaking : ''
            }`}
            onClick={isRecording ? stopRecording : startRecording}
            disabled={phase === 'thinking' || phase === 'speaking'}
            aria-label={isRecording ? 'Stop recording' : 'Start recording'}
          >
            <span className={styles.orbGlow} />
            <span className={styles.orbCore}>
              {phase === 'thinking' ? (
                <span className={styles.loader} />
              ) : isRecording ? (
                <span className={styles.stopIcon} />
              ) : (
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <rect x="9" y="2" width="6" height="13" rx="3" />
                  <path d="M5 11a7 7 0 0 0 14 0M12 18v4m-4 0h8" />
                </svg>
              )}
            </span>
          </button>

          <p className={styles.voiceHint}>
            {phase === 'listening'
              ? 'Tap again when you finish speaking'
              : phase === 'thinking'
              ? 'Your business data is being analysed...'
              : phase === 'speaking'
              ? 'Here is your answer'
              : 'Speak naturally. I am listening.'}
          </p>
        </section>

        {(transcript || englishQuestion || answer) && (
          <section className={styles.answerCard}>
            {transcript && (
              <div className={styles.transcript}>
                <span className={styles.responseLabel}>YOU SAID</span>
                <p>{transcript}</p>
              </div>
            )}
            {englishQuestion &&
              englishQuestion.trim().toLocaleLowerCase() !==
                transcript.trim().toLocaleLowerCase() && (
                <div className={styles.transcript}>
                  <span className={styles.responseLabel}>
                    ENGLISH USED FOR ANALYSIS
                  </span>
                  <p>{englishQuestion}</p>
                </div>
              )}
            {answer && (
              <div className={styles.answer}>
                <span className={styles.responseLabel}>DUKAAN SAATHI</span>
                <p>{answer}</p>
              </div>
            )}
          </section>
        )}

        

        {error && (
          <div className={styles.error} role="alert">
            {error}
            <button onClick={() => setError('')} aria-label="Dismiss error">
              ×
            </button>
          </div>
        )}

        <button className={styles.refresh} onClick={fetchStatus}>
          ↻ Refresh connections
        </button>
      </section>

      <footer className={styles.footer}>
        POWERED BY SARVAM AI
      </footer>
    </main>
  );
}
