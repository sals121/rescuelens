import { useEffect, useState } from 'react'

type Health = { status: string; version: string }

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/api/health')
      .then((r) => (r.ok ? (r.json() as Promise<Health>) : Promise.reject(new Error(r.statusText))))
      .then(setHealth)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  return (
    <main className="flex h-full items-center justify-center font-mono">
      <div className="rounded-lg border border-zinc-800 bg-zinc-900 px-8 py-6">
        <h1 className="text-lg tracking-widest text-amber-400">RESCUELENS</h1>
        <p className="mt-2 text-sm text-zinc-400">
          backend:{' '}
          {health ? (
            <span className="text-emerald-400">
              {health.status} (v{health.version})
            </span>
          ) : error ? (
            <span className="text-red-400">{error}</span>
          ) : (
            'connecting…'
          )}
        </p>
      </div>
    </main>
  )
}
