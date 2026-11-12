import type { ReactNode } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const CANDIDATE_ID = /^F\d{3}-C\d{2}$/

interface Props {
  markdown: string | null
  candidateIds: Set<string>
  onSelect: (id: string) => void
}

/** Claude's written search brief. Rendered without raw HTML, so model output cannot inject markup. */
export function ReportPanel({ markdown, candidateIds, onSelect }: Props) {
  if (!markdown)
    return (
      <p className="text-ink-soft px-4 py-6 text-sm">
        The written report appears once every candidate has been checked.
      </p>
    )
  // candidate ids in the report jump to that candidate on the map and in the photo
  const linkify = (children: ReactNode) =>
    typeof children === 'string' && CANDIDATE_ID.test(children) && candidateIds.has(children) ? (
      <button
        type="button"
        className="decoration-signal font-semibold underline underline-offset-2"
        onClick={() => onSelect(children)}
      >
        {children}
      </button>
    ) : (
      children
    )
  return (
    <article className="report px-4 pb-8 text-sm">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          td: ({ children }) => <td>{linkify(children)}</td>,
          strong: ({ children }) => <strong>{linkify(children)}</strong>,
          table: ({ children }) => (
            <div className="overflow-x-auto">
              <table>{children}</table>
            </div>
          ),
        }}
      >
        {markdown}
      </Markdown>
    </article>
  )
}
