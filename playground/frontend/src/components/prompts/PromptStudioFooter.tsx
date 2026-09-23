import type { SharedSuitePayload } from './promptStudioTypes';

interface PromptStudioFooterProps {
  payload: SharedSuitePayload;
}

export default function PromptStudioFooter({ payload }: PromptStudioFooterProps) {
  return (
  <footer className="prompt-studio-footer" style={{ flexWrap: 'wrap' }}>
    <span>Shared {payload.shared.id}@{payload.shared.version}
      {' · '}{payload.shared.overridden ? 'local override' : 'source default'}</span>
    <code title={payload.shared.sha256}>{payload.shared.sha256.slice(0, 12)}</code>
    <span>{payload.shared.document.memory_mode}; notes limit {payload.shared.document.max_notes_chars}</span>
  </footer>
  );
}
