import type { SharedPromptDocument } from './sharedPromptEditor';
import type { PromptSelection, StudioPreviewPayload } from './promptStudioTypes';

interface PromptPreviewPanelProps {
  compositionKey: 'decision' | 'speech' | null;
  preview: StudioPreviewPayload['preview'] | undefined;
  selection: PromptSelection | null;
  shared: SharedPromptDocument;
}

export default function PromptPreviewPanel({
  compositionKey,
  preview,
  selection,
  shared,
}: PromptPreviewPanelProps) {
  return (
    <section className="prompt-component-preview" aria-live="polite">
      <div className="prompt-panel-title">
        <span>{compositionKey ? 'Composition preview' : 'Shared definition previews'}</span>
         <code>{preview ? 'server preview' : 'awaiting valid candidate'}</code>
      </div>
      {(['decision', 'speech'] as const).filter((consumer) => (
        compositionKey ? consumer === compositionKey
          : selection?.kind === 'shared-phase' ? consumer === 'decision'
            : shared.compositions[consumer].order.includes(selection?.key || '')
      )).map((consumer) => {
        const group = consumer === 'decision' ? preview?.decision : preview?.communication;
        const components = compositionKey ? group?.components : group?.components.filter(
          (component) => selection?.kind === 'shared-phase'
            ? Object.hasOwn(component.variables, 'phase_guidance')
            : component.id === selection?.previewId,
        );
        return <article key={consumer}>
          <h3>{consumer} / {group?.status || 'preview pending'}</h3>
          <p>{group?.provenance || 'No current typed context available.'}</p>
          {selection?.kind === 'shared-phase' && group?.prompt_key !== selection.key
            ? <pre>Not active for the current phase.</pre>
            : components?.map((component) => <div key={component.id}>
              <h3>{component.id}</h3>
              <pre>{component.rendered || 'No current rendering. Authored template only:'}</pre>
              {!component.rendered && <pre>{component.template}</pre>}
              <h3>Input provenance</h3>
              <pre>{JSON.stringify(component.variables, null, 2)}</pre>
            </div>)}
          {!components?.length && <pre>No rendered component is available for this context.</pre>}
          {group?.board_presentation && <>
            <h3>Board presentation provenance</h3>
            <pre>{JSON.stringify(group.board_presentation, null, 2)}</pre>
          </>}
        </article>;
      })}
    </section>
  );
}
