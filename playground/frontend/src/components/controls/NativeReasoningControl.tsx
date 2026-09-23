import type { NativeReasoningEffort } from '../../types';

interface NativeReasoningControlProps {
  effort: NativeReasoningEffort;
  onChange: (effort: NativeReasoningEffort) => void;
  disabled: boolean;
}

export default function NativeReasoningControl({ effort, onChange, disabled }: NativeReasoningControlProps) {
  return (
    <div className="control-section">
      <h4>Native reasoning</h4>
      <label className="field-label" htmlFor="native-reasoning-effort">
        OpenRouter effort
      </label>
      <select
        id="native-reasoning-effort"
        className="replay-model-select"
        value={effort}
        onChange={(event) => onChange(
          event.target.value as NativeReasoningEffort,
        )}
        disabled={disabled}
      >
        <option value="off">Off — explicit no reasoning</option>
        <option value="minimal">Minimal</option>
        <option value="low">Low</option>
        <option value="medium">Medium</option>
        <option value="high">High — validated default</option>
        <option value="xhigh">XHigh</option>
        <option value="max">Max</option>
      </select>
      <p className="replay-context-hint">
        Sent explicitly to supported providers. Reasoning is read only from
        the provider-native channel and is not requested in the action XML.
      </p>
    </div>
  );
}
