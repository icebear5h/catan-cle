export const MODEL_PRESETS: Array<{ id: string; label: string }> = [
  // Teacher tier (large / frontier)
  { id: 'thinkingmachines/inkling-small', label: 'Inkling-Small · 276B/12B · VL' },
  { id: 'deepseek/deepseek-v4-flash-0731', label: 'DeepSeek V4 Flash · 284B/13B · cheap' },
  { id: 'z-ai/glm-5.2', label: 'GLM-5.2 · 744B/40B · frontier value' },
  { id: 'qwen/qwen3.5-397b-a17b', label: 'Qwen3.5-397B · VL teacher' },
  { id: 'xiaomi/mimo-v2.5', label: 'MiMo V2.5 · 310B/15B · VL value' },
  { id: 'minimax/minimax-m2.7', label: 'MiniMax M2.7 · 230B/10B' },
  { id: 'moonshotai/kimi-k3', label: 'Kimi K3 · 2.8T/104B · frontier' },
  { id: 'thinkingmachines/inkling', label: 'Inkling · 975B/41B' },
  { id: 'qwen/qwen3.8-max', label: 'Qwen3.8-Max · 2.4T/95B' },
  { id: 'qwen/qwen3.8-27b', label: 'Qwen3.8-27B · eval baseline' },
  { id: 'cerebras/qwen-3.8-27b', label: 'Qwen3.8-27B · Cerebras · fast' },
  { id: 'qwen/qwen3.7-flash', label: 'Qwen3.7 Flash · VL · cheap' },
  { id: 'google/gemini-2.5-flash', label: 'Gemini 2.5 Flash · default' },
  // Student tier (small)
  { id: 'qwen/qwen3.6-35b-a3b', label: 'Qwen3.6 35B-A3B · student' },
  { id: 'qwen/qwen3.5-27b', label: 'Qwen3.5-27B · student' },
  { id: 'google/gemma-4-31b-it', label: 'Gemma 4 31B · student' },
  { id: 'google/gemma-4-26b-a4b-it', label: 'Gemma 4 26B-A4B · MoE student' },
  { id: 'qwen/qwen3.5-9b', label: 'Qwen3.5-9B · local SFT twin' },
  { id: 'openai/gpt-oss-20b', label: 'GPT-OSS-20B · MoE student' },
  { id: 'nvidia/nemotron-3-nano-30b-a3b', label: 'Nemotron 3 Nano · student' },
];
