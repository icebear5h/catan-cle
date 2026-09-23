# React + TypeScript + Vite

## Frontend organization

Components use direct feature-folder imports. The September 2026 move map below
is relative to `src/components/`; each listed component's existing `.css` file
moved with its `.tsx` file. Standalone helpers keep their original names.

| Destination | Previously flat components/helpers | Direct files now |
| --- | --- | ---: |
| `activity/` | `GameLog`, `TableTalkLog` | 4 |
| `board/` | `HexBoard`, `PlayerInfo` | 7 |
| `controls/` | `BoardControlsDock`, `GameControls`, `SavedLiveGamesBar` | 9 |
| `prompts/` | `PromptSuiteStudio`, `sharedPromptEditor.ts` | 3 |
| `replay/` | `ReplayResponseCard`, `ReplayTranscriptPanel` | 6 |
| `traces/` | `LiveReasoningTrace`, `SavedStepReasoningTrace`, `RejectedLiveAttempts`, `TraceGamePlan`, `TraceStepNavigator`, `TraceUsage`, `traceModelCalls.ts` | 10 |

The old components folder had 31 direct files; it now contains feature folders
only. `activity/` deliberately avoids `logs/`, which existing Git ignore rules
exclude at every nesting depth.

In `evals/catan_board_bench_ui/src/`, the `DecisionSpotChecks.tsx`/`.css` pair
moved to `decisions/`, and the `ReasoningTraces.tsx`/`.css` pair to `reasoning/`.
That source root went from 17 direct files to 13; the new folders have 2 and 5
files, respectively. The eval UI imports the board from
`@playground/components/board/HexBoard`.

### Trace boundaries

- `replay/ReplayTranscriptPanel.tsx`: 492 → 279 lines. Tab state, scroll refs,
  and effects remain here. `ReplayModelTracePanel.tsx` (149 lines) owns the
  read-only model section; `replayTraceFormat.ts` (80) owns pure formatters.
- Eval `reasoning/ReasoningTraces.tsx`: 634 → 293 lines. State, fetching,
  selection, and effects remain here. `TraceInspector.tsx` (168) owns the
  existing inspector presentation, `traceFormat.ts` (44) the pure formatters,
  and `types.ts` (137) the existing trace contracts.

Keep CSS imports in their existing traversal order; importing through a new
barrel can change the cascade. The first pass moved 15 stylesheets byte-for-byte.
The continuation below splits four sheets into ordered imports, preserving
their rule content and order. Both complete production CSS bundles retain their
baseline SHA-256 hashes (recorded in [the organization review](tasks/todo.md)).
There were no relative CSS URLs in these stylesheets to rewrite.

Verification: all 61 existing Node tests and both TypeScript/Vite builds pass.
Every governed frontend folder is within 15 direct files. This is incremental
file-size cleanup: 16 pre-existing oversized source/test files remain across
the two roots (down from 25), including `App.tsx` (1663),
`PromptSuiteStudio.tsx` (883), `DecisionSpotChecks.tsx` (815), and large CSS files.
Relocations and import-only edits do not make those files line-limit compliant.

### Continuation: seven additional file limits closed

| Entry file | Before | After | Extracted implementation |
| --- | ---: | ---: | --- |
| `board/HexBoard.tsx` | 756 | 224 | `boardPresentation.ts` (224), `BoardTerrain.tsx` (178), `BoardAnnotations.tsx` (184) |
| `controls/BoardControlsDock.tsx` | 317 | 198 | `ReplayPlaybackActions.tsx` (121) |
| `controls/GameControls.tsx` | 337 | 292 | `NativeReasoningControl.tsx` (39), `modelPresets.ts` (24) |
| `board/PlayerInfo.css` | 338 | 2 imports | `board/styles/player-details.css` (186), `player-overlay.css` (151) |
| `controls/BoardControlsDock.css` | 302 | 3 imports | `controls/styles/dock-{layout,actions,responsive}.css` (140/127/33) |
| Eval `reasoning/ReasoningTraces.css` | 916 | 5 imports | `reasoning/styles/{overview,navigation,inspector,content,responsive}.css` (253/165/179/208/107) |
| Eval `SftDataExplorer.css` | 1005 | 6 imports | `sft/styles/{overview,readiness,workbench,example,distributions-spatial,responsive}.css` (118/185/213/215/212/57) |

The CSS import entrypoints retain their existing paths and traverse contiguous
original rule ranges. Every piece is below 300 lines; new style folders contain
2, 3, 5, and 6 direct files. HexBoard retains hover/toggle state, click handlers,
view bounds, and SVG layer order. Playback seek state stays inside the extracted
component, with its existing remount key retained by the parent; native reasoning
remains a controlled input.

## Vite template notes

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) (or [oxc](https://oxc.rs) when used in [rolldown-vite](https://vite.dev/guide/rolldown)) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```
