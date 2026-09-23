import type { EvalComparisonResponse, EvalModelSummary } from './types';

export default function EvalComparisonPanel({ data }: { data: EvalComparisonResponse }) {
  return (
    <section className="eval-comparison panel-surface">
      <div className="panel-heading">
        <span>Eval Model Comparison</span>
        <span>Qwen 3 vs 3.5 vs 3.6 Flash</span>
      </div>

      <div className="comparison-overview">
        {data.models.map((model) => (
          <article className="comparison-card" key={model.model}>
            <h4>{model.display_name}</h4>
            <div className="comparison-meta-grid">
              <span>Exact</span>
              <strong>{formatPercent(model.exact_accuracy)}</strong>
              <span>Component</span>
              <strong>{formatPercent(model.component_accuracy)}</strong>
              <span>Latency</span>
              <strong>{formatMs(model.avg_latency_ms)}</strong>
              <span>Errors</span>
              <strong>{model.errors}</strong>
            </div>
          </article>
        ))}
      </div>

      <div className="comparison-categories">
        {data.categories.map((category) => (
          <CategoryBarRow key={category} category={category} models={data.models} />
        ))}
      </div>
    </section>
  );
}

function CategoryBarRow({ category, models }: { category: string; models: EvalModelSummary[] }) {
  const label = category.replace(/_/g, ' ');

  return (
    <article className="comparison-category-row">
      <h5>{label}</h5>
      <div className="comparison-bars">
        {models.map((model, modelIndex) => {
          const categoryScore = model.categories[category]?.exact_accuracy || 0;
          return (
            <div key={`${model.model}-${category}`} className="comparison-bar-wrap">
              <div className="comparison-bar-header">
                <span>{model.display_name}</span>
                <strong>{formatPercent(categoryScore)}</strong>
              </div>
              <div className="comparison-bar-track">
                <span
                  className={`comparison-bar comparison-model-${modelIndex % 3}`}
                  style={{ width: `${Math.round(categoryScore * 100)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </article>
  );
}

function formatPercent(value: number) {
  return `${Math.max(0, Math.round((value || 0) * 1000) / 10).toFixed(1)}%`;
}

function formatMs(value: number) {
  return `${Math.round(value || 0)}ms`;
}
