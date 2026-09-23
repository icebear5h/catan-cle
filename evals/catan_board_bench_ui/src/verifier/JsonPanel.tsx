export default function JsonPanel({ title, value }: { title: string; value: unknown }) {
  return (
    <section className="json-panel panel-surface">
      <div className="panel-heading">
        <span>{title}</span>
        <span>oracle</span>
      </div>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </section>
  );
}
