import type { QuestionDetail, Verdict } from './types';

export default function QuestionCard({
  question,
  isActive,
  verdict,
  onSelect,
  onVerdict,
}: {
  question: QuestionDetail;
  isActive: boolean;
  verdict?: Verdict;
  onSelect: () => void;
  onVerdict: (verdict: Verdict) => void;
}) {
  return (
    <article className={`question-card ${isActive ? 'active' : ''}`} onClick={onSelect}>
      <div className="question-card-top">
        <span>{question.category}</span>
        <code>{question.id}</code>
      </div>
      <p>{question.question}</p>
      <div className="question-card-answer">
        <span>Expected</span>
        <code>{question.answer}</code>
      </div>
      <div className="verdict-strip compact" aria-label={`Human verification verdict for ${question.id}`}>
        <button
          className={`verdict-button good ${verdict === 'correct' ? 'active' : ''}`}
          onClick={(event) => {
            event.stopPropagation();
            onVerdict('correct');
          }}
        >
          Correct
        </button>
        <button
          className={`verdict-button bad ${verdict === 'wrong' ? 'active' : ''}`}
          onClick={(event) => {
            event.stopPropagation();
            onVerdict('wrong');
          }}
        >
          Wrong
        </button>
        <button
          className={`verdict-button mute ${verdict === 'unclear' ? 'active' : ''}`}
          onClick={(event) => {
            event.stopPropagation();
            onVerdict('unclear');
          }}
        >
          Unclear
        </button>
      </div>
    </article>
  );
}
