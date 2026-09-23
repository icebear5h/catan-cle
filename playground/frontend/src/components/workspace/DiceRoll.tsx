interface DiceRollProps {
  roll: [number, number];
}

export default function DiceRoll({ roll }: DiceRollProps) {
  return (
    <div style={{
      position: 'absolute',
      top: '74px',
      left: '50%',
      transform: 'translateX(-50%)',
      display: 'flex',
      gap: '8px',
      zIndex: 100,
    }}>
      {roll.map((die, i) => (
        <div key={i} style={{
          width: '48px',
          height: '48px',
          background: '#fff',
          borderRadius: '8px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '24px',
          fontWeight: 'bold',
          color: '#1a1a1a',
          boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
        }}>
          {die}
        </div>
      ))}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        fontSize: '20px',
        fontWeight: 'bold',
        color: '#fff',
        marginLeft: '4px',
      }}>
        = {roll[0] + roll[1]}
      </div>
    </div>
  );
}
