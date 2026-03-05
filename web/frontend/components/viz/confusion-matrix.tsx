"use client";

interface ConfusionMatrixProps {
  matrix: number[][];
  classNames?: string[];
}

export function ConfusionMatrix({ matrix, classNames }: ConfusionMatrixProps) {
  if (!matrix || matrix.length === 0) return null;

  const labels = classNames || matrix.map((_, i) => `Class ${i}`);
  const maxVal = Math.max(...matrix.flat());

  return (
    <div className="overflow-x-auto">
      <h3 className="text-lg font-semibold mb-2">Confusion Matrix</h3>
      <table className="border-collapse text-sm">
        <thead>
          <tr>
            <th className="p-2 border border-gray-300 bg-gray-50"></th>
            <th className="p-2 border border-gray-300 bg-gray-50 text-center" colSpan={labels.length}>
              Predicted
            </th>
          </tr>
          <tr>
            <th className="p-2 border border-gray-300 bg-gray-50"></th>
            {labels.map((label) => (
              <th key={label} className="p-2 border border-gray-300 bg-gray-50 text-center min-w-[60px]">
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th className="p-2 border border-gray-300 bg-gray-50 text-right">
                {i === 0 && <span className="text-xs text-gray-500 block">Actual</span>}
                {labels[i]}
              </th>
              {row.map((val, j) => {
                const intensity = maxVal > 0 ? val / maxVal : 0;
                const isDiagonal = i === j;
                const bg = isDiagonal
                  ? `rgba(34, 197, 94, ${0.1 + intensity * 0.6})`
                  : `rgba(239, 68, 68, ${intensity * 0.5})`;
                return (
                  <td
                    key={j}
                    className="p-2 border border-gray-300 text-center font-mono"
                    style={{ backgroundColor: bg }}
                  >
                    {val}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
