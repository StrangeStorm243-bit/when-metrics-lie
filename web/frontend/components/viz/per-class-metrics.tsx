"use client";

interface PerClassMetricsProps {
  metrics: Record<string, { precision: number; recall: number; f1: number; support: number }>;
  classNames?: string[];
}

export function PerClassMetrics({ metrics, classNames }: PerClassMetricsProps) {
  if (!metrics || Object.keys(metrics).length === 0) return null;

  const classes = classNames || Object.keys(metrics).sort();

  return (
    <div className="overflow-x-auto">
      <h3 className="text-lg font-semibold mb-2">Per-Class Metrics</h3>
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="bg-gray-50">
            <th className="p-2 border border-gray-300 text-left">Class</th>
            <th className="p-2 border border-gray-300 text-right">Precision</th>
            <th className="p-2 border border-gray-300 text-right">Recall</th>
            <th className="p-2 border border-gray-300 text-right">F1</th>
            <th className="p-2 border border-gray-300 text-right">Support</th>
          </tr>
        </thead>
        <tbody>
          {classes.map((cls) => {
            const m = metrics[cls];
            if (!m) return null;
            return (
              <tr key={cls} className="hover:bg-gray-50">
                <td className="p-2 border border-gray-300 font-medium">{cls}</td>
                <td className="p-2 border border-gray-300 text-right font-mono">
                  {m.precision.toFixed(3)}
                </td>
                <td className="p-2 border border-gray-300 text-right font-mono">
                  {m.recall.toFixed(3)}
                </td>
                <td className="p-2 border border-gray-300 text-right font-mono">
                  <span style={{ color: m.f1 >= 0.7 ? "#16a34a" : m.f1 >= 0.5 ? "#d97706" : "#dc2626" }}>
                    {m.f1.toFixed(3)}
                  </span>
                </td>
                <td className="p-2 border border-gray-300 text-right font-mono">{m.support}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
