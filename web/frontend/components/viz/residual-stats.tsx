"use client";

interface ResidualStatsProps {
  stats: {
    mean: number;
    std: number;
    min: number;
    max: number;
    median: number;
    mae: number;
    rmse: number;
  };
}

export function ResidualStats({ stats }: ResidualStatsProps) {
  if (!stats) return null;

  const items = [
    { label: "MAE", value: stats.mae, description: "Mean Absolute Error" },
    { label: "RMSE", value: stats.rmse, description: "Root Mean Squared Error" },
    { label: "Mean Residual", value: stats.mean, description: "Avg prediction error (bias)" },
    { label: "Std Residual", value: stats.std, description: "Spread of errors" },
    { label: "Median Residual", value: stats.median, description: "Middle prediction error" },
    { label: "Min Residual", value: stats.min, description: "Largest underprediction" },
    { label: "Max Residual", value: stats.max, description: "Largest overprediction" },
  ];

  return (
    <div>
      <h3 className="text-lg font-semibold mb-2">Regression Diagnostics</h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {items.map((item) => (
          <div key={item.label} className="bg-white border border-gray-200 rounded-lg p-3">
            <div className="text-xs text-gray-500 uppercase tracking-wide">{item.label}</div>
            <div className="text-xl font-mono font-bold mt-1">
              {item.value >= 0 ? "+" : ""}{item.value.toFixed(4)}
            </div>
            <div className="text-xs text-gray-400 mt-1">{item.description}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
