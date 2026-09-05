import { Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function ATEComparison({ allMethods, trueAte }) {
  const data = Object.entries(allMethods).map(([method, v]) => ({
    method: method.toUpperCase(),
    ate: v.ate,
  }));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#eaecf0" />
        <XAxis dataKey="method" />
        <YAxis />
        <Tooltip formatter={(v) => `$${v.toFixed(2)}`} />
        <ReferenceLine y={trueAte} stroke="#12b76a" strokeDasharray="4 4" label={{ value: "true ATE", fill: "#12b76a", fontSize: 12 }} />
        <Bar dataKey="ate" fill="#4f46e5" radius={[6, 6, 0, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}
