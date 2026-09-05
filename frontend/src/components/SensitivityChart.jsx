import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function SensitivityChart({ curve }) {
  return (
    <>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={curve.points}>
          <CartesianGrid strokeDasharray="3 3" stroke="#eaecf0" />
          <XAxis dataKey="gamma" label={{ value: "Γ (hidden bias)", position: "insideBottom", offset: -4 }} />
          <YAxis domain={[0, 1]} />
          <Tooltip formatter={(v) => v.toFixed(3)} />
          <ReferenceLine y={0.05} stroke="#f04438" strokeDasharray="4 4" label={{ value: "p=0.05", fill: "#f04438", fontSize: 12 }} />
          <Line type="monotone" dataKey="p_upper" stroke="#4f46e5" strokeWidth={2} dot isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
      <p style={{ fontSize: 13, color: "#667085" }}>
        Breakpoint Γ: {curve.breakpoint_gamma ?? "> " + curve.bias_robustness} — hidden confounding
        of this magnitude would be needed to erase significance.
      </p>
    </>
  );
}
