export default function BalanceTable({ balance }) {
  const names = Object.keys(balance.before);
  return (
    <table>
      <thead>
        <tr>
          <th>Covariate</th>
          <th>SMD before</th>
          <th>SMD after</th>
        </tr>
      </thead>
      <tbody>
        {names.map((name) => (
          <tr key={name}>
            <td>{name}</td>
            <td>{balance.before[name].toFixed(3)}</td>
            <td>{balance.after ? balance.after[name].toFixed(3) : "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
