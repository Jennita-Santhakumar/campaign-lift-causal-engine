export default function CATEBreakdown({ cate }) {
  return (
    <table>
      <thead>
        <tr>
          <th>Segment</th>
          <th>CATE</th>
          <th>N</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(cate.by_segment).map(([name, v]) => (
          <tr key={name}>
            <td>{name}</td>
            <td>${v.cate.toFixed(2)}</td>
            <td>{v.n}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
