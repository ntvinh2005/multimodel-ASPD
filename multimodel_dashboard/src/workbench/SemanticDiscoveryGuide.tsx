export function SemanticDiscoveryGuide({ example, modelNames, archived = false }: { example?: { left: string; right: string; distance?: number }; modelNames?: string; archived?: boolean }) {
  return <details className="semantic-guide">
    <summary>How to use Semantic Discovery</summary>
    <ol>
      <li><strong>Select a candidate.</strong> Start with a pair that beats its matched controls on held-out documents. Check how many independent documents each component has.</li>
      <li><strong>Read each component’s examples.</strong> Scan several documents. {archived ? "The highlight marks the recorded anchor." : "The outline marks the aligned anchor. Darker shading marks a larger measured contribution."}</li>
      <li><strong>Compare models.</strong> Look for a specific relationship that explains both sets of examples. The two components can respond to different words or appear in different layers.</li>
      <li><strong>Check contrasting examples.</strong> Look for similar contexts with zero or weak contribution and for strong examples that challenge your explanation.</li>
      <li><strong>Inspect dependencies.</strong> Group graphs show available records for group members. Pair graphs show the two selected endpoints. Check the model and anchor for each panel before tracing a path.</li>
      <li><strong>Save a hypothesis.</strong> Write down the observed pattern, a testable explanation, counterexamples, and open questions in Research Notes.</li>
      <li><strong>Validate it.</strong> Test the explanation on new documents. Establish a causal role with a controlled intervention and measure the predicted change.</li>
    </ol>
    {example && <p><strong>Worked example.</strong> Select <code>{example.left}</code> and <code>{example.right}</code> ({modelNames}).
      {" "}Their held-out OT distance is {example.distance?.toPrecision(3) ?? "Unavailable"}.
      {" "}Open a high-contribution document and a matched zero for each component. Use those contexts to propose a rule that predicts both strong and weak contributions.
      {" "}The distance measures context similarity. The examples and any exact-anchor graphs provide observational evidence.</p>}
  </details>;
}
