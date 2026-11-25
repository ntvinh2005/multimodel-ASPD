export default function SharedGateSearch({ value, onChange, onSelectFirst }: {
  value: string;
  onChange: (query: string) => void;
  onSelectFirst: () => void;
}) {
  return <label className="uw-gate-search">
    Find a shared gate
    <input
      type="search"
      value={value}
      placeholder="MID:6608 or 6608"
      onChange={(event) => onChange(event.target.value)}
      onKeyDown={(event) => {
        if (event.key === "Enter" && !event.nativeEvent.isComposing && value.trim()) {
          event.preventDefault();
          onSelectFirst();
        }
      }}
    />
  </label>;
}
