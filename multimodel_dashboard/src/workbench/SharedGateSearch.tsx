export default function SharedGateSearch({ value, onChange, onSelectFirst, label = "Find a shared gate", placeholder = "MID:6608 or 6608" }: {
  value: string;
  onChange: (query: string) => void;
  onSelectFirst: () => void;
  label?: string;
  placeholder?: string;
}) {
  return <label className="uw-gate-search">
    {label}
    <input
      type="search"
      value={value}
      placeholder={placeholder}
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
