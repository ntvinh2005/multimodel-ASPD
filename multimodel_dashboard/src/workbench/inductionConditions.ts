export type InductionCondition = "induction" | "counterfactual" | "no_match" | "repeated_no_association";
export const CONDITIONS: InductionCondition[] = ["induction", "counterfactual", "no_match", "repeated_no_association"];
export const CONDITION_LABEL: Record<InductionCondition, string> = {
  induction: "Induction",
  counterfactual: "Counterfactual (+)",
  no_match: "No prior match",
  repeated_no_association: "Repeated / wrong association",
};
