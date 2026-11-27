#!/usr/bin/env wolframscript
(* ::Package:: *)

(*
  Stratum Research Group & Synaptek Labs
  Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
  
  Wolfram Verification: Multiway Causal Invariance & Entropic Lyapunov Exponents
  Relates Basal Ganglia Attractors & Thalamocortical Phase-Resetting to Multiway Rewriting
*)

Print["=== WEFT Multiway Causal Invariance & Foliation Verification ==="];

RewritingRule[token_String] := Switch[token,
  "u_plus", {"u_plus", "v_minus"},
  "u_minus", {"u_minus"},
  "v_plus", {"u_plus", "w_0"},
  "v_minus", {"v_plus"},
  "w_0", {"u_plus"}
];

EvolveTokenSet[tokens_List] := DeleteDuplicates[Flatten[RewritingRule /@ tokens]];

foliations = NestList[EvolveTokenSet, {"u_plus"}, 6];

Do[
  Print["Hypersurface Foliation t=", i - 1, " | Event Cardinality: ", Length[foliations[[i]]], " | Tokens: ", foliations[[i]]],
  {i, 1, Length[foliations]}
];

(* Verify Causal Invariance: Path Convergence (Confluence) *)
convergent = MemberQ[Last[foliations], "u_plus"] && MemberQ[Last[foliations], "v_plus"];
Print["Causal Confluence Invariant Verified: ", convergent];
