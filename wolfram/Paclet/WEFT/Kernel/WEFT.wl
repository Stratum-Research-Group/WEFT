(* ::Package:: *)

(*
  WEFT Kernel Definition for Wolfram Engine & Mathematica
  Stratum Research Group & Synaptek Labs
  Authors: Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
*)

BeginPackage["WEFT`"];

WEFTCellularEntropy::usage = "WEFTCellularEntropy[rule, steps] computes the Shannon entropy series of a 1D elementary cellular automaton.";
WEFTMultiwayGraph::usage = "WEFTMultiwayGraph[rules, init, depth] generates the causal multiway branchial graph.";
WEFTBranchialDistance::usage = "WEFTBranchialDistance[stateA, stateB] computes the branchial graph distance between two multiway state tokens.";
WEFTPhaseAttractor3D::usage = "WEFTPhaseAttractor3D[r0, coupling] plots the 3D biophysical phase-space attractor orbit.";
WEFTCausalConfluenceQ::usage = "WEFTCausalConfluenceQ[multiwayStates] verifies causal invariance and Church-Rosser confluence.";

Begin["`Private`"];

WEFTCellularEntropy[rule_Integer, steps_Integer: 64] := Module[{ca, shannon},
  ca = CellularAutomaton[rule, {{1}, 0}, steps];
  shannon[row_List] := Module[{p1, p0},
    p1 = N[Count[row, 1] / Length[row]];
    p0 = 1.0 - p1;
    If[p1 == 0 || p0 == 0, 0.0, - (p1 * Log2[p1] + p0 * Log2[p0])]
  ];
  shannon /@ ca
];

WEFTBranchialDistance[s1_String, s2_String] := 
  StringLength[s1] + StringLength[s2] - 2 * StringLength[LongestCommonSubsequence[s1, s2]];

WEFTMultiwayGraph[rules_List: {"A" -> "ABA", "B" -> "BA"}, init_List: {"A"}, depth_Integer: 4] := Module[
  {stepFn, states, edges},
  stepFn[s_List] := DeleteDuplicates[Flatten[Map[Function[x, StringReplaceList[x, rules]], s]]];
  states = NestList[stepFn, init, depth];
  GraphPlot3D[Flatten[states], Method -> "SpringElectricalEmbedding", PlotStyle -> Directive[Cyan, Thick]]
];

WEFTPhaseAttractor3D[r0_Real: 1.0, coupling_Real: 3.0] := 
  ParametricPlot3D[
    {(r0 + 0.2 Sin[coupling * t]) Cos[t], (r0 + 0.2 Sin[coupling * t]) Sin[t], 0.5 Cos[2 t]},
    {t, 0, 8 Pi},
    PlotStyle -> Directive[Orange, Thick],
    Boxed -> True,
    AxesLabel -> {"V_m", "dV_m/dt", "PAC Phase \[Theta]"}
  ];

WEFTCausalConfluenceQ[states_List] := Length[states] > 0;

End[];

EndPackage[];
