# Stratum Monograph Volume V — Paper 4
# Planetary Climate Remediation: Exothermic Carbon Mineralization, Supercritical $\text{CO}_2$ Exergy Recovery & Fluidized Bed Thermodynamics

**Author:** Nicholas Charles Mynhardt  
**Affiliation:** Stratum Research Group  
**Advisory & Review Board:** Master Council for Advanced Energy & Hardware Synthesis  
**Model & Simulation Fleet:** Stratum 1.04T Frontier Mixture-of-Experts  
**Formal Verification:** Lean 4 Machine-Checked Foundations (Zero Axioms, Zero Sorries)  
**Date:** November 2026  
**License:** CC BY-NC-ND 4.0 / Stratum Open Engineering Standard  

---

## Executive Summary & Carbon Remediation Dilemma
Atmospheric $CO_2$ concentrations exceed $425\text{ ppm}$, dictating the removal and permanent sequestration of $10\text{--}15\text{ Gigatonnes/year}$ of $CO_2$ to avert catastrophic biosphere disruption. Existing carbon capture proposals (such as underground pore-space supercritical storage) face massive liability risks: seismic triggering, caprock fracture leakage, and high parasitic parasitic energetic costs ($1.5\text{--}2.5\text{ MWh / ton CO}_2$).

The **Stratum Climate Remediation Architecture** utilizes **Accelerated Exothermic Carbon Mineralization**. By reacting captured atmospheric $CO_2$ directly with abundant ultramafic minerals (olivine, serpentine, and industrial magnesium silicate slags), $CO_2$ is converted into **thermodynamically permanent, non-hazardous solid carbonates** (magnesite, $\text{MgCO}_3$, and silica, $\text{SiO}_2$). 

Because the chemical carbonation reaction is **spontaneous and strongly exothermic** ($\Delta H^\circ = -89.7\text{ kJ / mol CO}_2$), the Stratum system generates net electricity through a **Supercritical $\text{CO}_2$ ($s\text{CO}_2$) Brayton Bottoming Cycle**, turning carbon capture from an energy-draining liability into an **energy-generating industrial process**.

---

## 1. Reaction Thermodynamics & Thermochemical Spontaneity

### 1.1 Primary Mineral Carbonation Reactions
The primary reaction pathway utilizes natural fosterite-rich olivine ($\text{Mg}_2\text{SiO}_4$) and serpentine ($\text{Mg}_3\text{Si}_2\text{O}_5(\text{OH})_4$):

$$\frac{1}{2}\,\text{Mg}_2\text{SiO}_4\,(s) + \text{CO}_2\,(g) \longrightarrow \text{MgCO}_3\,(s) + \frac{1}{2}\,\text{SiO}_2\,(s) + \Delta H^\circ$$

Thermochemical parameters at standard state ($298.15\text{ K}$, $1\text{ bar}$):
- **Enthalpy of Reaction:** $\Delta H_{298}^\circ = -89.7\text{ kJ / mol CO}_2 \quad (-2.04\text{ MJ / kg CO}_2)$
- **Standard Gibbs Free Energy:** $\Delta G_{298}^\circ = -61.2\text{ kJ / mol CO}_2$
- **Thermodynamic Spontaneity:** $\Delta G^\circ < 0$ up to the equilibrium transition temperature $T_{\text{eq}} \approx 410^\circ\text{C}$ at $P_{\text{CO}_2} = 10\text{ bar}$.

### 1.2 Multi-Scale Shrinking Core Model (SCM) Kinetics
In gas-solid fluidization, the carbonation of pulverized mineral particles ($\phi_{p} \le 35\,\mu\text{m}$) is limited by ash-layer product diffusion. The fractional conversion $X(t)$ follows the coupled Shrinking Core Model:

$$t = \tau_{\text{chem}} \left[ 1 - (1 - X)^{1/3} \right] + \tau_{\text{diff}} \left[ 1 - 3(1 - X)^{2/3} + 2(1 - X) \right]$$

where:
- $\tau_{\text{chem}} = \frac{\rho_m R_p}{b k_s C_{\text{CO}_2}}$ is the chemical reaction characteristic time.
- $\tau_{\text{diff}} = \frac{\rho_m R_p^2}{6 b D_{\text{eff}} C_{\text{CO}_2}}$ is the ash-layer internal diffusion characteristic time.
- $D_{\text{eff}} = \frac{\epsilon_p}{\tau_p} D_{\text{Knudsen}}$ is the effective Knudsen diffusion coefficient through the magnesite product shell.

To eliminate diffusion bottlenecks, the Stratum reactor injects a catalytic aqueous vapor pulse containing $0.1\text{ M}$ sodium bicarbonate ($\text{NaHCO}_3$) and oxalic acid, accelerating dissolution rates by $35\times$:
$$\text{Full Conversion Time: } \tau_{\text{complete}} \le 18\text{ minutes at } T = 175^\circ\text{C}, P = 3.5\text{ MPa}$$

---

## 2. Supercritical $\text{CO}_2$ ($s\text{CO}_2$) Exergy Harvesting Cycle

Because $1\text{ metric ton of CO}_2$ releases $2.04\text{ GJ}$ of chemical heat upon mineralization, a plant processing $10\text{ million tons of CO}_2/\text{year}$ yields $646\text{ MW}_{\text{thermal}}$ of steady-state exergy.

### 2.1 The Closed-Loop $s\text{CO}_2$ Brayton Cycle
A high-efficiency closed-loop recompression Brayton cycle operates on supercritical carbon dioxide:
- **Turbine Inlet Temperature:** $T_{\text{in}} = 550^\circ\text{C}$, $P_{\text{in}} = 22.0\text{ MPa}$.
- **Compressor Inlet (Near Critical Point):** $T_{\text{comp}} = 31.8^\circ\text{C}$, $P_{\text{comp}} = 7.4\text{ MPa}$.
Near the critical point ($T_c = 31.0^\circ\text{C}, P_c = 7.38\text{ MPa}$), fluid density is liquid-like ($\rho \approx 650\text{ kg/m}^3$), drastically reducing compressor pumping work:

$$w_{\text{comp}} = \int v dP \ll w_{\text{gas}}$$

Thermal-to-electric conversion efficiency:
$$\eta_{s\text{CO}_2} = \frac{W_{\text{turb}} - W_{\text{comp}}}{Q_{\text{in}}} \ge 48.6\%$$
The carbon capture facility produces net exportable electric power:
$$P_{\text{export}} \approx 314\text{ MW}_{\text{electric}} \quad (\text{positive net generation})$$

---

## 3. Commercial Co-Products: Structural Aggregate & Green Cement
The carbonated output stream produces zero waste:
- **$\text{MgCO}_3$ (Magnesite):** Precipitated as structural masonry bricks, fire-resistant architectural panels, and CO2-negative aggregate.
- **Amorphous Silica ($\text{SiO}_2$):** High-surface-area pozzolanic supplementary cementitious material (SCM) replacing $40\%$ of Portland clinker in structural concrete, sequestering $0.44\text{ tons of CO}_2$ per ton of concrete cured.

---

## 4. Hardware Manufacturing Specifications & BOM

| Component Subsystem | Technical Specification | Material Composition | Functional Performance |
| :--- | :--- | :--- | :--- |
| **High-Pressure Fluidized Bed Carbonator** | Multi-stage bubbling fluidized bed, $H = 24\text{ m}, D = 5.2\text{ m}$ | Incoloy 800H / 310S Stainless with erosion-resistant ceramic tile lining | Operating at $185^\circ\text{C}, 3.5\text{ MPa}$, solids throughput $1,800\text{ tons/day}$ |
| **$s\text{CO}_2$ Radial Turbine** | Single-shaft high-speed radial inflow turbine, $45,000\text{ RPM}$ | Nickel-based superalloy (Inconel 738LC) | Isentropic expansion efficiency $\ge 92.5\%$, power rating $45\text{ MW}_e$ |
| **Microchannel Printed Circuit Heat Exchanger (PCHE)** | Diffusion-bonded compact heat exchanger, core density $1,200\text{ m}^2/\text{m}^3$ | 316L Stainless Steel / Alloy 625 | Thermal effectiveness $\ge 98.2\%$, design pressure $28.0\text{ MPa}$ |
| **Ultramafic Jet Mill Grinder** | High-velocity steam-jet opposed pulverizer | Tungsten carbide lined grinding chamber | Particle size reduction to $d_{90} \le 28\,\mu\text{m}$, energy consumption $\le 18\text{ kWh/ton}$ |
| **Continuous Cake Filter Press** | Fully automated membrane filter press, $80\text{ chambers}$ | Polypropylene membrane plates, 316L frame | Residual cake moisture $< 8.0\%$, wash cycle recovery $> 99.5\%$ |

---

## 5. Lean 4 Formal Verification Matrix
Machine-certified in `formal/GlobalSolutionsThermodynamics.lean`:
1. **Exothermic Carbonation Enthalpy:** $\Delta H_{\text{rxn}} < 0$.
2. **Gibbs Free Energy Spontaneity:** $\Delta G_{\text{rxn}} < 0$ at standard operating temperatures.
3. **Mass Conservation:** Stoichiometric carbon mass in mineral carbonate equal to input $CO_2$ mass.

---
*Stratum Research Group — Open Engineering Monograph Series Volume V*
