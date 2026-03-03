# Predictive Maintenance Across Air and Sea Platform Domains

**ADS-599 Capstone Project | University of San Diego — Applied Data Science**

**Team:** Arjun Venkatesh · Duy-Anh Dang · Jorge Roldan

---

## Project Overview

The U.S. Department of War spends over $50 billion annually on aviation maintenance alone; $80-100 billion when including Naval maintenance. Unplanned equipment failures across military platforms impose severe operational and financial consequences. This project builds a predictive maintenance (PdM) system that uses sensor telemetry to forecast equipment degradation before failure occurs — directly supporting the DoD's Condition-Based Maintenance Plus (CBM+) initiative.

**Research Question:**
> Can a unified deep learning pipeline accurately predict equipment degradation across two structurally different military platform domains — aerospace turbofan engines and naval gas turbine propulsion systems — using only sensor telemetry data?

---

## Platform Domains & Datasets

| Domain | Dataset | Platform | Task |
|--------|---------|----------|------|
| Aerospace | NASA CMAPSS | Turbofan jet engine | RUL prediction |
| Naval | UCI Naval Propulsion Plants | Navy Frigate Gas Turbine | Degradation state (kMc, kMt) |

### NASA CMAPSS — `Data/CMaps/`

Simulated run-to-failure data for turbofan jet engines under varying operating conditions and fault modes. Four sub-datasets (FD001–FD004) with increasing complexity.

| File | Description |
|------|-------------|
| `train_FD00X.txt` | Full engine run-to-failure sequences |
| `test_FD00X.txt` | Truncated sequences — model predicts RUL |
| `RUL_FD00X.txt` | Ground-truth RUL for each test engine |

Each file: 26 columns — `unit_id`, `cycle`, 3 operational settings, 21 sensor readings.
Reference: Saxena et al. (2008). See `Data/CMaps/Damage Propagation Modeling.pdf`.

### UCI Naval Propulsion Plants — `Data/UCI CBM Dataset/`

Sensor data from a Navy Frigate CODLAG gas turbine propulsion system, generated from a real-data validated simulator (confidentiality constraints with the Navy required use of the simulator rather than raw operational data).

| File | Description |
|------|-------------|
| `data.txt` | 11,934 rows × 18 columns (space-delimited, no header) |
| `Features.txt` | Column name reference |

Columns 1–16: sensor features. Column 17: `kMc` (GT Compressor decay). Column 18: `kMt` (GT Turbine decay).

---

## Repository Structure (To be Updated)

```
ADS599/
│
├── Data/
│   ├── CMaps/                   ← NASA CMAPSS (FD001–FD004)
│   └── UCI CBM Dataset/         ← UCI Naval Propulsion Plants
│
├── code_library/                ← Functions called by the main notebook
├── images/                      ← Figures generated during analysis
├── other_material/              ← Requirements, deployment apps
│
└── main_notebook.ipynb          ← Main project notebook (in progress)
```

---

## References

Coraddu, A., Oneto, L., Ghio, A., Savio, S., Anguita, D., & Figari, M. (2014). *Condition based
maintenance of naval propulsion plants* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5K31K

Coraddu, A., Oneto, L., Ghio, A., Savio, S., Anguita, D., & Figari, M. (2016). Machine learning
approaches for improving condition-based maintenance of naval propulsion plants. *Proceedings of
the Institution of Mechanical Engineers, Part M: Journal of Engineering for the Maritime
Environment, 230*(1), 136–153. https://doi.org/10.1177/1475090214540874

Cipollini, F., Oneto, L., Coraddu, A., Murphy, A. J., & Anguita, D. (2018). Condition-based
maintenance of naval propulsion systems with supervised data analysis. *Ocean Engineering, 149*,
268–278. https://doi.org/10.1016/j.oceaneng.2017.12.002

Li, X., Ding, Q., & Sun, J. Q. (2018). Remaining useful life estimation in prognostics using deep
convolution neural networks. *Reliability Engineering & System Safety, 172*, 1–11.
https://doi.org/10.1016/j.ress.2017.11.021

Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation modeling for aircraft
engine run-to-failure simulation. *2008 International Conference on Prognostics and Health
Management* (pp. 1–9). IEEE. https://doi.org/10.1109/PHM.2008.4711414

Saxena, A., & Goebel, K. (2008). *CMAPSS turbofan engine degradation simulation data set*
[Dataset]. NASA Ames Prognostics Data Repository.
https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6

Zhang, C., Lim, P., Qin, A. K., & Tan, K. C. (2017). Multiobjective deep belief networks ensemble
for remaining useful life estimation in prognostics. *IEEE Transactions on Neural Networks and
Learning Systems, 28*(10), 2306–2318. https://doi.org/10.1109/TNNLS.2016.2582798