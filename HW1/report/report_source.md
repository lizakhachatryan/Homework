# Innovation Diffusion Analysis – ASUS Zenbook Duo (2024)

## Step 1 – Chosen Innovation
**ASUS Zenbook Duo (2024)** — TIME Best Inventions 2024:  

## Step 2 – Similar Innovation from the Past

**Look-alike:** Lenovo Yoga Book 9i (2023), a dual-screen OLED laptop.  

The ASUS Zenbook Duo (2024) and Lenovo Yoga Book 9i (2023) both represent the evolution of portable dual-display computing. Functionally, each device integrates two full-sized 14-inch OLED touchscreens that can operate as a traditional laptop, a tablet, or an extended multitasking workspace. This flexible design allows users to switch between creative, professional, and entertainment modes, positioning both products at the intersection of productivity and design innovation.

---

## Step 3 – Historical Data (Look-Alike)
**Data source:** Statista (based on IDC).  
*Tablet and 2-in-1 tablet unit shipments forecast worldwide from 2013 to 2019 (in millions).*  
Accessed October 2025.  
Stored in `data/2_in_1_tablet_shipments_forecast.xlsx`.

| Year | Shipments (M) |
|------|--------------:|
| 2013 | 220 |
| 2014 | 230 |
| 2015 | 212 |
| 2016 | 245 |
| 2017 | 254 |
| 2018 | 263 |
| 2019 | 239 |

---

## Step 4 – Bass Model Estimation (Look-Alike)

Using the above time series, the Bass diffusion model was estimated by randomized least squares.

| Parameter | Symbol | Value |
|------------|:------:|------:|
| Coefficient of Innovation | p | 0.0541 |
| Coefficient of Imitation | q | 0.1184 |
| Market Potential | M | 3,971.39 million cumulative units |

Goodness-of-fit: **SSE ≈ 948.31**  
A relatively high **p** indicates strong innovation-driven growth in early years, while moderate **q** suggests steady but not viral imitation.  
(See `img/lookalike_fit.png`.)

---

## Step 5 – Predicting Diffusion for ASUS Zenbook Duo (2024)

The parameters **p** and **q** are transferred from the look-alike category.  
Market potential **M** is estimated using Fermi logic for a premium niche.

| Scenario | M (million units) | Interpretation |
|-----------|:----------------:|----------------|
| Low | 0.10 | ~100k cumulative adopters (early enthusiasts, designers) |
| Base | 0.25 | ~250k units (global realistic demand) |
| High | 0.50 | ~500k units (broader premium sub-segment) |

Forecast range: **2025–2035**  
Results saved to `data/zenbook_duo_forecast.csv`.  
Plots saved to `img/zenbook_duo_annual.png` and `img/zenbook_duo_cumulative.png`.

---

## Step 6 – Scope

**Global scope** is used because:
- Both the source data (Statista/IDC) and ASUS sales network are global.
- The diffusion pattern depends on global innovation exposure, not one market.

---

## Step 7 – Estimated Adopters by Period (Base Scenario)

| Year | New Adopters (M) | Cumulative (M) |
|------|-----------------:|---------------:|
| 2025 | 0.0135 | 0.0135 |
| 2026 | 0.0143 | 0.0278 |
| 2027 | 0.0149 | 0.0427 |
| 2028 | 0.0153 | 0.0580 |
| 2029 | 0.0156 | 0.0736 |

(Full 2025–2035 table: `data/zenbook_duo_forecast.csv`.)

---

### References
1. TIME. *A Double-Screen Laptop (ASUS Zenbook Duo)*, Best Inventions 2024.  
2. Statista . *Tablet and 2-in-1 tablet unit shipments forecast worldwide, 2013–2019 (in millions).* Accessed Oct 2025.
