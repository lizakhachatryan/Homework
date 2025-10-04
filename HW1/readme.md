# Innovation Diffusion Analysis – TIME’s Best Innovations (2024)

**Step 1 – Chosen Innovation:**  
**ASUS Zenbook Duo (2024)** — TIME Best Inventions 2024:  
  


This project analyzes the diffusion of the ASUS Zenbook Duo using the Bass Model.  
It compares the innovation with a similar product from the past to estimate adoption parameters and predict diffusion over time.

---

## Step 2 – Similar Innovation from the Past

**Look-alike:** Lenovo Yoga Book 9i (2023), a dual-screen OLED laptop.  

The ASUS Zenbook Duo (2024) and Lenovo Yoga Book 9i (2023) both represent the evolution of portable dual-display computing. Functionally, each device integrates two full-sized 14-inch OLED touchscreens that can operate as a traditional laptop, a tablet, or an extended multitasking workspace. This flexible design allows users to switch between creative, professional, and entertainment modes, positioning both products at the intersection of productivity and design innovation.

Technologically, the Yoga Book 9i was the first widely marketed dual-screen OLED laptop, establishing a proof of concept for this form factor. The ASUS Zenbook Duo builds on that foundation with more powerful Intel Core Ultra processors, improved battery life, and enhanced hinge and keyboard ergonomics. From a market perspective, both target premium laptop consumers—professionals, designers, and early adopters—who are willing to pay a higher price for cutting-edge features. The Yoga Book 9i’s modest but steady adoption provides a realistic diffusion pattern for forecasting the Zenbook Duo’s future market trajectory.


---

## Directory Details

**img/**
- `lookalike_fit.png`: Bass fit vs actual (look-alike series).  
- `zenbook_duo_annual.png`: Annual adopters (scenarios).  
- `zenbook_duo_cumulative.png`: Cumulative adopters (scenarios).

**data/**
- `2_in_1_tablet_shipments_forecast.xlsx`: Statista/IDC source file.  
- `lookalike_timeseries.csv`: Cleaned time series used to fit Bass model (millions).  
- `zenbook_duo_forecast.csv`: Forecast results for ASUS Zenbook Duo (millions), 2025–2035.

**report/**
- `report_source.md`: Main report (Steps 1–7).  
- `report.pdf`: Final exported PDF version.

---

### Project Overview

1. Load real-world shipment data for 2-in-1 devices (2013–2019) from Statista (IDC).
2. Estimate Bass model parameters (`p`, `q`, `M`) for the look-alike innovation.
3. Transfer parameters to forecast ASUS Zenbook Duo diffusion (2025–2035).
4. Produce figures, CSVs, and a final report in Markdown and PDF formats.

---

### References
1. TIME. *A Double-Screen Laptop (ASUS Zenbook Duo)*, Best Inventions 2024.  
2. Statista (based on IDC). *Tablet and 2-in-1 tablet unit shipments forecast worldwide from 2013 to 2019 (in millions).* Accessed Oct 2025.
