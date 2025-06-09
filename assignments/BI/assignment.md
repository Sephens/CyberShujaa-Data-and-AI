# **Hotel Business Data Analysis with Power BI**  
**Name:** [Your Name]  
**Program:** [Your Program Name]  
**Date:** [Submission Date]  

---

## **1. Introduction**  
The goal of this assignment is to analyze hotel business data to understand client needs and support decision-making. The tasks include:  
- Loading and transforming datasets (e.g., `dim_date`, `dim_rooms`).  
- Building a **star schema** data model with proper relationships.  
- Creating **DAX measures and columns** for analysis.  
- Designing an **interactive Power BI dashboard**.  
- Publishing the report and documenting the process.  

This report provides a step-by-step walkthrough with screenshots as evidence of completion.  

---

## **2. Task Completion**  

### **2.1 Data Loading & Transformation**  
I imported the following datasets into Power BI:  
- `dim_date` (date dimension)  
- `dim_rooms` (room details)  
- `fact_bookings` (booking transactions)  

**Steps:**  
1. Clicked **"Get Data"** → Selected **Excel/CSV** (depending on source).  
2. Applied transformations in **Power Query Editor**:  
   - Removed duplicates.  
   - Corrected data types (e.g., date formats).  
   - Handled missing values.  

**Screenshot:**  
*(Insert screenshot of Power Query Editor showing applied transformations)*  

---

### **2.2 Building the Data Model (Star Schema)**  
I structured the data model with:  
- **Fact table:** `fact_bookings` (central table with transactional data).  
- **Dimension tables:** `dim_date`, `dim_rooms`, `dim_customers` (linked via relationships).  

**Relationships:**  
- `fact_bookings[room_id]` → `dim_rooms[room_id]` (One-to-Many)  
- `fact_bookings[booking_date]` → `dim_date[date]` (One-to-Many)  

**Screenshot:**  
*(Insert screenshot of the Model View showing star schema relationships)*  

---

### **2.3 DAX Measures & Columns**  
I created the following **DAX** calculations:  

#### **New Columns:**  
1. **`Revenue`** (for each booking):  
   ```dax
   Revenue = fact_bookings[price_per_night] * fact_bookings[stay_duration]  
   ```

#### **Key Measures:**  
2. **`Total Revenue`**:  
   ```dax
   Total Revenue = SUM(fact_bookings[Revenue])  
   ```
3. **`Occupancy Rate`**:  
   ```dax
   Occupancy Rate = 
   DIVIDE(
       COUNTROWS(fact_bookings),
       COUNTROWS(dim_rooms),
       0
   )
   ```
4. **`Average Stay Duration`**:  
   ```dax
   Avg Stay Duration = AVERAGE(fact_bookings[stay_duration])  
   ```

**Screenshot:**  
*(Insert screenshot of DAX formulas in Power BI)*  

---

### **2.4 Interactive Dashboard**  
I designed a **4-page dashboard** with:  
- **Revenue Trends** (Line chart by month).  
- **Room Occupancy** (Heatmap by room type).  
- **Customer Segmentation** (Pie chart by customer type).  
- **Filters** (Date range, room category).  

**Screenshot:**  
*(Insert dashboard screenshot with visuals and filters)*  

---

### **2.5 Publishing & Sharing**  
1. Clicked **"Publish"** to Power BI Service.  
2. Shared the report with **public access**.  
3. Embedded it in my **[portfolio/blog link]**.  

**Screenshot:**  
*(Insert screenshot of published report in Power BI Service)*  

---

## **3. Conclusion**  
This project helped me:  
✅ Understand hotel business analytics.  
✅ Apply **ETL (Extract, Transform, Load)** techniques.  
✅ Build a **star schema** data model.  
✅ Create **DAX measures** for KPIs.  
✅ Design an **interactive Power BI dashboard**.  

**Final Dashboard Link:** [Insert Power BI Public Link]  
**Portfolio Link:** [Insert Your Portfolio/Blog URL]  

---

**Submission:**  
- **PDF Report** (with screenshots and links).  
- **Power BI Public Link** (shared with permissions).  

--- 

**End of Report**  

---

### **Notes for Submission:**  
- Replace **[Your Name]**, **[Program]**, and placeholder links with actual details.  
- Ensure all screenshots are clear and relevant.  
- Verify that the Power BI report is **publicly accessible**.  

Would you like any refinements in the structure or additional details?