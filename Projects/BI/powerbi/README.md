

# Deep Revenue Insights and Operational Intelligence for the Hospitality Sector Using Power BI

## Introduction

This assignment leverages Power BI to analyze hotel business data, transforming raw datasets (`dim_date`, `dim_rooms`, `fact_bookings`) into actionable insights. Business Intelligence (BI) enables data-driven decision-making by processing, modeling, and visualizing information. Power BI, a leading BI tool, integrates data from multiple sources, builds semantic models, and creates interactive dashboards.

The solution follows a **star schema** for efficient analytics, implements **DAX measures** for KPIs like occupancy and revenue, and delivers user-friendly visualizations. Key benefits include optimized pricing, operational efficiency, and enhanced guest experiences.

The project covers:

* Data preparation
* Modeling
* Dashboard design
* Deployment

It demonstrates BI's value in hospitality management through **descriptive**, **diagnostic**, and **predictive analytics**.

---

## Tasks Completed

### 2.1 Data Loading & Transformation: Building the Foundation for Hotel Analytics
In the Data Loading & Transformation phase, we import key hotel datasets into Power BI from CSV files, then clean and standardize them using Power Query. This includes fixing data types, renaming columns, correcting location data, and ensuring consistency, laying a clean, structured foundation for accurate analysis and insightful hotel business reporting.

![transform1](/Projects/BI/powerbi/screenshots/get_data.png)
![transform1](/Projects/BI/powerbi/screenshots/get_data2.png)

**Data Connection:**

* Selected `Get Data` → `More` → `Folder` option
* Connected to centralized data repository containing all CSV files

Imported five critical datasets into Power BI Desktop:

**Dimensional Tables (Master Reference Data):**

* `dim_date`: Temporal dimension enabling time intelligence analysis
* `dim_rooms`: Room attributes and classifications
* `dim_hotels`: Metadata including locations and amenities

**Fact Tables (Transactional Data):**

* `fact_bookings`: Reservation records
* `fact_aggregated_bookings`: Pre-consolidated metrics for benchmarking

**Data Transformation:**

* Changed cities from Indian to Kenyan in `dim_hotels`
* Standardized date formats
* Established consistent naming conventions
* Promoted first row as headers in `dim_rooms`
* Verified data types in Schema View
* Documented steps in Power Query’s "Applied Steps" pane

![transform1](/Projects/BI/powerbi/screenshots/transform1.png)

![transform1](/Projects/BI/powerbi/screenshots/transform2.png)

![transform1](/Projects/BI/powerbi/screenshots/transform3.png)
---

### 2.2 Architecting the Data Model: A Star Schema Implementation

In this phase, we build a star schema data model by linking a central fact table with related dimension tables through one-to-many relationships. This structure enhances query efficiency, supports intuitive data analysis, and ensures accurate aggregation of metrics, forming the backbone of insightful, hotel-focused business intelligence in Power BI.

Constructed an optimized **star schema** data model for performance and intuitive structure.

**Data Model Structure:**

* **Fact Table:** `fact_bookings`
* **Dimension Tables:** `dim_date`, `dim_rooms`, `dim_hotels`

**Relationships:**

* `fact_bookings[room_id]` → `dim_rooms[room_id]` (One-to-Many)
* `fact_bookings[booking_date]` → `dim_date[date]` (One-to-Many)

![transform1](/Projects/BI/powerbi/screenshots/star_schema.png)

---

### 2.3 DAX Implementation

In the DAX Implementation phase, we create calculated columns and dynamic measures using Data Analysis Expressions (DAX) to generate key business metrics. These include revenue, bookings, occupancy rate, and average ratings. DAX enables responsive, filter-aware calculations, turning raw data into meaningful, interactive insights for hotel performance analysis in Power BI.

**DAX (Data Analysis Expressions)** is used for modeling, calculations, and analytics in Power BI.

**Calculated Columns:**

* Added to `dim_date` to categorize **Weekdays vs Weekends** based on the day number


![transform1](/Projects/BI/powerbi/screenshots/DAX_column.png)

**Measures Created:**

* `Revenue = SUM(fact_bookings[revenue_realized])`
* `Total Bookings = COUNT(fact_bookings[booking_id])`
* `Total Capacity = SUM(fact_aggregated_bookings[capacity])`
* `Total Successful Bookings = SUM(fact_aggregated_bookings[successful_bookings])`
* `Occupancy % = DIVIDE([Total Successful Bookings],[Total Capacity],0)`
* `Average Rating = AVERAGE(fact_bookings[ratings_given])`
* `No of days = DATEDIFF(MIN(dim_date[date]), MAX(dim_date[date]), DAY) + 1`


![transform1](/Projects/BI/powerbi/screenshots/DAX_measure.png)

---

### 2.4 Interactive Dashboard

In the Interactive Dashboard phase, we design a single-page Power BI report with visualizations for revenue, occupancy, booking trends, customer ratings, and cancellations. Filters by city and room type allow user-driven exploration. The dashboard translates complex data into clear, actionable insights for hotel performance monitoring and strategic decision-making.

Designed a **single-page dashboard** with insights into:

* **Filters** by City and Room Type
* **Revenue** (Overall and filtered)
* **Cancellation Rates** by room type and city
* **Customer Satisfaction** (average ratings)
* **Weekend vs Weekday Metrics**
* **Occupancy by Category**
* **Trend Analysis**
* **Property-wise Performance**
* **Booking Platform Analysis**

![transform1](/Projects/BI/powerbi/screenshots/dashboard.png)

---

## Conclusion

This project demonstrates how Power BI transforms raw hotel data into strategic insights through structured modeling, DAX analytics, and dynamic visualizations. By implementing a star schema, calculating key metrics, and building an interactive dashboard, we enable data-driven decisions in hospitality management. The solution enhances operational efficiency, revenue optimization, and customer satisfaction, establishing a scalable business intelligence foundation that empowers stakeholders to respond proactively in an increasingly competitive and data-centric industry.

This Power BI project showcases how **data-driven decision-making** can transform hospitality management and other data-rich sectors.

**End-to-End BI Solution** that addresses:

* Hotel operations
* Revenue management
* Customer experience



---

## 📎 Link to Project File

[GitHub: Deep Revenue Insights in Hospitality (PBIX)](https://github.com/Sephens/CyberShujaa-Data-and-AI/blob/master/assignments%2FBI%2FDeep%20Revenue%20Insights%20in%20Hospitality.pbix)

