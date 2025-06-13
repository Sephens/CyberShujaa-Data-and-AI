# Advanced Database Management Systems - Resit TCA Autumn 2022

## Introduction

This report provides a comprehensive solution to the Advanced Database Management Systems Time Constrained Assignment (TCA) for Autumn 2022. The assignment consists of five questions covering various aspects of database management systems, including relational algebra, concurrency control, normalization, distributed databases, and data warehousing. Each question will be addressed in detail with appropriate examples, diagrams, and insights drawn from the provided scenarios.

---

## Question 1: Relational Algebra and Calculus

### a) Relational Algebra Queries

**i) Projection Operation: π[StudentNo, StudentName](Students)**

This operation selects only the StudentNo and StudentName columns from the Students table.

**Result:**
| StudentNo | StudentName   |
|-----------|---------------|
| 2016001   | Dave Cahill   |
| 2016002   | Moses Cramp   |
| 2016003   | Afzal Ahmed   |
| 2016004   | Pauline Weller|

**ii) Selection and Projection: π[StudentNo, Course](σ[FeesDue < 10000](Students))**

This operation first selects students with FeesDue less than 10,000, then projects only the StudentNo and Course columns.

**Result:**
| StudentNo | Course      |
|-----------|-------------|
| 2016001   | BA History  |
| 2016004   | BA English  |

### b) Relational Algebra Expression

To find all student names for students studying BA History who owe more than 5000 in fees:

π[StudentName](σ[Course = "BA History" ∧ FeesDue > 5000](Students))

**Result:**
| StudentName |
|-------------|
| Dave Cahill |

### c) Relational Algebra vs. Relational Calculus

**Similarities:**
1. Both are formal languages for querying relational databases.
2. Both operate on relations (tables) and produce relations as results.
3. Both can express the same set of queries (they are equivalent in expressive power).

**Differences:**
1. **Syntax:** Relational algebra uses procedural operations (σ, π, ⋈, etc.), while relational calculus uses declarative logic (∀, ∃, ∧, ∨).
   
   - Algebra example: π[StudentName](σ[Course="BA History"](Students))
   - Calculus example: {S.StudentName | Students(S) ∧ S.Course = "BA History"}

2. **Approach:** Algebra specifies how to get the result (procedural), while calculus specifies what the result should be (declarative).

3. **Complexity:** Calculus can be more intuitive for complex queries involving existential or universal quantifiers.

### d) Join Operator in Relational Algebra

The join operator (⋈) combines rows from two tables based on a related column. 

**Example:**
Consider an additional table "Enrollments":
| StudentNo | Module     | Grade |
|-----------|------------|-------|
| 2016001   | DBMS       | A     |
| 2016002   | Chemistry  | B     |

To find students and their grades:
Students ⋈[Students.StudentNo = Enrollments.StudentNo] Enrollments

**Result:**
| StudentNo | StudentName   | FeesDue | Course      | Module    | Grade |
|-----------|---------------|---------|-------------|-----------|-------|
| 2016001   | Dave Cahill   | 6000    | BA History  | DBMS      | A     |
| 2016002   | Moses Cramp   | 13000   | BSc Chemistry | Chemistry | B     |

---

## Question 2: Concurrency Control

### a) Lost Update Problem

The lost update problem occurs when two transactions read the same data item and then update it based on the read value, causing one update to be lost.

**Example:**
Transaction T1 reads X (value=100) and adds 50 (X=150).
Transaction T2 reads X (value=100) before T1 commits and subtracts 20 (X=80).
T1 commits (X=150), then T2 commits (X=80), losing T1's update.

**Diagram:**
```
T1: Read X (100) → Write X (100+50=150)
T2:           Read X (100) → Write X (100-20=80)
Time →→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→
Result: X=80 (T1's update is lost)
```

### b) Serialisability in Concurrency Control

Serialisability ensures that concurrent transaction execution produces the same result as some serial execution of those transactions.

**Possible Relationships Between Two Transactions:**
1. **Conflict Serialisability:** Two transactions are conflict serializable if their operations can be reordered to form a serial schedule without changing the final result.
   
   Example: If T1 and T2 only access different data items, they can run concurrently.

2. **View Serialisability:** A weaker form where transactions see the same data values as in some serial execution, even if the order of operations differs.

**Role of Serialisability:**
- Ensures database consistency
- Prevents anomalies like lost updates, dirty reads, or unrepeatable reads
- Allows for concurrent execution while maintaining correctness

### c) User Privileges in SQL

SQL privileges control access to database objects. Example:

```sql
GRANT SELECT, INSERT ON Students TO teacher_role;
REVOKE DELETE ON Students FROM student_role;
```

**Example Scenario:**
A university database might have:
- Professors: Can SELECT, INSERT, UPDATE all tables
- Students: Can only SELECT from Courses table
- Administrators: Full privileges on Student records

This prevents unauthorized access (e.g., students modifying grades) while allowing necessary operations.

---

## Question 3: Normalization

### a) Normalization to 3NF

**Original Form:**
| Pupil Number | Pupil Name | Pupil Type Code | Pupil Type Name | Class ID | Class Name | Date | Price | Instrument Used Code | Instrument Name |
|--------------|------------|-----------------|------------------|----------|------------|------|-------|----------------------|-----------------|
| B451 | Janice Porter | FT | Full Time | C1 | Cello | 01/08/18 | £30.00 | C | Cello |

**1NF (Eliminate Repeating Groups):**
- Pupil Information:
  | PupilNo | PupilName | TypeCode | TypeName |
  |---------|-----------|----------|----------|
  | B451 | Janice Porter | FT | Full Time |

- Classes:
  | PupilNo | ClassID | ClassName | Date | Price | InstCode | InstName |
  |---------|---------|-----------|------|-------|----------|----------|
  | B451 | C1 | Cello | 01/08/18 | £30.00 | C | Cello |

**2NF (Remove Partial Dependencies):**
- Pupil (same as above)
- Instrument:
  | InstCode | InstName |
  |----------|----------|
  | C | Cello |
  | G | Guitar |

- Class:
  | ClassID | ClassName | Price |
  |---------|-----------|-------|
  | C1 | Cello | £30.00 |

- Enrollment:
  | PupilNo | ClassID | Date |
  |---------|---------|------|
  | B451 | C1 | 01/08/18 |

**3NF (Remove Transitive Dependencies):**
- Pupil:
  | PupilNo | PupilName | TypeCode |
  |---------|-----------|----------|
  | B451 | Janice Porter | FT |

- PupilType:
  | TypeCode | TypeName |
  |----------|----------|
  | FT | Full Time |

(Other tables remain same as 2NF)

### b) Multi-valued Dependencies and 4NF

**Multi-valued Dependency (MVD):** 
X →→ Y means for each X value, there is a set of Y values independent of other attributes.

**Example:**
Consider a table:
| Employee | Skill | Language |
|----------|-------|----------|
| John | DBMS | English |
| John | DBMS | Spanish |
| John | Java | English |
| John | Java | Spanish |

Here, Employee →→ Skill and Employee →→ Language (skills and languages are independent).

**Resolution to 4NF:**
Split into two tables:
1. EmployeeSkills:
   | Employee | Skill |
   |----------|-------|
   | John | DBMS |
   | John | Java |

2. EmployeeLanguages:
   | Employee | Language |
   |----------|----------|
   | John | English |
   | John | Spanish |

### c) Beyond 4NF

5NF (Project-Join Normal Form) addresses cases where information can be reconstructed from smaller pieces but not from any single projection.

**Example:**
A table representing "Suppliers supply Parts to Projects":
| Supplier | Part | Project |
|----------|------|---------|
| S1 | P1 | J1 |
| S1 | P2 | J2 |
| S2 | P1 | J1 |

This might appear in 4NF but still have redundancy. 5NF would decompose it into three binary relations showing all valid combinations separately.

---

## Question 4: Distributed Database for Discomania

### Motivations for Distributed Database:

1. **Improved Performance:**
   - Local stores can access customer/sales data quickly from local databases
   - Central stock database ensures consistent inventory across branches

2. **Enhanced Availability:**
   - If one store's system fails, others remain operational
   - Web orders can be routed to nearest available branch

3. **Scalability:**
   - New branches can be added with local databases
   - Web presence can scale separately from physical stores

4. **Data Ownership:**
   - Stores maintain control over their customer data
   - Central office manages overall stock and supplier data

5. **Cost Reduction:**
   - Reduced network traffic (local queries stay local)
   - Can use cheaper hardware for local databases

**Scenario-Specific Benefits:**
- Mail/phone orders can be routed to nearest store with stock
- Web orders can check real-time stock across all branches
- Manufacturing in Southampton can update stock centrally
- Local promotions can use local customer data while corporate can analyze nationwide trends

**Potential Architecture:**
```
[Central DB: Stock, Suppliers]
       ↑
[WAN Connection]
       ↓
[Branch DBs: Customers, Sales] ←→ [Web Storefront]
```

---

## Question 5: Data Warehouse for Construction Company

### Benefits:

1. **Integrated View:**
   - Combine project, worker, subcontractor, and financial data for complete project visibility

2. **Historical Analysis:**
   - Track material costs over time to identify inflation trends
   - Analyze labor shortages by skill type and location

3. **Decision Support:**
   - Predict resource needs based on project type/location
   - Optimize subcontractor usage based on past performance

4. **Cross-System Reporting:**
   - Generate unified reports from currently separate systems
   - Compare actual vs. projected costs across projects

**Scenario-Specific Applications:**
- Identify which skills are most frequently in shortage to guide hiring
- Analyze which countries/project types are most profitable
- Correlate weather data with project delays from historical records
- Combine third-party research data with internal project outcomes

**Potential Warehouse Structure:**
```
[Data Sources] → [ETL Process] → [Data Warehouse]
  │                │                  │
  │                │                  ├─ [Project Facts]
  ├─ Projects      │                  ├─ [Resource Dimensions]
  ├─ Workers       │                  ├─ [Financial Facts]
  ├─ Subcontractors│                  └─ [Time Dimensions]
  └─ Financials    └─ [Cleaning/Transformation]
```

---

## Conclusion

This assignment covered fundamental and advanced database concepts through practical scenarios. From basic relational operations to complex distributed architectures, the questions demonstrated how database principles apply to real-world situations like retail expansion and construction resource management. Proper implementation of these concepts can significantly improve data consistency, availability, and decision-making capabilities for organizations.