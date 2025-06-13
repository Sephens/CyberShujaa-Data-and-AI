# Comprehensive Solution to Question 1

## Part a) Relational Algebra Queries

### i. Borrower information for Hatton Cross branch

**Query:** Produce a relation that shows the borrower surname, borrower number and book titles for all borrowers at the Hatton Cross branch.

**Solution:**

```
π Borrower_Surname, Borrower_Number, Book_Title (σ Library_Branch = "Hatton Cross" (Loan_Record))
```

**Explanation:**
1. We first select (σ) all tuples from the Loan_Record relation where the Library_Branch is "Hatton Cross"
2. Then we project (π) only the attributes we need: Borrower_Surname, Borrower_Number, and Book_Title

**Resulting Relation:**
| Borrower_Surname | Borrower_Number | Book_Title       |
|------------------|-----------------|------------------|
| Lodge            | B580            | Ulysses          |
| Fisher           | B591            | Simon's Garden   |
| Fisher           | B591            | Coral Bridge     |

### ii. Borrowers from public libraries with holdings > 30,000

**Query:** Produce a relation that shows borrower surname and borrower number for all borrowers that borrowed books from public libraries which have holdings above 30,000.

**Solution:**

```
π Borrower_Surname, Borrower_Number (Loan_Record ⋈ (σ Library_Type = "Public" ∧ Holdings > 30000 (Library)))
```

**Explanation:**
1. First, we select (σ) libraries that are "Public" AND have holdings > 30,000 from the Library relation
2. Then we perform a natural join (⋈) between Loan_Record and the result from step 1
3. Finally, we project (π) only the Borrower_Surname and Borrower_Number attributes

**Resulting Relation:**
| Borrower_Surname | Borrower_Number |
|------------------|-----------------|
| Landis           | B602            |
| Choudhury        | B613            |
| Wu               | B624            |

## Part b) Deriving Results from Relational Algebra Queries

### i. Library branches with holdings ≤50,000 in South London

**Query:** 
```
π Library_Branch (σ Holdings ≤50000 ∧ District= "South London" (Library))
```

**Solution:**
1. Select libraries where Holdings ≤ 50,000 AND District = "South London"
2. From the Library table, the matching tuples are:
   - Lewisham (Holdings: 25,000)
   - Deptford (Holdings: 60,000) → Doesn't meet Holdings condition
3. Project only the Library_Branch attribute

**Result:**
| Library_Branch |
|----------------|
| Lewisham       |

### ii. Library branches not in Berkshire with holdings ≥100,000

**Query:**
```
π Library_Branch (σ District ≠ 'Berkshire' ∧ Holdings ≥ 100000(Library))
```

**Solution:**
1. Select libraries where District ≠ "Berkshire" AND Holdings ≥ 100,000
2. From the Library table, the matching tuples are:
   - New Cross (South London, University, 100,000)
   - Hatton Cross is in Berkshire → excluded by first condition
3. Project only the Library_Branch attribute

**Result:**
| Library_Branch |
|----------------|
| New Cross      |

### iii. Borrower surnames who borrowed "Coral Bridge"

**Query:**
```
π Borrower_surname (σ Book_Title = "Coral Bridge" (Loan_Record))
```

**Solution:**
1. Select loan records where Book_Title = "Coral Bridge"
2. From the Loan_Record table, the matching tuples are:
   - (New Cross, B580, Lodge, 1406, Coral Bridge)
   - (Hatton Cross, B591, Fisher, 1406, Coral Bridge)
3. Project only the Borrower_surname attribute

**Result:**
| Borrower_surname |
|------------------|
| Lodge            |
| Fisher           |

## Part c) Query Optimization

**Definition:**
Query optimization is the process of selecting the most efficient execution strategy for a given database query. The goal is to minimize resource usage (CPU, I/O, memory) and response time while producing the correct results.

**How Relational Algebra is Used:**
1. **Query Representation:** Queries are first translated into relational algebra expressions
2. **Transformation Rules:** Algebraic equivalences are used to rewrite queries into more efficient forms:
   - Commutativity and associativity of joins
   - Pushing selections down the expression tree
   - Combining projections
3. **Cost Estimation:** Different algebraic expressions are evaluated based on:
   - Size of intermediate relations
   - Available indexes
   - Join algorithms (nested loops, hash join, merge join)
4. **Plan Generation:** The optimizer generates multiple execution plans from the algebraic expressions and selects the one with lowest estimated cost

**Example Optimization:**
Original query: 
```
π A (σ C=1 (R ⋈ S))
```
Optimized version: 
```
π A (σ C=1 (R) ⋈ S)
```
By pushing the selection down, we reduce the size of R before the join operation.

## Part d) Relational Calculus

**Definition:**
Relational calculus is a non-procedural query language that describes what data to retrieve rather than how to retrieve it. It comes in two forms:
1. Tuple Relational Calculus (TRC)
2. Domain Relational Calculus (DRC)

**Main Features:**
1. **Declarative Nature:** Specifies what to find without specifying how to find it
2. **Based on Predicate Logic:** Uses logical expressions to define result tuples
3. **Variables:** 
   - TRC uses tuple variables that range over tuples
   - DRC uses domain variables that range over attribute values
4. **Quantifiers:**
   - Existential (∃): "there exists"
   - Universal (∀): "for all"
5. **Formulas:** Built from atoms using logical operators (∧, ∨, ¬, →)
6. **Safety:** Ensures queries return finite results through range restrictions
7. **Equivalence:** Relational algebra and calculus have equivalent expressive power

**Example (TRC):**
```
{ t | t ∈ Loan_Record ∧ t.Library_Branch = "Hatton Cross" }
```
This finds all tuples in Loan_Record where the branch is Hatton Cross.

**Comparison with Relational Algebra:**
- Calculus is declarative; algebra is procedural
- Calculus describes properties of result; algebra describes operations to produce result
- Both are formally equivalent in expressive power
- Algebra is closer to actual implementation in DBMS

This comprehensive solution covers all parts of Question 1 with detailed explanations and examples for each concept. The relational algebra operations are clearly shown with their corresponding results, and the theoretical aspects of query optimization and relational calculus are thoroughly explained.

