# 📊 Exploratory Data Analysis (EDA) Report

---

## 1. Data Overview

* **Rows:** 25,480
* **Columns:** 12
* **Target Variable:** `case_status` (Certified ≈ 66%, Denied ≈ 33%). The data is **imbalanced**.
* **Missing Values:** None.
* **Unique Identifier:** `case_id` (all values are unique and can be dropped from modeling).

---

## 2. Data Quality Issues

### Invalid Values
* `no_of_employees`: Contains 33 negative values, which is illogical.
* `prevailing_wage`: The minimum values are suspiciously low and require verification.

### Skewness & Outliers
* `no_of_employees`: The data is **right-skewed** with many high-value outliers.
* `yr_of_estab`: The data is **left-skewed** with some outliers below the lower bound.
* `prevailing_wage`: The data is **right-skewed** with high-value outliers.

---

## 3. Feature Observations

### 3.1 Categorical Features

* **`continent`**: Highly biased towards Asia.
    * **Suggested Grouping:** `Asia`, `Europe`, `North America`, `South America`, `Other`.
* **`unit_of_wage`**: `Yearly` accounts for approximately 90% of the data.
    * **Suggested:** Create a new binary feature, `is_yearly`.
* **`requires_job_training`**: A Chi-square test indicates no significant relationship with `case_status`.

### 3.2 Numerical Features

* **`no_of_employees`**: Outliers and negative values need to be cleaned and potentially transformed.
* **`yr_of_estab`**: Most companies were established between 2000 and 2005.
* **`prevailing_wage`**: Requires a **log transformation** or specific outlier treatment.

### 3.3 Correlation
* No multicollinearity was detected among the numerical features.

---

## 4. Key Insights from Univariate & Bivariate Analysis

### Continent vs. Case Status
* **65.3%** of certified applications are from Asia.
* **Europe** has the highest approval rate (79.2%), followed by Africa.
* Asia has the highest volume of applicants but a slightly lower approval rate compared to Europe.

### Education Level vs. Case Status
* Applicants with **Doctorate** and **Master's** degrees have higher approval rates.

### Work Experience vs. Case Status
* Applicants with experience have a **74.5%** approval rate.
* Applicants without experience have a **56%** approval rate.
* Work experience increases approval chances, but the effect size is moderate.

### Job Training Requirement vs. Case Status
* No significant effect on the approval rate was found (confirmed by Chi-square test).
* 88% of applicants do not require training.

### Unit of Wage vs. Case Status
* Applicants with an hourly pay rate have a **65%** denial rate.
* Applicants with a yearly pay rate have a **69%** approval rate.
* Weekly and monthly rates have approval rates that are slightly better than hourly but worse than yearly.

### Other Interesting Patterns
* **Region of Employment**: Approval rates are consistent across all regions.
* **Prevailing Wage Patterns**:
    * Higher wages correlate slightly with higher education.
    * **Unexpected:** Doctorate holders earn less on average than Master's holders.
    * **Unexpected:** Applicants without experience earn slightly more than experienced ones.
    * **Unexpected:** Europe offers lower wages compared to Asia.
    * Monthly contracts yield the highest wages; hourly contracts yield the lowest.
    * Smaller companies tend to offer higher wages.

---

## 5. Recommendations for Feature Engineering

* **Drop `case_id`** as it's irrelevant for prediction.
* **Handle negative `no_of_employees` values** (e.g., Drop or replace with a logical value).
* **Apply log transformation** to `no_of_employees` and `prevailing_wage` to address skewness.
* **Bin `continent`** into five groups (`Asia`, `Europe`, `North America`, `South America`, `Other`).
* **Create a binary feature `is_yearly`** from `unit_of_wage`.
* **Handle outliers** for all skewed numerical features.
* **Apply class balancing techniques** (e.g., SMOTE, class weights) to the target variable.
* **Encode categorical variables** into a numerical format for modeling.

---

## 6. Final Notes

The data shows clear biases: **continent**, **education**, and **work experience** significantly influence approval rates.

Two peculiar findings, **Doctorate holders earning less than Master's**, and **unexperienced applicants earning more than experienced ones**, require a more in-depth domain investigation before proceeding with modeling.

