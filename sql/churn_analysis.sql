-- Query 1: Overall churn rate, based on the 0/1 Churn column.
SELECT COUNT(*) AS total_customers,
       SUM(Churn) AS churned_customers,
       ROUND(100.0 * AVG(Churn), 2) AS churn_rate_pct
FROM customers;

-- Query 2: Which contracts have the largest churn rate?
SELECT Contract,
       COUNT(*) AS customers,
       SUM(Churn) AS churned,
       ROUND(100.0 * AVG(Churn), 2) AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

-- Query 3: Does churn vary with how long customers have stayed?
SELECT CASE
           WHEN tenure < 12 THEN '00-11 months'
           WHEN tenure < 24 THEN '12-23 months'
           WHEN tenure < 48 THEN '24-47 months'
           ELSE '48+ months'
       END AS tenure_group,
       COUNT(*) AS customers,
       ROUND(100.0 * AVG(Churn), 2) AS churn_rate_pct
FROM customers
GROUP BY tenure_group
ORDER BY MIN(tenure);

-- Query 4: Internet service churn comparison.
SELECT InternetService,
       COUNT(*) AS customers,
       ROUND(100.0 * AVG(Churn), 2) AS churn_rate_pct
FROM customers
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;

-- Query 5: Billing amounts among customers who left vs stayed.
SELECT CASE WHEN Churn = 1 THEN 'Left' ELSE 'Stayed' END AS outcome,
       COUNT(*) AS customers,
       ROUND(AVG(MonthlyCharges), 2) AS average_monthly_charges,
       ROUND(AVG(TotalCharges), 2) AS average_total_charges
FROM customers
GROUP BY Churn
ORDER BY Churn DESC;
