-- SQL Feature Engineering for RecSys using Window Functions
-- Executable in DuckDB and ANSI-compliant SQL engines (Postgres, BigQuery, Spark SQL)

-- 1. Item-Level Cumulative Statistics (Popularity, CTR, CVR)
CREATE OR REPLACE TABLE item_features AS
SELECT
    item_id,
    MODE(category_id) AS category_id,
    COUNT(*) AS total_interactions,
    COUNT(CASE WHEN behavior_type = 'pv' THEN 1 END) AS item_pv_count,
    COUNT(CASE WHEN behavior_type = 'cart' THEN 1 END) AS item_cart_count,
    COUNT(CASE WHEN behavior_type = 'fav' THEN 1 END) AS item_fav_count,
    COUNT(CASE WHEN behavior_type = 'buy' THEN 1 END) AS item_buy_count,
    -- Smoothed conversion rates (Bayesian / Laplace prior)
    (COUNT(CASE WHEN behavior_type = 'cart' THEN 1 END) + 1.0) /
        (COUNT(CASE WHEN behavior_type = 'pv' THEN 1 END) + 20.0) AS item_pv_to_cart_rate,
    (COUNT(CASE WHEN behavior_type = 'buy' THEN 1 END) + 1.0) /
        (COUNT(CASE WHEN behavior_type = 'pv' THEN 1 END) + 50.0) AS item_pv_to_buy_rate
FROM train_events
GROUP BY item_id;

-- 2. User-Level Interaction Sequence with Window Functions
CREATE OR REPLACE TABLE user_event_windows AS
SELECT
    user_id,
    item_id,
    category_id,
    behavior_type,
    timestamp,
    -- Rank user interactions by time descending (1 = most recent)
    ROW_NUMBER() OVER (
        PARTITION BY user_id
        ORDER BY timestamp DESC
    ) AS seq_rank_desc,
    -- Time elapsed since previous interaction
    timestamp - LAG(timestamp, 1) OVER (
        PARTITION BY user_id
        ORDER BY timestamp ASC
    ) AS time_since_prev_event,
    -- Cumulative user actions up to current event
    COUNT(*) OVER (
        PARTITION BY user_id
        ORDER BY timestamp ASC
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS user_cum_actions
FROM raw_events;

-- 3. Session Identification Window Function (30-minute inactivity boundary)
CREATE OR REPLACE TABLE user_sessions AS
SELECT
    *,
    SUM(CASE WHEN time_since_prev_event > 1800 OR time_since_prev_event IS NULL THEN 1 ELSE 0 END) OVER (
        PARTITION BY user_id
        ORDER BY timestamp ASC
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS session_id
FROM user_event_windows;

-- 4. Session-Level Impression Candidate Units (Deduplicated within session)
CREATE OR REPLACE TABLE session_candidate_units AS
SELECT
    user_id,
    session_id,
    item_id,
    category_id,
    MIN(timestamp) AS session_timestamp,
    -- Funnel flags within session
    MAX(CASE WHEN behavior_type IN ('pv', 'cart', 'fav', 'buy') THEN 1 ELSE 0 END) AS is_clicked,
    MAX(CASE WHEN behavior_type = 'buy' THEN 1 ELSE 0 END) AS is_bought
FROM user_sessions
GROUP BY user_id, session_id, item_id, category_id;

-- 5. User Aggregated Profile Features
CREATE OR REPLACE TABLE user_features AS
SELECT
    user_id,
    COUNT(*) AS user_total_actions,
    COUNT(CASE WHEN behavior_type = 'pv' THEN 1 END) AS user_pv_count,
    COUNT(CASE WHEN behavior_type = 'buy' THEN 1 END) AS user_buy_count,
    COUNT(DISTINCT category_id) AS user_distinct_categories,
    COUNT(DISTINCT item_id) AS user_distinct_items
FROM raw_events
GROUP BY user_id;
