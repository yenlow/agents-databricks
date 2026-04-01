CREATE OR REPLACE FUNCTION {catalog}.{schema}.get_requests_history(user_name STRING)
RETURNS TABLE (requests INT, issue_category STRING)
COMMENT "This takes a customer's name as an input and returns the number of requests per issue category"
LANGUAGE SQL
RETURN
    SELECT count(*) as requests, issue_category
    FROM retail_prod.agents.cust_service_data
    WHERE name = user_name
    GROUP BY issue_category;
