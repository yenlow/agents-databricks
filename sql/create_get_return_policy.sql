CREATE OR REPLACE FUNCTION {catalog}.{schema}.get_return_policy()
RETURNS TABLE (policy STRING, policy_details STRING, last_updated DATE)
COMMENT 'Returns the details of the Return Policy'
LANGUAGE SQL
RETURN
    SELECT policy, policy_details, last_updated
    FROM retail_prod.agents.policies
    WHERE policy = 'Return Policy'
LIMIT 1;
