CREATE OR REPLACE FUNCTION {catalog}.{schema}.extract_product(text STRING)
RETURNS STRING
COMMENT 'Returns the product mentioned in issue_description'
LANGUAGE SQL
RETURN ai_extract(text, array('product')).product;
