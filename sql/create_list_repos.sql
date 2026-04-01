CREATE OR REPLACE FUNCTION {catalog}.{schema}.list_repos(
  user STRING  COMMENT 'GitHub username'
)
RETURNS STRING
COMMENT 'Returns JSON array of public repos for a user'
RETURN http_request(
  conn => 'yen_github_conn',
  method => 'GET',
  path => concat('/users/', user, '/repos'),
  headers => map('Accept', "application/vnd.github+json")
).text;
