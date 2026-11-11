# API and Rate Limits

## Authentication
API requests use a personal access token sent in the Authorization header. Tokens are created under Settings, then Developer, and are shown only once. A token has the same permissions as its owner.

## Rate limits
The API allows 100 requests per minute per token on the Team plan and 600 requests per minute on the Business plan. Exceeding the limit returns HTTP 429 with a Retry-After header.

## Pagination
List endpoints return 50 items per page by default and at most 200. Use the cursor value in the response to fetch the next page.

## Versioning
The current API version is v2. Version v1 stops working on 31 March 2027, and a migration guide is available in the developer section.
