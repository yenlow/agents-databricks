CREATE OR REPLACE FUNCTION {catalog}.{schema}.get_recall_api(product_name STRING)
RETURNS STRING
COMMENT 'Returns product recalls from the Consumer Product Safety Commissions'
LANGUAGE PYTHON
AS $$
import requests

if product_name is None:
    return None

elif product_name.lower() == 'product':
    return None

elif len(product_name) == 0:
    return None

else:
    url = "https://www.saferproducts.gov/RestWebServices/Recall"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    data = {
        "ProductName": product_name,
        "format": "json"
    }
    response = requests.get(url, params=data, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if len(data) > 0:
            try:
                return data[0].get('Remedies')[0].get('Name')
            except:
                return None
        else:
            return None

    else:  # API request failed
        return None
$$;
