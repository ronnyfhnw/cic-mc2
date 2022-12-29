import flask
from flask import request, jsonify
import json
import os
from azure.cosmos import CosmosClient, PartitionKey
import uuid
import datetime

# read config.json
with open('config.json') as json_file:
    config = json.load(json_file)

# get config values
auth_token = config['auth_token']
db_endpoint = config['db_endpoint']
db_key = config['db_key']
admin_auth_token = config['admin_auth_token']
db_client = CosmosClient(db_endpoint, credential=db_key)
db = db_client.create_database_if_not_exists(id='cic-watcher')
partition_key_path = PartitionKey(path="/id")

db_service_request_container = db.create_container_if_not_exists(
    id='azure-cognitive-service-requests',
    partition_key=partition_key_path,
    offer_throughput=400
)

db_configuration_container = db.create_container_if_not_exists(
    id='azure-cs-configuration',
    partition_key=partition_key_path,
    offer_throughput=400
)

# read the only document in the configuration container


def load_cs_config():
    db_azure_cs_config_result = db_configuration_container.read_all_items(
        max_item_count=1)

    for item in db_azure_cs_config_result:
        return item


azure_cs_config = load_cs_config()

app = flask.Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return '''<h1>CIC-Watcher works fine</h1>'''


@app.route('/api/v1/check-balance-allowance', methods=['GET'])
def api_check_balance_allowance():
    # check that auth token is in header and is valid
    if 'auth_token' not in request.headers:
        return "Error: No auth_token field provided. Please specify an auth_token."
    if request.headers['auth_token'] != auth_token:
        return "Error: auth_token is not valid."

    # get count of all documents in the service request container
    QUERY = "SELECT COUNT(1)"
    items = db_service_request_container.query_items(
        query=QUERY
    )

    total_requests = 0
    for item in items:
        total_requests = item['$1']

    # get the balance allowance from the configuration container
    total_balance = azure_cs_config['total_balance']
    price_per_request = azure_cs_config['price_per_request']

    if (total_requests * price_per_request < total_balance):
        # insert new document into service request container
        new_item = {
            "id": str(uuid.uuid4()),
            "route": "/api/v1/check-balance-allowance",
            "request_timestamp": str(datetime.datetime.now())
        }
        db_service_request_container.create_item(body=new_item)

        return jsonify(True)
    else:
        return jsonify(False)


@app.route('/api/v1/get-configuration', methods=['GET'])
def api_get_config():
    # check that auth token is in header and is valid
    if 'auth_token' not in request.headers:
        return "Error: No auth_token field provided. Please specify an auth_token."
    if request.headers['auth_token'] != admin_auth_token:
        return "Error: auth_token is not valid."

    azure_cs_config = load_cs_config()
    return jsonify(azure_cs_config)


@app.route('/api/v1/get-current-used-balance', methods=['GET'])
def api_get_current_balance():
    # check that auth token is in header and is valid
    if 'auth_token' not in request.headers:
        return "Error: No auth_token field provided. Please specify an auth_token."
    if request.headers['auth_token'] != admin_auth_token:
        return "Error: auth_token is not valid."

    # get count of all documents in the service request container
    QUERY = "SELECT COUNT(1)"
    items = db_service_request_container.query_items(
        query=QUERY
    )

    total_requests = 0
    for item in items:
        total_requests = item['$1']

    # get the balance allowance from the configuration container
    price_per_request = azure_cs_config['price_per_request']

    return jsonify(total_requests * price_per_request)


@ app.route("/api/v1/update-total-balance",  methods=['POST'])
def api_update_total():
    # check that auth token is in header and is valid
    if 'auth_token' not in request.headers:
        return "Error: No auth_token field provided. Please specify an auth_token."
    if request.headers['auth_token'] != admin_auth_token:
        return "Error: auth_token is not valid."

    # check that request body is valid
    if 'total_balance' not in request.json:
        return "Error: No total_balance field provided. Please specify a total_balance."

    # update the total balance in the configuration container
    azure_cs_config['total_balance'] = request.json['total_balance']

    db_configuration_container.upsert_item(body=azure_cs_config)

    return "Success: Configuration has been updated."


@ app.route("/api/v1/update-price-per-request",  methods=['POST'])
def api_update_price_per_req():
    # check that auth token is in header and is valid
    if 'auth_token' not in request.headers:
        return "Error: No auth_token field provided. Please specify an auth_token."
    if request.headers['auth_token'] != admin_auth_token:
        return "Error: auth_token is not valid."

    # check that request body is valid
    if 'price_per_request' not in request.json:
        return "Error: No price_per_request field provided. Please specify a price_per_request."

    # update the total balance in the configuration container
    azure_cs_config['price_per_request'] = request.json['price_per_request']

    db_configuration_container.upsert_item(body=azure_cs_config)

    return "Success: Configuration has been updated."


if __name__ == '__main__':
    app.run(host="0.0.0.0")
