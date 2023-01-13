import flask
from flask import request, jsonify
import json
import os
from azure.cosmos import CosmosClient, PartitionKey
import uuid
import datetime
import time
import requests
from apscheduler.schedulers.background import BackgroundScheduler


# read config.json
with open('./config.json') as json_file:
    config = json.load(json_file)

# get config values
auth_token = config['auth_token']
db_endpoint = config['db_endpoint']
db_key = config['db_key']
middleware_url = config["middleware_url"]
watcher_url = config["watcher_url"]
admin_auth_token = config['admin_auth_token']
db_client = CosmosClient(db_endpoint, credential=db_key)
db = db_client.create_database_if_not_exists(id='cic-logger')
partition_key_path = PartitionKey(path="/id")

db_log_container = db.create_container_if_not_exists(
    id='cic-log',
    partition_key=partition_key_path,
)


def check_single_resource(url):
    start_time = time.time()
    response = requests.get(url)
    end_time = time.time()
    response_time = end_time - start_time
    if response.status_code == 200:
        return response_time
    else:
        return -1


def check_resource_helth():
    middleware_reponse_time = check_single_resource(middleware_url)
    watcher_response_time = check_single_resource(watcher_url)

    return {
        "id": str(uuid.uuid4()),
        "current_time": str(datetime.datetime.utcnow()),
        "watcher_response_time": watcher_response_time,
        "middlware_response_time": middleware_reponse_time
    }


def log_resource_health_to_db():
    status = check_resource_helth()
    db_log_container.create_item(body=status)


sched = BackgroundScheduler(daemon=True)
sched.add_job(log_resource_health_to_db, 'interval', minutes=5)
sched.start()

app = flask.Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return '''<h1>cic-logger works fine</h1>'''


@app.route('/status', methods=['GET'])
def status():
    return f'''<h1>Resource Health</h1>
    {check_resource_helth()}
    '''


@app.route('/trigger', methods=['GET'])
def trigger():
    if request.args.get("key") != admin_auth_token:
        return {"message": "Forbidden Request"}, 403

    log_resource_health_to_db()

    return "Success"


if __name__ == '__main__':
    app.run(host="0.0.0.0")
