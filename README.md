# Markov Decision Processes
*Assignment #4 - CS 7641 Machine Learning course - Charles Isbell & Michael Littman - Georgia Tech*

Please clone this git to a local project if you want to replicate the experiments reported in the assignment paper.

Virtual Environment
----
This project contains a virtual environment folder ```venv```. This folder contains all the files needed to create a virtual environment in which the project is supposed to run.

requirements.txt
----
This file contains all the necessary packages for this project. (Running ```pip install -r requirements.txt``` will install all the packages in your project's environment - should not be necessary if you are using the given ```venv```folder here)

frozen_lake.py and mountain_car.py
----
These files contain the Python scripts that are useful for this assignment. You simply have to run them. These scripts will:
- set up an environment (some application of a Markov Decision Process). Each file focuses on one of the two environments studied here. These environments are given by the OpenAI gym ilbrary, and are the [frozen lake environment](http://gym.openai.com/envs/FrozenLake-v0/) (which is a Grid world with a small number of states - i.e. 16) and the [mountain car environment](http://gym.openai.com/envs/MountainCar-v0/) (which is described by a continuous space of states that we will discretize, in order to get a large number of discrete states - i.e. 6461)
- run an implementation of the Value Iteration algorithm (returns an estimation of the optimal policy for the studied environment)
- run an implementation of the Policy Iteration algorithm (returns an estimation of the optimal policy for the studied environment)
- compare results (policies) and efficiencies of the Value and Policy Iteration algorithms
- run an implementation of a Q-learning algorithm (returns an estimation of the optimal policy for the studied environment)
- render some episodes of this Q-learning algorithm and plot a graph of the average reward versus number of episodes. The graphs will be stored in the folder named "Rewards_graphs". If you prefer, you could also simply change the second-to-last line (```plt.savefig('../Rewards_graphs/[NAME_OF_THE_GRAPH].jpg')```) on both scripts so that the graphs are stored in a directory of your choice.
The different algorithms have been adapted to each problem (which explains the differences between their implementation in both files), but apart from these minor differences they operate exactly the same way for both environments. Which means we can directly compare the results of these algorithms on the two environments, to analyze their behavior and efficiency when applied to different MDPs.

Using Google Cloud Compute Engine
----
If you wish to use Google Cloud Compute Engine there are some tweaks to do to the Python scripts and some commands to run in a shell.
The first thing you want to do is add a way to retrieve the plots generated by the scripts. We can use Google Cloud Storage API to upload the plots as png files to a Google bucket from where we can download them. For this add:

```
from google.cloud import storage


def upload_file(filename):
    """ Upload data to a bucket"""

    # Explicitly use service account credentials by specifying the private key
    # file.
    storage_client = storage.Client.from_service_account_json('/path/to/credentials.json')

    bucket = storage_client.get_bucket("ml-assignment-1-graphs")
    blob = bucket.blob(filename)
    blob.upload_from_filename(filename)

    #returns a public url
    return blob.public_url
```

to the Python scripts and the line ```upload_file("{}.png".format(title))``` after ```plt.savefig("{}.png".format(title))``` at the end of the Q-learning commands.

For this to work you will need a Google Cloud Platform account and create a project with an access to their Google Cloud Storage API. There you can generate a json file containint you credentials (download it and replace ```'/path/to/credentials.json'``` with the correct path to this file). Finally, create a bucket on Google Cloud Storage interface.

You will need to [create a repo in the GCP console](https://console.cloud.google.com/code/develop/repo?hl=fr&_ga=2.48832550.-1843116383.1569054709). Upload your local project to this repo by using Google's SDK and the commands ```git remote add google https://source.developers.google.com/p/[YOUR_PROJECT_ID]/r/[YOUR_BUCKET_NAME]``` (insert your project's ID and your bucket's name where needed), ```git commit -am "Commit title"``` and ```git push cloud master```. Or, you can also use the graphic interface on GCP to clone a github repository directly.

You will also need to activate Google Compute Engine API on your account.

Use this shell script to create a Compute Engine instance and run your python script in it (you can change the zone and machine-type tags values if needed) :

```
    gcloud compute instances create mdp-app-instance \
    --image-family=debian-9 \
    --image-project=debian-cloud \
    --machine-type=c2-standard-16 \
    --scopes userinfo-email,cloud-platform \
    --metadata-from-file startup-script=startup-script.sh \
    --zone us-central1-f \
    --tags http-server
```
    
You will need to have saved the ```startup-script.sh``` in the directory where you execute this command. This script is the following:

```set -v

# Talk to the metadata server to get the project id
PROJECTID=$(curl -s "http://metadata.google.internal/computeMetadata/v1/project/project-id" -H "Metadata-Flavor: Google")

# Install logging monitor. The monitor will automatically pickup logs sent to
# syslog.
curl -s "https://storage.googleapis.com/signals-agents/logging/google-fluentd-install.sh" | bash
service google-fluentd restart &

# Install dependencies from apt
apt-get update
apt-get install -yq \
    git build-essential supervisor python python-dev python-pip libffi-dev \
    libssl-dev

# Create a pythonapp user. The application will run as this user.
useradd -m -d /home/pythonapp pythonapp

# pip from apt is out of date, so make it update itself and install virtualenv.
pip install --upgrade pip virtualenv

# Get the source code from the Google Cloud Repository
# git requires $HOME and it's not set during the startup script.
export HOME=/root
git config --global credential.helper gcloud.sh
git clone https://source.developers.google.com/p/$PROJECTID/r/[YOUR_REPO_NAME] /opt/app

# Install app dependencies
virtualenv -p python3 /opt/app/venv
source /opt/app/venv/bin/activate
/opt/app/venv/bin/pip install -r /opt/app/requirements.txt

# Make sure the pythonapp user owns the application code
chown -R pythonapp:pythonapp /opt/app

# Configure supervisor to run the application.
cat >/etc/supervisor/conf.d/python-app.conf << EOF
[program:pythonapp]
directory=/opt/app
command=/opt/app/venv/bin/python /opt/app/[YOUR_PYTHON_SCRIPT_NAME].py
autostart=true
autorestart=true
user=pythonapp
# Environment variables ensure that the application runs inside of the
# configured virtualenv.
environment=VIRTUAL_ENV="/opt/app/venv",PATH="/opt/app/venv/bin",\
    HOME="/home/pythonapp",USER="pythonapp"
stdout_logfile=syslog
stderr_logfile=syslog
EOF

supervisorctl reread
supervisorctl update

# Application should now be running under supervisor
```

Replace ```[YOUR_REPO_NAME]``` with the name of the repo you created in the previous steps and ```[YOUR_PYTHON_SCRIPT_NAME]``` with the name of the python script you want to run. (This part can generate some errors due to the paths that can change from one project to another).

Once the Compute Engine instance is created, you can access its logs [here](https://console.cloud.google.com/logs?service=compute.googleapis.com&hl=fr&_ga=2.244371077.-1843116383.1569054709). And if everything goes well, the plots will start being added to your bucket.
